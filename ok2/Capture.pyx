import sys
import time

import cv2
import numpy as np
import psutil
import win32api
import win32con
import win32gui
import win32process
import win32ui
from qfluentwidgets import FluentIcon

import ctypes
import json
import os
import platform
import re
import threading
from enum import IntEnum
from ok.color.Color import is_close_to_pure_color
from ok.config.Config import Config
from ok.config.ConfigOption import ConfigOption
from ok.gui.Communicate import communicate
from ok.logging.Logger import get_logger
from ok.util.Handler import Handler

# This is an undocumented nFlag value for PrintWindow
PW_RENDERFULLCONTENT = 0x00000002
PBYTE = ctypes.POINTER(ctypes.c_ubyte)
WGC_NO_BORDER_MIN_BUILD = 20348
WGC_MIN_BUILD = 19041

logger = get_logger(__name__)


class CaptureException(Exception):
    pass


cdef class BaseCaptureMethod:
    name = "None"
    description = ""
    cdef public tuple _size
    cdef public object exit_event

    def __init__(self):
        self._size = (0, 0)
        self.exit_event = None

    def close(self):
        # Some capture methods don't need an initialization process
        pass

    @property
    def width(self):
        return self._size[0]

    @property
    def height(self):
        return self._size[1]

    cpdef object get_frame(self):
        cdef object frame
        if self.exit_event.is_set():
            return
        try:
            frame = self.do_get_frame()
            if frame is not None:
                self._size = (frame.shape[1], frame.shape[0])
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
            return frame
        except Exception as e:
            raise CaptureException(str(e)) from e

    def __str__(self):
        return f'{self.__class__.__name__}_{self.width}x{self.height}'

    def do_get_frame(self):
        pass

    def draw_rectangle(self):
        pass

    def clickable(self):
        pass

    def connected(self):
        pass

cdef class BaseWindowsCaptureMethod(BaseCaptureMethod):
    cdef public object _hwnd_window

    def __init__(self, hwnd_window: HwndWindow):
        super().__init__()
        self._hwnd_window = hwnd_window

    @property
    def hwnd_window(self):
        return self._hwnd_window

    @hwnd_window.setter
    def hwnd_window(self, hwnd_window):
        self._hwnd_window = hwnd_window

    def connected(self):
        logger.debug(f"check connected {self._hwnd_window}")
        return self.hwnd_window is not None and self.hwnd_window.exists

    def get_abs_cords(self, x, y):
        return self.hwnd_window.get_abs_cords(x, y)

    def clickable(self):
        return self.hwnd_window is not None and self.hwnd_window.visible

    def __str__(self):
        result = f'{self.__class__.__name__}_{self.width}x{self.height}'
        if self.hwnd_window is None:
            result += '_no_window'
        else:
            result += f'_{self.hwnd_window}'
        return result

def get_crop_point(frame_width, frame_height, target_width, target_height):
    x = round((frame_width - target_width) / 2)
    y = (frame_height - target_height) - x
    return x, y

cdef class WindowsGraphicsCaptureMethod(BaseWindowsCaptureMethod):
    name = "Windows Graphics Capture"
    description = "fast, most compatible, capped at 60fps"

    cdef object last_frame
    cdef double last_frame_time
    cdef object frame_pool
    cdef object item
    cdef object session
    cdef object cputex
    cdef object rtdevice
    cdef object dxdevice
    cdef object immediatedc
    cdef object evtoken
    cdef object last_size

    def __init__(self, hwnd_window: HwndWindow):
        super().__init__(hwnd_window)
        self.last_frame = None
        self.last_frame_time = time.time()
        self.frame_pool = None
        self.item = None
        self.session = None
        self.cputex = None
        self.rtdevice = None
        self.dxdevice = None
        self.immediatedc = None
        self.start_or_stop()

    cdef frame_arrived_callback(self, x, y):
        cdef object next_frame
        try:
            self.last_frame_time = time.time()
            next_frame = self.frame_pool.TryGetNextFrame()
            if next_frame is not None:
                self.last_frame = self.convert_dx_frame(next_frame)
            else:
                logger.warning('frame_arrived_callback TryGetNextFrame returned None')
        except Exception as e:
            logger.error(f"TryGetNextFrame error {e}")
            self.close()
            return

    cdef object convert_dx_frame(self, frame):
        if not frame:
            # logger.warning('convert_dx_frame self.last_dx_frame is none')
            return None
        cdef bint need_reset_framepool = False
        if frame.ContentSize.Width != self.last_size.Width or frame.ContentSize.Height != self.last_size.Height:
            need_reset_framepool = True
            self.last_size = frame.ContentSize

        if need_reset_framepool:
            logger.info('need_reset_framepool')
            self.reset_framepool(frame.ContentSize)
            return
        cdef bint need_reset_device = False

        cdef object tex = None

        cdef object cputex = None
        cdef object desc = None
        cdef object mapinfo = None
        cdef object img = None
        try:
            from ok.capture.windows import d3d11
            from ok.rotypes.Windows.Graphics.DirectX.Direct3D11 import IDirect3DDxgiInterfaceAccess
            from ok.rotypes.roapi import GetActivationFactory
            tex = frame.Surface.astype(IDirect3DDxgiInterfaceAccess).GetInterface(
                d3d11.ID3D11Texture2D.GUID).astype(d3d11.ID3D11Texture2D)
            desc = tex.GetDesc()
            desc2 = d3d11.D3D11_TEXTURE2D_DESC()
            desc2.Width = desc.Width
            desc2.Height = desc.Height
            desc2.MipLevels = desc.MipLevels
            desc2.ArraySize = desc.ArraySize
            desc2.Format = desc.Format
            desc2.SampleDesc = desc.SampleDesc
            desc2.Usage = d3d11.D3D11_USAGE_STAGING
            desc2.CPUAccessFlags = d3d11.D3D11_CPU_ACCESS_READ
            desc2.BindFlags = 0
            desc2.MiscFlags = 0
            cputex = self.dxdevice.CreateTexture2D(ctypes.byref(desc2), None)
            self.immediatedc.CopyResource(cputex, tex)
            mapinfo = self.immediatedc.Map(cputex, 0, d3d11.D3D11_MAP_READ, 0)
            img = np.ctypeslib.as_array(ctypes.cast(mapinfo.pData, PBYTE),
                                        (desc.Height, mapinfo.RowPitch // 4, 4))[
                  :, :desc.Width].copy()
            self.immediatedc.Unmap(cputex, 0)
            # logger.debug(f'frame latency {(time.time() - start):.3f} {(time.time() - dx_time):.3f}')
            return img
        except OSError as e:
            if e.winerror == d3d11.DXGI_ERROR_DEVICE_REMOVED or e.winerror == d3d11.DXGI_ERROR_DEVICE_RESET:
                need_reset_framepool = True
                need_reset_device = True
                logger.error('convert_dx_frame win error', e)
            else:
                raise e
        finally:
            if tex is not None:
                tex.Release()
            if cputex is not None:
                cputex.Release()
        if need_reset_framepool:
            self.reset_framepool(frame.ContentSize, need_reset_device)
            return self.get_frame()

    @property
    def hwnd_window(self):
        return self._hwnd_window

    @hwnd_window.setter
    def hwnd_window(self, hwnd_window):
        self._hwnd_window = hwnd_window
        self.start_or_stop()

    def connected(self):
        return self.hwnd_window is not None and self.hwnd_window.exists and self.frame_pool is not None

    def start_or_stop(self, capture_cursor=False):
        if self.hwnd_window.hwnd and self.hwnd_window.exists and self.frame_pool is None:
            try:
                from ok.capture.windows import d3d11
                from ok.rotypes import IInspectable
                from ok.rotypes.Windows.Foundation import TypedEventHandler
                from ok.rotypes.Windows.Graphics.Capture import Direct3D11CaptureFramePool, IGraphicsCaptureItemInterop, \
                    IGraphicsCaptureItem, GraphicsCaptureItem
                from ok.rotypes.Windows.Graphics.DirectX import DirectXPixelFormat
                from ok.rotypes.Windows.Graphics.DirectX.Direct3D11 import IDirect3DDevice, \
                    CreateDirect3D11DeviceFromDXGIDevice, \
                    IDirect3DDxgiInterfaceAccess
                from ok.rotypes.roapi import GetActivationFactory
                logger.info('init windows capture')
                interop = GetActivationFactory('Windows.Graphics.Capture.GraphicsCaptureItem').astype(
                    IGraphicsCaptureItemInterop)
                self.rtdevice = IDirect3DDevice()
                self.dxdevice = d3d11.ID3D11Device()
                self.immediatedc = d3d11.ID3D11DeviceContext()
                self.create_device()
                item = interop.CreateForWindow(self.hwnd_window.hwnd, IGraphicsCaptureItem.GUID)
                self.item = item
                self.last_size = item.Size
                delegate = TypedEventHandler(GraphicsCaptureItem, IInspectable).delegate(
                    self.close)
                self.evtoken = item.add_Closed(delegate)
                self.frame_pool = Direct3D11CaptureFramePool.CreateFreeThreaded(self.rtdevice,
                                                                                DirectXPixelFormat.B8G8R8A8UIntNormalized,
                                                                                1, item.Size)
                self.session = self.frame_pool.CreateCaptureSession(item)
                pool = self.frame_pool
                pool.add_FrameArrived(
                    TypedEventHandler(Direct3D11CaptureFramePool, IInspectable).delegate(
                        self.frame_arrived_callback))
                self.session.IsCursorCaptureEnabled = capture_cursor
                if WINDOWS_BUILD_NUMBER >= WGC_NO_BORDER_MIN_BUILD:
                    self.session.IsBorderRequired = False
                self.session.StartCapture()
                return True
            except Exception as e:
                logger.error(f'start_or_stop failed: {self.hwnd_window}', exception=e)
                return False
        elif not self.hwnd_window.exists and self.frame_pool is not None:
            self.close()
            return False
        return self.hwnd_window.exists

    def create_device(self):
        from ok.capture.windows import d3d11
        from ok.rotypes.Windows.Graphics.DirectX.Direct3D11 import CreateDirect3D11DeviceFromDXGIDevice
        d3d11.D3D11CreateDevice(
            None,
            d3d11.D3D_DRIVER_TYPE_HARDWARE,
            None,
            d3d11.D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            0,
            d3d11.D3D11_SDK_VERSION,
            ctypes.byref(self.dxdevice),
            None,
            ctypes.byref(self.immediatedc)
        )
        self.rtdevice = CreateDirect3D11DeviceFromDXGIDevice(self.dxdevice)
        self.evtoken = None

    def close(self):
        logger.info('destroy windows capture')
        if self.frame_pool is not None:
            self.frame_pool.Close()
            self.frame_pool = None
        if self.session is not None:
            self.session.Close()  # E_UNEXPECTED ???
            self.session = None
        self.item = None
        if self.rtdevice:
            self.rtdevice.Release()
        if self.dxdevice:
            self.dxdevice.Release()
        if self.cputex:
            self.cputex.Release()

    cpdef object do_get_frame(self):
        cdef object frame
        cdef double latency
        if self.start_or_stop():
            frame = self.last_frame
            self.last_frame = None
            if frame is None:
                if time.time() - self.last_frame_time > 10:
                    logger.warning(f'no frame for 10 sec, try to restart')
                    self.close()
                    self.last_frame_time = time.time()
                    return self.do_get_frame()
                else:
                    return None
            latency = time.time() - self.last_frame_time

            frame = self.crop_image(frame)

            if frame is not None:
                new_height, new_width = frame.shape[:2]
                if new_width <= 0 or new_width <= 0:
                    logger.warning(f"get_frame size <=0 {new_width}x{new_height}")
                    frame = None
            if latency > 2:
                logger.warning(f"latency too large return None frame: {latency}")
                return None
            else:
                # logger.debug(f'frame latency: {latency}')
                return frame

    def reset_framepool(self, size, reset_device=False):
        logger.info(f'reset_framepool')
        from ok.rotypes.Windows.Graphics.DirectX import DirectXPixelFormat
        if reset_device:
            self.create_device()
        self.frame_pool.Recreate(self.rtdevice,
                                 DirectXPixelFormat.B8G8R8A8UIntNormalized, 2, size)

    def crop_image(self, frame):
        if frame is not None:
            x, y = get_crop_point(frame.shape[1], frame.shape[0], self.hwnd_window.width, self.hwnd_window.height)
            if x > 0 or y > 0:
                frame = crop_image(frame, x, y)
        return frame

def crop_image(image, border, title_height):
    # Load the image
    # Image dimensions
    height, width = image.shape[:2]

    # Calculate the coordinates for the bottom-right corner
    x2 = width - border
    y2 = height - border

    # Crop the image
    cropped_image = image[title_height:y2, border:x2]

    # print(f"cropped image: {title_height}-{y2}, {border}-{x2} {cropped_image.shape}")
    #
    # cv2.imshow('Image Window', cropped_image)
    #
    # # Wait for any key to be pressed before closing the window
    # cv2.waitKey(0)

    return cropped_image

WINDOWS_BUILD_NUMBER = int(platform.version().split(".")[-1]) if sys.platform == "win32" else -1

def windows_graphics_available():
    logger.info(
        f"check available WINDOWS_BUILD_NUMBER:{WINDOWS_BUILD_NUMBER} >= {WGC_MIN_BUILD} {WINDOWS_BUILD_NUMBER >= WGC_MIN_BUILD}")
    if WINDOWS_BUILD_NUMBER >= WGC_MIN_BUILD:
        try:
            from ok.rotypes import idldsl
            from ok.rotypes.roapi import GetActivationFactory
            from ok.rotypes.Windows.Graphics.Capture import IGraphicsCaptureItemInterop
            GetActivationFactory('Windows.Graphics.Capture.GraphicsCaptureItem').astype(
                IGraphicsCaptureItemInterop)
            return True
        except Exception as e:
            logger.error(f'check available failed: {e}', exception=e)
            return False

def is_blank(image):
    """
    BitBlt can return a balnk buffer. Either because the target is unsupported,
    or because there's two windows of the same name for the same executable.
    """
    return not image.any()

cdef bint render_full
render_full = False

cdef class BitBltCaptureMethod(BaseWindowsCaptureMethod):
    name = "BitBlt"
    short_description = "fastest, least compatible"
    description = (
            "\nThe best option when compatible. But it cannot properly record "
            + "\nOpenGL, Hardware Accelerated or Exclusive Fullscreen windows. "
            + "\nThe smaller the selected region, the more efficient it is. "
    )

    cpdef object do_get_frame(self):
        cdef int x, y
        if self.hwnd_window.real_x_offset != 0 or self.hwnd_window.real_y_offset != 0:
            x = self.hwnd_window.real_x_offset
            y = self.hwnd_window.real_y_offset
        else:
            x, y = get_crop_point(self.hwnd_window.window_width, self.hwnd_window.window_height,
                                  self.hwnd_window.width, self.hwnd_window.height)
        return bit_blt_capture_frame(self.hwnd_window.hwnd, x,
                                     y,
                                     self.hwnd_window.real_width or self.hwnd_window.width,
                                     self.hwnd_window.real_height or self.hwnd_window.height,
                                     render_full)

    def test_exclusive_full_screen(self):
        frame = self.do_get_frame()
        if frame is None:
            logger.error(f'Failed to test_exclusive_full_screen {self.hwnd_window}')
            return False
        return True

    def test_is_not_pure_color(self):
        frame = self.do_get_frame()
        if frame is None:
            logger.error(f'Failed to test_is_not_pure_color frame is None {self.hwnd_window}')
            return False
        else:
            if is_close_to_pure_color(frame):
                logger.error(f'Failed to test_is_not_pure_color failed {self.hwnd_window}')
                return False
            else:
                return True

cdef object bit_blt_capture_frame(object hwnd, int border, int title_height, int width, int height,
                                  bint _render_full_content=False):
    if hwnd is None:
        return None
    if width <= 0 or height <= 0:
        return None
    cdef double start
    start = time.time()
    cdef object image
    image = None

    cdef int x, y
    x = border
    y = title_height

    cdef object dc_object, bitmap, window_dc, compatible_dc
    try:
        window_dc = win32gui.GetWindowDC(hwnd)
        dc_object = win32ui.CreateDCFromHandle(window_dc)

        # Causes a 10-15x performance drop. But allows recording hardware accelerated windows
        if _render_full_content:
            ctypes.windll.user32.PrintWindow(hwnd, dc_object.GetSafeHdc(), PW_RENDERFULLCONTENT)

        # On Windows there is a shadow around the windows that we need to account for.
        # left_bounds, top_bounds = 3, 0
        compatible_dc = dc_object.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dc_object, width, height)

        compatible_dc.SelectObject(bitmap)
        compatible_dc.BitBlt(
            (0, 0),
            (width, height),
            dc_object,
            (x, y),
            win32con.SRCCOPY,
        )
        image = np.frombuffer(bitmap.GetBitmapBits(True), dtype=np.uint8)
    except:
        # Invalid handle or the window was closed while it was being manipulated
        return None

    if is_blank(image):
        image = None
    else:
        image.shape = (height, width, BGRA_CHANNEL_COUNT)

    # Cleanup DC and handle
    try_delete_dc(dc_object)
    try_delete_dc(compatible_dc)
    win32gui.ReleaseDC(hwnd, window_dc)
    win32gui.DeleteObject(bitmap.GetHandle())
    return image

cdef class HwndWindow:
    cdef public object app_exit_event, stop_event, hwnd, mute_option, thread, device_manager
    cdef public str exe_name, title, exe_full_path, hwnd_class, _hwnd_title
    cdef public int player_id, window_width, window_height, x, y, width, height, frame_width, frame_height, real_width, real_height, real_x_offset, real_y_offset
    cdef public bint visible, exists, pos_valid
    cdef public double scaling, frame_aspect_ratio
    cdef public list monitors_bounds

    def __init__(self, exit_event, title, exe_name=None, frame_width=0, frame_height=0, player_id=-1, hwnd_class=None,
                 global_config=None, device_manager=None):
        super().__init__()
        logger.info(f'HwndWindow init title:{title} player_id:{player_id} exe_name:{exe_name} hwnd_class:{hwnd_class}')
        self.app_exit_event = exit_event
        self.exe_name = exe_name
        self.device_manager = device_manager
        self.title = title
        self.stop_event = threading.Event()
        self.visible = False
        self.player_id = player_id
        self.window_width = 0
        self.window_height = 0
        self.visible = True
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.hwnd = None
        self.frame_width = 0
        self.frame_height = 0
        self.exists = False
        self.title = None
        self.exe_full_path = None
        self.real_width = 0
        self.real_height = 0
        self.real_x_offset = 0
        self.real_y_offset = 0
        self.scaling = 1.0
        self.frame_aspect_ratio = 0
        self.hwnd_class = hwnd_class
        self.pos_valid = False
        self._hwnd_title = ""
        self.monitors_bounds = get_monitors_bounds()
        mute_config_option = ConfigOption('Game Sound', {
            'Mute Game while in Background': False
        }, validator=self.validate_mute_config, icon=FluentIcon.MUTE)
        self.mute_option = global_config.get_config(mute_config_option)
        self.update_window(title, exe_name, frame_width, frame_height, player_id, hwnd_class)
        self.thread = threading.Thread(target=self.update_window_size, name="update_window_size")
        self.thread.start()

    def validate_mute_config(self, key, value):
        if key == 'Mute Game while in Background' and not value and self.hwnd:
            logger.info('unmute game because option is turned off')
            set_mute_state(self.hwnd, 0)
        return True, None

    def stop(self):
        self.stop_event.set()

    def update_window(self, title, exe_name, frame_width, frame_height, player_id=-1, hwnd_class=None):
        self.player_id = player_id
        self.title = title
        self.exe_name = exe_name
        self.update_frame_size(frame_width, frame_height)
        self.hwnd_class = hwnd_class

    def update_frame_size(self, width, height):
        logger.debug(f"update_frame_size:{self.frame_width}x{self.frame_height} to {width}x{height}")
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            if width > 0 and height > 0:
                self.frame_aspect_ratio = width / height
                logger.debug(f"HwndWindow: frame ratio: width: {width}, height: {height}")
        self.hwnd = None
        self.do_update_window_size()

    def update_window_size(self):
        while not self.app_exit_event.is_set() and not self.stop_event.is_set():
            self.do_update_window_size()
            time.sleep(0.2)
        if self.hwnd and self.mute_option.get('Mute Game while in Background'):
            logger.info(f'exit reset mute state to 0')
            set_mute_state(self.hwnd, 0)

    def get_abs_cords(self, x, y):
        return self.x + x, self.y + y

    def do_update_window_size(self):
        try:
            visible, x, y, window_width, window_height, width, height, scaling = self.visible, self.x, self.y, self.window_width, self.window_height, self.width, self.height, self.scaling
            if self.hwnd is None:
                name, self.hwnd, self.exe_full_path, self.real_x_offset, self.real_y_offset, self.real_width, self.real_height = find_hwnd(
                    self.title,
                    self.exe_name,
                    self.frame_width, self.frame_height, player_id=self.player_id, class_name=self.hwnd_class)
                if self.hwnd is not None:
                    logger.info(
                        f'found hwnd {self.hwnd} {self.exe_full_path} {win32gui.GetClassName(self.hwnd)} real:{self.real_x_offset},{self.real_y_offset},{self.real_width},{self.real_height}')
                self.exists = self.hwnd is not None
            if self.hwnd is not None:
                self.exists = win32gui.IsWindow(self.hwnd)
                if self.exists:
                    visible = is_foreground_window(self.hwnd)
                    x, y, window_width, window_height, width, height, scaling = get_window_bounds(
                        self.hwnd)
                    if self.frame_aspect_ratio != 0 and height != 0:
                        window_ratio = width / height
                        if window_ratio < self.frame_aspect_ratio:
                            cropped_window_height = int(width / self.frame_aspect_ratio)
                            height = cropped_window_height
                    pos_valid = check_pos(x, y, width, height, self.monitors_bounds)
                    if not pos_valid and pos_valid != self.pos_valid and self.device_manager.executor is not None:
                        if self.device_manager.executor.pause():
                            logger.error(f'ok.gui.executor.pause pos_invalid: {x, y, width, height}')
                            communicate.notification.emit('Paused because game window is minimized or out of screen!',
                                                          None,
                                                          True, True)
                    if pos_valid != self.pos_valid:
                        self.pos_valid = pos_valid
                else:
                    if self.device_manager.executor is not None and self.device_manager.executor.pause():
                        communicate.notification.emit('Paused because game exited', None, True, True)
                    self.hwnd = None
                changed = False
                if visible != self.visible:
                    self.visible = visible
                    changed = True
                    self.handle_mute()
                if (window_width != self.window_width or window_height != self.window_height or
                    x != self.x or y != self.y or width != self.width or height != self.height or scaling != self.scaling) and (
                        (x >= -1 and y >= -1) or self.visible):
                    self.x, self.y, self.window_width, self.window_height, self.width, self.height, self.scaling = x, y, window_width, window_height, width, height, scaling
                    changed = True
                if changed:
                    logger.info(
                        f"do_update_window_size changed,visible:{self.visible} x:{self.x} y:{self.y} window:{self.width}x{self.height} self.window:{self.window_width}x{self.window_height} real:{self.real_width}x{self.real_height}")
                    communicate.window.emit(self.visible, self.x + self.real_x_offset, self.y + self.real_y_offset,
                                            self.window_width, self.window_height,
                                            self.width,
                                            self.height, self.scaling)
        except Exception as e:
            logger.error(f"do_update_window_size exception", e)

    def handle_mute(self):
        if self.hwnd and self.mute_option.get('Mute Game while in Background'):
            set_mute_state(self.hwnd, 0 if self.visible else 1)

    def frame_ratio(self, size):
        if self.frame_width > 0 and self.width > 0:
            return int(size / self.frame_width * self.width)
        else:
            return size

    @property
    def hwnd_title(self):
        if not self._hwnd_title:
            if self.hwnd:
                self._hwnd_title = win32gui.GetWindowText(self.hwnd)
        return self._hwnd_title

    def __str__(self) -> str:
        return str(
            f"title_{self.title}_{self.exe_name}_{self.width}x{self.height}_{self.hwnd}_{self.exists}_{self.visible}")

def check_pos(x, y, width, height, monitors_bounds):
    return width >= 0 and height >= 0 and is_window_in_screen_bounds(x, y, width, height, monitors_bounds)

def get_monitors_bounds():
    monitors_bounds = []
    monitors = win32api.EnumDisplayMonitors()
    for monitor in monitors:
        monitor_info = win32api.GetMonitorInfo(monitor[0])
        monitor_rect = monitor_info['Monitor']
        monitors_bounds.append(monitor_rect)
    return monitors_bounds

def is_window_in_screen_bounds(window_left, window_top, window_width, window_height, monitors_bounds):
    window_right, window_bottom = window_left + window_width, window_top + window_height

    for monitor_rect in monitors_bounds:
        monitor_left, monitor_top, monitor_right, monitor_bottom = monitor_rect

        # Check if the window is within the monitor bounds
        if (window_left >= monitor_left and window_top >= monitor_top and
                window_right <= monitor_right and window_bottom <= monitor_bottom):
            return True

    return False

def find_hwnd(title, exe_name, frame_width, frame_height, player_id=-1, class_name=None):
    results = []
    if exe_name is None and title is None:
        return None, None, None, 0, 0, 0, 0
    frame_aspect_ratio = frame_width / frame_height if frame_height != 0 else 0

    def callback(hwnd, lParam):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            text = win32gui.GetWindowText(hwnd)
            if title:
                if isinstance(title, str):
                    if title != text:
                        return True
                elif not re.search(title, text):
                    return True
            name, full_path, cmdline = get_exe_by_hwnd(hwnd)
            if not name:
                return True
            x, y, _, _, width, height, scaling = get_window_bounds(
                hwnd)
            ret = (hwnd, full_path, width, height, x, y, text)
            if exe_name:
                if name != exe_name and exe_name != full_path:
                    return True
            if player_id != -1:
                if player_id != get_player_id_from_cmdline(cmdline):
                    logger.debug(
                        f'player id check failed,cmdline {cmdline} {get_player_id_from_cmdline(cmdline)} != {player_id}')
                    return True
            if class_name is not None:
                if win32gui.GetClassName(hwnd) != class_name:
                    return True
            results.append(ret)
        return True

    win32gui.EnumWindows(callback, None)
    if len(results) > 0:
        logger.info(f'find_hwnd {results}')
        biggest = None
        for result in results:
            if biggest is None or (result[2] * result[3]) > biggest[2] * biggest[3]:
                biggest = result
        x_offset = 0
        y_offset = 0
        real_width = 0
        real_height = 0
        if frame_aspect_ratio != 0:
            real_width, real_height = biggest[2], biggest[3]
            matching_child = enum_child_windows(biggest, frame_aspect_ratio)
            if matching_child is not None:
                x_offset, y_offset, real_width, real_height = matching_child
            logger.info(
                f'find_hwnd {frame_width, frame_height} {biggest} {x_offset, y_offset, real_width, real_height}')
        return biggest[6], biggest[0], biggest[1], x_offset, y_offset, real_width, real_height

    return None, None, None, 0, 0, 0, 0

def get_mute_state(hwnd):
    from pycaw.api.audioclient import ISimpleAudioVolume
    from pycaw.utils import AudioUtilities
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        if session.Process and session.Process.pid == pid:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            return volume.GetMute()
    return 0

# Function to get the mute state
def set_mute_state(hwnd, mute):
    from pycaw.api.audioclient import ISimpleAudioVolume
    from pycaw.utils import AudioUtilities
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        if session.Process and session.Process.pid == pid:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            volume.SetMute(mute, None)  # 0 to unmute, 1 to mute
            break

def get_player_id_from_cmdline(cmdline):
    for i in range(len(cmdline)):
        if i != 0:
            if cmdline[i].isdigit():
                return int(cmdline[i])
    for i in range(len(cmdline)):
        if i != 0:
            value = re.search(r'index=(\d+)', cmdline[i])
            # Return the value if it exists, otherwise return None
            if value is not None:
                return int(value.group(1))
    return 0

def enum_child_windows(biggest, frame_aspect_ratio):
    ratio_match = []
    """
    A function to enumerate all child windows of the given parent window handle
    and print their handle and window title.
    """

    def child_callback(hwnd, _):
        visible = win32gui.IsWindowVisible(hwnd)
        parent = win32gui.GetParent(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        real_width = rect[2] - rect[0]
        real_height = rect[3] - rect[1]
        logger.info(f'find_hwnd child_callback {visible} {biggest[0]} {parent} {rect} {real_width} {real_height}')
        if visible and parent == biggest[0]:
            ratio = real_width / real_height
            difference = abs(ratio - frame_aspect_ratio)
            support = difference <= 0.01 * frame_aspect_ratio
            percent = (real_width * real_height) / (biggest[2] * biggest[3])
            if support and percent >= 0.7:
                x_offset = rect[0] - biggest[4]
                y_offset = rect[1] - biggest[5]
                ratio_match.append((x_offset, y_offset, real_width, real_height))
        return True

    win32gui.EnumChildWindows(biggest[0], child_callback, None)
    if len(ratio_match) > 0:
        return ratio_match[0]

def get_exe_by_hwnd(hwnd):
    # Get the process ID associated with the window
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        # Get the process name and executable path
        if pid > 0:
            process = psutil.Process(pid)
            return process.name(), process.exe(), process.cmdline()
        else:
            return None, None, None
    except Exception as e:
        logger.error('get_exe_by_hwnd error', e)
        return None, None, None

# orignal https://github.com/Toufool/AutoSplit/blob/master/src/capture_method/DesktopDuplicationCaptureMethod.py
cdef class DesktopDuplicationCaptureMethod(BaseWindowsCaptureMethod):
    name = "Direct3D Desktop Duplication"
    short_description = "slower, bound to display"
    description = (
            "\nDuplicates the desktop using Direct3D. "
            + "\nIt can record OpenGL and Hardware Accelerated windows. "
            + "\nAbout 10-15x slower than BitBlt. Not affected by window size. "
            + "\nOverlapping windows will show up and can't record across displays. "
            + "\nThis option may not be available for hybrid GPU laptops, "
            + "\nsee D3DDD-Note-Laptops.md for a solution. "
    )
    cdef object desktop_duplication

    def __init__(self, hwnd_window: HwndWindow):
        super().__init__(hwnd_window)
        import d3dshot
        self.desktop_duplication = d3dshot.create(capture_output="numpy")

    cpdef object do_get_frame(self):

        hwnd = self.hwnd_window.hwnd
        if hwnd is None:
            return None

        hmonitor = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        if not hmonitor:
            return None

        self.desktop_duplication.display = find_display(hmonitor, self.desktop_duplication.displays)

        cdef int left, top, right, bottom
        cdef object screenshot
        left = self.hwnd_window.x
        top = self.hwnd_window.y
        right = left + self.hwnd_window.width
        bottom = top + self.hwnd_window.height
        screenshot = self.desktop_duplication.screenshot((left, top, right, bottom))
        if screenshot is None:
            return None
        return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    def close(self):
        if self.desktop_duplication is not None:
            self.desktop_duplication.stop()

cdef find_display(hmonitor, displays):
    for display in displays:
        if display.hmonitor == hmonitor:
            return display
    raise ValueError("Display not found")

DWMWA_EXTENDED_FRAME_BOUNDS = 9
MAXBYTE = 255
"""How many channels in a BGR image"""
cdef int BGRA_CHANNEL_COUNT
BGRA_CHANNEL_COUNT = 4
"""How many channels in a BGRA image"""


class ImageShape(IntEnum):
    Y = 0
    X = 1
    Channels = 2


class ColorChannel(IntEnum):
    Blue = 0
    Green = 1
    Red = 2
    Alpha = 3


def decimal(value: float):
    # Using ljust instead of :2f because of python float rounding errors
    return f"{int(value * 100) / 100}".ljust(4, "0")

def is_digit(value: str | int | None):
    """Checks if `value` is a single-digit string from 0-9."""
    if value is None:
        return False
    try:
        return 0 <= int(value) <= 9  # noqa: PLR2004
    except (ValueError, TypeError):
        return False

def is_valid_hwnd(hwnd: int):
    """Validate the hwnd points to a valid window and not the desktop or whatever window obtained with `""`."""
    if not hwnd:
        return False
    if sys.platform == "win32":
        return bool(win32gui.IsWindow(hwnd) and win32gui.GetWindowText(hwnd))
    return True

def try_delete_dc(dc):
    try:
        dc.DeleteDC()
    except win32ui.error:
        pass

cdef class ADBCaptureMethod(BaseCaptureMethod):
    name = "ADB command line Capture"
    description = "use the adb screencap command, slow but works when in background/minimized, takes 300ms per frame"
    cdef bint _connected
    cdef object device_manager

    def __init__(self, device_manager, exit_event, width=0, height=0):
        super().__init__()
        self.exit_event = exit_event
        self._connected = (width != 0 and height != 0)
        self.device_manager = device_manager

    cpdef object do_get_frame(self):
        return self.screencap()

    cdef object screencap(self):
        if self.exit_event.is_set():
            return None
        cdef object frame
        frame = self.device_manager.do_screencap(self.device_manager.device)
        if frame is not None:
            self._connected = True
        else:
            self._connected = False
        return frame

    def connected(self):
        if not self._connected and self.device_manager.device is not None:
            self.screencap()
        return self._connected and self.device_manager.device is not None

cdef class ImageCaptureMethod(BaseCaptureMethod):
    name = "Image capture method "
    description = "for debugging"
    cdef list images

    def __init__(self, images):
        super().__init__()
        self.images = []
        self.set_images(images)

    def set_images(self, images):
        self.images = list(reversed(images))
        self.get_frame()  # fill size
        self.images = list(reversed(images))

    cpdef object do_get_frame(self):
        cdef str image_path
        if len(self.images) > 0:
            image_path = self.images.pop()
            if image_path:
                frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                return frame

    def connected(self):
        return True


class DeviceManager:

    def __init__(self, app_config, exit_event=None, global_config=None):
        logger.info('__init__ start')
        self._device = None
        self._adb = None
        self.executor = None
        self.global_config = global_config
        self._adb_lock = threading.Lock()
        supported_resolution = app_config.get(
            'supported_resolution', {})
        self.supported_ratio = parse_ratio(supported_resolution.get('ratio'))
        self.windows_capture_config = app_config.get('windows')
        self.adb_capture_config = app_config.get('adb')
        self.debug = app_config.get('debug')
        self.interaction = None
        self.device_dict = {}
        self.exit_event = exit_event
        self.resolution_dict = {}
        if self.windows_capture_config is not None:
            self.hwnd = HwndWindow(exit_event, self.windows_capture_config.get('title'),
                                   self.windows_capture_config.get('exe'),
                                   hwnd_class=self.windows_capture_config.get('hwnd_class'),
                                   global_config=self.global_config, device_manager=self)
            if self.windows_capture_config.get(
                    'interaction') == 'PostMessage':
                from ok.interaction.PostMessageInteraction import PostMessageInteraction
                self.win_interaction_class = PostMessageInteraction
            else:
                from ok.interaction.PyDirectInteraction import PyDirectInteraction
                self.win_interaction_class = PyDirectInteraction
        else:
            self.hwnd = None
        self.config = Config("devices", {"preferred": "none", "pc_full_path": "none", 'capture': 'windows'})
        self.capture_method = None
        self.handler = Handler(exit_event, 'RefreshAdb')
        logger.info('__init__ end')

    def refresh(self):
        logger.debug('calling refresh')
        self.handler.post(self.do_refresh, remove_existing=True, skip_if_running=True)

    @property
    def adb(self):
        with self._adb_lock:
            if self._adb is None:
                import adbutils
                logger.debug(f'init adb')
                from adbutils._utils import _get_bin_dir
                bin_dir = _get_bin_dir()
                exe = os.path.join(bin_dir, "adb.exe" if os.name == 'nt' else 'adb')
                from adbutils._utils import _is_valid_exe
                if os.path.isfile(exe) and _is_valid_exe(exe):
                    os.environ['ADBUTILS_ADB_PATH'] = exe
                    logger.info(f'set ADBUTILS_ADB_PATH {os.getenv("ADBUTILS_ADB_PATH")}')
                else:
                    logger.error(f'set ADBUTILS_ADB_PATH failed {exe}')
                self._adb = adbutils.AdbClient(host="127.0.0.1", socket_timeout=4)
                from adbutils import AdbError
                try:
                    self._adb.device_list()
                except AdbError as e:
                    self.try_kill_adb(e)
            return self._adb

    def try_kill_adb(self, e=None):
        logger.error('try kill adb server', e)
        import psutil
        for proc in psutil.process_iter():
            # Check whether the process name matches
            if proc.name() == 'adb.exe' or proc.name() == 'adb':
                logger.info(f'kill adb by process name {proc.cmdline()}')
                try:
                    proc.kill()
                except Exception as e:
                    logger.error(f'kill adb server failed', e)
        logger.info('try kill adb end')

    def adb_connect(self, addr, try_connect=True):
        from adbutils import AdbError
        try:
            for device in self.adb.list():
                if self.exit_event.is_set():
                    logger.error(f"adb_connect exit_event is set")
                    return None
                if device.serial == addr:
                    if device.state == 'offline':
                        logger.debug(f'adb_connect offline disconnect first {addr}')
                        self.adb.disconnect(addr)
                    else:
                        logger.debug(f'adb_connect already connected {addr}')
                        return self.adb.device(serial=addr)
            if try_connect:
                ret = self.adb.connect(addr, timeout=5)
                logger.debug(f'adb_connect try_connect {addr} {ret}')
                return self.adb_connect(addr, try_connect=False)
            else:
                logger.debug(f'adb_connect {addr} not in device list {self.adb.list()}')
        except AdbError as e:
            logger.error(f"adb connect error {addr}", e)
            self.try_kill_adb(e)
        except Exception as e:
            logger.error(f"adb connect error return none {addr}", e)

    def get_devices(self):
        return list(self.device_dict.values())

    def update_pc_device(self):
        if self.windows_capture_config is not None:
            name, hwnd, full_path, x, y, width, height = find_hwnd(self.windows_capture_config.get('title'),
                                                                   self.windows_capture_config.get('exe'), 0, 0,
                                                                   class_name=self.windows_capture_config.get(
                                                                       'hwnd_class'), player_id=-1)
            nick = name or self.windows_capture_config.get('exe')
            pc_device = {"address": "", "imei": 'pc', "device": "windows",
                         "model": "", "nick": nick, "width": width,
                         "height": height,
                         "hwnd": nick, "capture": "windows",
                         "connected": hwnd is not None,
                         "full_path": full_path or self.config.get('pc_full_path')
                         }
            if full_path and full_path != self.config.get('pc_full_path'):
                self.config['pc_full_path'] = full_path

            if width != 0:
                pc_device["resolution"] = f"{width}x{height}"
            self.device_dict['pc'] = pc_device

    def do_refresh(self, current=False):
        self.update_pc_device()
        self.refresh_emulators(current)
        self.refresh_phones(current)

        if self.exit_event.is_set():
            return
        self.do_start()

        logger.debug(f'refresh {self.device_dict}')

    def refresh_phones(self, current=False):
        if self.adb_capture_config is None:
            return
        for adb_device in self.adb.iter_device():
            imei = self.adb_get_imei(adb_device)
            if imei is not None:
                preferred = self.get_preferred_device()
                if current and preferred is not None and preferred['imei'] != imei:
                    logger.debug(f"refresh current only skip others {preferred['imei']} != {imei}")
                    continue
                found = False
                for device in self.device_dict.values():
                    if device.get('adb_imei') == imei:
                        found = True
                        break
                if not found:
                    width, height = self.get_resolution(adb_device)
                    logger.debug(f'refresh_phones found an phone {adb_device}')
                    phone_device = {"address": adb_device.serial, "device": "adb", "connected": True, "imei": imei,
                                    "nick": adb_device.prop.model or imei, "player_id": -1,
                                    "resolution": f'{width}x{height}'}
                    self.device_dict[imei] = phone_device
        logger.debug(f'refresh_phones done')

    def refresh_emulators(self, current=False):
        if self.adb_capture_config is None:
            return
        from ok.alas.emulator_windows import EmulatorManager
        manager = EmulatorManager()
        installed_emulators = manager.all_emulator_instances
        logger.info(f'installed emulators {installed_emulators}')
        for emulator in installed_emulators:
            preferred = self.get_preferred_device()
            if current and preferred is not None and preferred['imei'] != emulator.name:
                logger.debug(f"refresh current only skip others {preferred['imei']} != {emulator.name}")
                continue
            adb_device = self.adb_connect(emulator.serial)
            logger.info(f'adb_connect emulator result {emulator.type} {adb_device}')
            width, height = self.get_resolution(adb_device) if adb_device is not None else 0, 0
            name, hwnd, full_path, x, y, width, height = find_hwnd(None,
                                                                   emulator.path, width, height, emulator.player_id)
            connected = adb_device is not None and name is not None
            emulator_device = {"address": emulator.serial, "device": "adb", "full_path": emulator.path,
                               "connected": connected, "imei": emulator.name, "player_id": emulator.player_id,
                               "nick": name or emulator.name, "emulator": emulator}
            if adb_device is not None:
                emulator_device["resolution"] = f"{width}x{height}"
                emulator_device["adb_imei"] = self.adb_get_imei(adb_device)
            self.device_dict[emulator.name] = emulator_device
        logger.info(f'refresh emulators {self.device_dict}')

    def get_resolution(self, device=None):
        if device is None:
            device = self.device
        width, height = 0, 0
        if device is not None:
            if resolution := self.resolution_dict.get(device.serial):
                return resolution
            frame = self.do_screencap(device)
            if frame is not None:
                height, width, _ = frame.shape
                if self.supported_ratio is None or abs(width / height - self.supported_ratio) < 0.01:
                    self.resolution_dict[device.serial] = (width, height)
                else:
                    logger.warning(f'resolution error {device.serial} {self.supported_ratio} {width, height}')
        return width, height

    def set_preferred_device(self, imei=None, index=-1):
        logger.debug(f"set_preferred_device {imei} {index}")
        if index != -1:
            imei = self.get_devices()[index]['imei']
        elif imei is None:
            imei = self.config.get("preferred")
        preferred = self.device_dict.get(imei)
        if preferred is None:
            if len(self.device_dict) > 0:
                connected_device = None
                for device in self.device_dict.values():
                    if device.get('connected') or connected_device is None:
                        connected_device = device
                logger.info(f'first start use first or connected device {connected_device}')
                preferred = connected_device
                imei = preferred['imei']
            else:
                logger.warning(f'no devices')
                return
        if self.config.get("preferred") != imei:
            logger.info(f'preferred device did change {imei}')
            self.config["preferred"] = imei
            self.start()
        logger.debug(f'preferred device: {preferred}')

    def shell_device(self, device, *args, **kwargs):
        kwargs.setdefault('timeout', 5)
        try:
            return device.shell(*args, **kwargs)
        except Exception as e:
            logger.error(f"adb shell error maybe offline {device}", e)
            return None

    def adb_get_imei(self, device):
        return (self.shell_device(device, "settings get secure android_id") or
                self.shell_device(device, "service call iphonesubinfo 4") or device.prop.model)

    def do_screencap(self, device) -> np.ndarray | None:
        if device is None:
            return None
        try:
            png_bytes = self.shell_device(device, "screencap -p", encoding=None)
            if png_bytes is not None and len(png_bytes) > 0:
                image_data = np.frombuffer(png_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image is not None:
                    return image
                else:
                    logger.error(f"Screencap image decode error, probably disconnected")
        except Exception as e:
            logger.error('screencap', e)

    def get_preferred_device(self):
        imei = self.config.get("preferred")
        preferred = self.device_dict.get(imei)
        return preferred

    def get_preferred_capture(self):
        return self.config.get("capture")

    def set_hwnd_name(self, hwnd_name):
        preferred = self.get_preferred_device()
        if preferred.get("hwnd") != hwnd_name:
            preferred['hwnd'] = hwnd_name
            if self.hwnd:
                self.hwnd.title = hwnd_name
            self.config.save_file()

    def set_capture(self, capture):
        if self.config.get("capture") != capture:
            self.config['capture'] = capture
            self.start()

    def get_hwnd_name(self):
        preferred = self.get_preferred_device()
        return preferred.get('hwnd')

    def ensure_hwnd(self, title, exe, frame_width=0, frame_height=0, player_id=-1, hwnd_class=None):
        if self.hwnd is None:
            self.hwnd = HwndWindow(self.exit_event, title, exe, frame_width, frame_height, player_id,
                                   hwnd_class, global_config=self.global_config, device_manager=self)
        else:
            self.hwnd.update_window(title, exe, frame_width, frame_height, player_id, hwnd_class)

    def use_windows_capture(self, override_config=None, require_bg=False, use_bit_blt_only=False,
                            bit_blt_render_full=False):
        if not override_config:
            override_config = self.windows_capture_config
        self.capture_method = update_capture_method(override_config, self.capture_method, self.hwnd,
                                                    require_bg, use_bit_blt_only=use_bit_blt_only,
                                                    bit_blt_render_full=bit_blt_render_full, exit_event=self.exit_event)
        if self.capture_method is None:
            logger.error(f'cant find a usable windows capture')
        else:
            logger.info(f'capture method {type(self.capture_method)}')

    def start(self):
        self.handler.post(self.do_start, remove_existing=True, skip_if_running=True)

    def do_start(self):
        logger.debug(f'do_start')
        preferred = self.get_preferred_device()
        if preferred is None:
            if self.device_dict:
                self.set_preferred_device()
            return

        if preferred['device'] == 'windows':
            self.ensure_hwnd(self.windows_capture_config.get('title'), self.windows_capture_config.get('exe'),
                             hwnd_class=self.windows_capture_config.get('hwnd_class'))
            self.use_windows_capture(self.windows_capture_config,
                                     bit_blt_render_full=self.windows_capture_config.get('bit_blt_render_full'))
            if not isinstance(self.interaction, self.win_interaction_class):
                self.interaction = self.win_interaction_class(self.capture_method, self.hwnd)
            preferred['connected'] = self.capture_method is not None and self.capture_method.connected()
        else:
            width, height = self.get_resolution()
            if self.config.get('capture') == "windows":
                self.ensure_hwnd(None, preferred.get('full_path'), width, height, preferred['player_id'])
                logger.info(f'do_start use windows capture {self.hwnd.title}')
                self.use_windows_capture({'can_bit_blt': True}, require_bg=True, use_bit_blt_only=True,
                                         bit_blt_render_full=False)
            elif self.config.get('capture') == 'ipc':
                if not isinstance(self.capture_method, NemuIpcCaptureMethod):
                    if self.capture_method is not None:
                        self.capture_method.close()
                    self.capture_method = NemuIpcCaptureMethod(self, self.exit_event)
                self.capture_method.update_emulator(self.get_preferred_device()['emulator'])
            else:
                if not isinstance(self.capture_method, ADBCaptureMethod):
                    logger.debug(f'use adb capture')
                    if self.capture_method is not None:
                        self.capture_method.close()
                    self.capture_method = ADBCaptureMethod(self, self.exit_event, width=width,
                                                           height=height)
                if self.debug and preferred.get('full_path'):
                    self.ensure_hwnd(None, preferred.get('full_path'), width, height, preferred['player_id'])
                elif self.hwnd is not None:
                    self.hwnd.stop()
                    self.hwnd = None
            from ok.interaction.ADBInteraction import ADBBaseInteraction
            if not isinstance(self.interaction, ADBBaseInteraction):
                self.interaction = ADBBaseInteraction(self, self.capture_method, width, height)
            else:
                self.interaction.capture = self.capture_method
                self.interaction.width = width
                self.interaction.height = height

        communicate.adb_devices.emit(True)

    def update_resolution_for_hwnd(self):
        if self.hwnd is not None and self.hwnd.frame_aspect_ratio == 0 and self.adb_capture_config:
            width, height = self.get_resolution()
            logger.debug(f'update resolution for {self.hwnd} {width}x{height}')
            self.hwnd.update_frame_size(width, height)

    @property
    def device(self):
        if preferred := self.get_preferred_device():
            if self._device is None:
                logger.debug(f'get device connect {preferred}')
                self._device = self.adb_connect(preferred.get('address'))
            if self._device is not None and self._device.serial != preferred.get('address'):
                logger.info(f'get device adb device addr changed {preferred}')
                self._device = self.adb_connect(preferred.get('address'))
        else:
            logger.error(f'self.get_preferred_device returned None')
        return self._device

    def adb_kill_server(self):
        if self.adb is not None:
            self.adb.server_kill()
            logger.debug('adb kill_server')

    @property
    def width(self):
        if self.capture_method is not None:
            return self.capture_method.width
        return 0

    @property
    def height(self):
        if self.capture_method is not None:
            return self.capture_method.height
        return 0

    def update_device_list(self):
        pass

    def shell(self, *args, **kwargs):
        # Set default timeout to 5 if not provided

        device = self.device
        if device is not None:
            return self.shell_device(device, *args, **kwargs)
        else:
            raise Exception('Device is none')

    def device_connected(self):
        if self.get_preferred_device()['device'] == 'windows':
            return True
        elif self.device is not None:
            try:
                state = self.shell('echo 1')
                logger.debug(f'device_connected check device state is {state}')
                return state is not None
            except Exception as e:
                logger.error(f'device_connected error occurred, {e}')

    def get_exe_path(self, device):
        path = device.get('full_path')
        if path != 'none' and device.get(
                'device') == 'windows' and self.windows_capture_config and self.windows_capture_config.get(
            'calculate_pc_exe_path'):
            path = self.windows_capture_config.get('calculate_pc_exe_path')(path)
            logger.info(f'calculate_pc_exe_path {path}')
            if os.path.exists(path):
                return path
        elif emulator := device.get('emulator'):
            from ok.alas.platform_windows import get_emulator_exe
            return get_emulator_exe(emulator)
        else:
            return None

    def adb_check_installed(self, packages):
        installed = self.shell('pm list packages')
        if isinstance(packages, str):
            packages = [packages]
        for package in packages:
            if package in installed:
                return package

    def adb_check_in_front(self, packages):
        front = self.device.app_current()
        logger.debug(f'adb_check_in_front {front}')
        if front:
            if isinstance(packages, str):
                packages = [packages]
            for package in packages:
                if package == front.package:
                    return True

    def adb_start_package(self, package):
        self.shell(f'monkey -p {package} -c android.intent.category.LAUNCHER 1')

    def adb_ensure_in_front(self, packages):
        front = self.adb_check_in_front(packages)
        logger.debug(f'adb_ensure_in_front {front}')
        if front:
            return front
        elif installed := self.adb_check_installed(packages):
            self.adb_start_package(installed)
            return True


def parse_ratio(ratio_str):
    if ratio_str:
        # Split the string into two parts: '16' and '9'
        numerator, denominator = ratio_str.split(':')
        # Convert the strings to integers and perform the division
        ratio_float = int(numerator) / int(denominator)
        return ratio_float

cdef class NemuIpcCaptureMethod(BaseCaptureMethod):
    name = "Nemu Ipc Capture"
    description = "mumu player 12 only"
    cdef bint _connected
    cdef object device_manager, nemu_impl, emulator

    def __init__(self, device_manager, exit_event, width=0, height=0):
        super().__init__()
        self.device_manager = device_manager
        self.exit_event = exit_event
        self._connected = (width != 0 and height != 0)
        self.nemu_impl = None
        self.emulator = None

    def update_emulator(self, emulator):
        self.emulator = emulator
        logger.info(f'update_path_and_id {emulator}')
        if self.nemu_impl:
            self.nemu_impl.disconnect()
            self.nemu_impl = None

    def init_nemu(self):
        self.check_mumu_app_keep_alive_400()
        if not self.nemu_impl:
            from ok.capture.adb.nemu_ipc import NemuIpcImpl
            self.nemu_impl = NemuIpcImpl(
                nemu_folder=self.base_folder(),
                instance_id=self.emulator.player_id,
                display_id=0
            ).__enter__()

    def base_folder(self):
        return os.path.dirname(os.path.dirname(self.emulator.path))

    def check_mumu_app_keep_alive_400(self):
        """
        Check app_keep_alive from emulator config if version >= 4.0

        Args:
            file: E:/ProgramFiles/MuMuPlayer-12.0/vms/MuMuPlayer-12.0-1/config/customer_config.json

        Returns:
            bool: If success to read file
        """
        file = os.path.abspath(os.path.join(
            self.base_folder(),
            f'vms/MuMuPlayer-12.0-{self.emulator.player_id}/configs/customer_config.json'))

        # with E:\ProgramFiles\MuMuPlayer-12.0\shell\MuMuPlayer.exe
        # config is E:\ProgramFiles\MuMuPlayer-12.0\vms\MuMuPlayer-12.0-1\config\customer_config.json
        try:
            with open(file, mode='r', encoding='utf-8') as f:
                s = f.read()
                data = json.loads(s)
        except FileNotFoundError:
            logger.warning(f'Failed to check check_mumu_app_keep_alive, file {file} not exists')
            return False
        value = deep_get(data, keys='customer.app_keptlive', default=None)
        logger.info(f'customer.app_keptlive {value}')
        if str(value).lower() == 'true':
            # https://mumu.163.com/help/20230802/35047_1102450.html
            logger.error('Please turn off enable background keep alive in MuMuPlayer settings')
            raise Exception('Please turn off enable background keep alive in MuMuPlayer settings')
        return True

    def close(self):
        super().close()
        if self.nemu_impl:
            self.nemu_impl.disconnect()
            self.nemu_impl = None

    cpdef object do_get_frame(self):
        self.init_nemu()
        return self.screencap()

    cdef object screencap(self):
        if self.exit_event.is_set():
            return None
        if self.nemu_impl:
            return self.nemu_impl.screenshot(timeout=0.5)

    def connected(self):
        return True

def deep_get(d, keys, default=None):
    """
    Get values in dictionary safely.
    https://stackoverflow.com/questions/25833613/safe-method-to-get-value-of-nested-dictionary

    Args:
        d (dict):
        keys (str, list): Such as `Scheduler.NextRun.value`
        default: Default return if key not found.

    Returns:

    """
    if isinstance(keys, str):
        keys = keys.split('.')
    assert type(keys) is list
    if d is None:
        return default
    if not keys:
        return d
    return deep_get(d.get(keys[0]), keys[1:], default)

def update_capture_method(config, capture_method, hwnd, require_bg=False, use_bit_blt_only=False,
                          bit_blt_render_full=False, exit_event=None):
    try:
        if config.get('can_bit_blt'):  # slow try win graphics first
            # if bit_blt_render_full:
            #     if win_graphic := get_win_graphics_capture(capture_method, hwnd, exit_event):
            #         return win_graphic
            #     logger.debug(
            #         f"try BitBlt method {config} {hwnd} current_type:{type(capture_method)}")
            global render_full
            render_full = config.get('bit_blt_render_full', False)
            target_method = BitBltCaptureMethod
            capture_method = get_capture(capture_method, target_method, hwnd, exit_event)
            if bit_blt_render_full or capture_method.test_is_not_pure_color():
                return capture_method
            else:
                logger.info("test_is_not_pure_color failed, can't use BitBlt")
        if use_bit_blt_only:
            return None
        if win_graphic := get_win_graphics_capture(capture_method, hwnd, exit_event):
            return win_graphic

        if not require_bg:
            target_method = DesktopDuplicationCaptureMethod
            capture_method = get_capture(capture_method, target_method, hwnd, exit_event)
            return capture_method
    except Exception as e:
        logger.error(f'update_capture_method exception, return None: ', e)

def get_win_graphics_capture(capture_method, hwnd, exit_event):
    if windows_graphics_available():
        target_method = WindowsGraphicsCaptureMethod
        capture_method = get_capture(capture_method, target_method, hwnd, exit_event)
        if capture_method.start_or_stop():
            return capture_method

def get_capture(capture_method, target_method, hwnd, exit_event):
    if not isinstance(capture_method, target_method):
        if capture_method is not None:
            capture_method.close()
        capture_method = target_method(hwnd)
    capture_method.hwnd_window = hwnd
    capture_method.exit_event = exit_event
    return capture_method

MDT_EFFECTIVE_DPI = 0
user32 = ctypes.WinDLL('user32', use_last_error=True)

def is_window_minimized(hWnd):
    return user32.IsIconic(hWnd) != 0

def get_window_bounds(hwnd):
    try:
        extended_frame_bounds = ctypes.wintypes.RECT()
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(extended_frame_bounds),
            ctypes.sizeof(extended_frame_bounds),
        )
        client_x, client_y, client_width, client_height = win32gui.GetClientRect(hwnd)
        window_left, window_top, window_right, window_bottom = win32gui.GetWindowRect(hwnd)
        window_width = window_right - window_left
        window_height = window_bottom - window_top
        client_x, client_y = win32gui.ClientToScreen(hwnd, (client_x, client_y))
        monitor = user32.MonitorFromWindow(hwnd, 2)  # 2 = MONITOR_DEFAULTTONEAREST

        # Get the DPI
        dpiX = ctypes.c_uint()
        dpiY = ctypes.c_uint()
        ctypes.windll.shcore.GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, ctypes.byref(dpiX), ctypes.byref(dpiY))
        return client_x, client_y, window_width, window_height, client_width, client_height, dpiX.value / 96
    except Exception as e:
        logger.error(f'get_window_bounds exception', e)
        return 0, 0, 0, 0, 0, 0, 1

def is_foreground_window(hwnd):
    return win32gui.IsWindowVisible(hwnd) and win32gui.GetForegroundWindow() == hwnd
