import sys
import time

from PySide6.QtCore import QCoreApplication
from ok.Capture import CaptureException

import ctypes
import threading
from ok.color.Color import calculate_color_percentage
from ok.config.Config import Config
from ok.config.ConfigOption import ConfigOption
from ok.config.InfoDict import InfoDict
from ok.feature.Box import Box, find_box_by_name, relative_box
from ok.feature.FeatureSet import adjust_coordinates
from ok.gui.Communicate import communicate
from ok.logging.Logger import get_logger
from ok.stats.StreamStats import StreamStats
from ok.util.Handler import Handler
from ok.util.clazz import init_class_by_name
from typing import List

logger = get_logger(__name__)

cdef object executor

cdef class BaseTask(ExecutorOperation):
    cdef public object logger
    cdef public str name
    cdef public str description
    cdef public object feature_set
    cdef public bint _enabled
    cdef public object config
    cdef public object info
    cdef public dict default_config
    cdef public dict config_description
    cdef public dict config_type
    cdef public bint _paused
    cdef public object lock
    cdef public object _handler
    cdef public bint running
    cdef public bint trigger_interval
    cdef public double last_trigger_time
    cdef public double start_time
    cdef public object icon

    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.name = self.__class__.__name__
        self.description = ""
        self.feature_set = None
        self._enabled = False
        self.config = None
        self.info = InfoDict()
        self.default_config = {}
        self.config_description = {}
        self.config_type = {}
        self._paused = False
        self.lock = threading.Lock()
        self._handler = None
        self.running = False
        self.trigger_interval = 0
        self.last_trigger_time = 0
        self.start_time = 0
        self.icon = None

    def should_trigger(self):
        if self.trigger_interval == 0:
            return True
        now = time.time()
        time_diff = now - self.last_trigger_time
        if time_diff > self.trigger_interval:
            self.last_trigger_time = now
            return True
        return False

    def get_status(self):
        if self.running:
            return "Running"
        elif self.enabled:
            if self.paused:
                return "Paused"
            else:
                return "In Queue"
        else:
            return "Not Started"

    def enable(self):
        if not self._enabled:
            self._enabled = True
            self.info_clear()
        communicate.task.emit(self)

    @property
    def handler(self) -> Handler:
        with self.lock:
            if self._handler is None:
                self._handler = Handler(self.executor.exit_event, __name__)
            return self._handler

    def pause(self):
        if isinstance(self, TriggerTask):
            self.executor.pause()
        else:
            self.executor.pause(self)
            self._paused = True
            communicate.task.emit(self)
        if self.executor.is_executor_thread():
            self.sleep(1)

    def unpause(self):
        self._paused = False
        self.executor.start()
        communicate.task.emit(self)

    @property
    def paused(self):
        return self._paused

    def log_info(self, message, notify=False, tray=False):
        self.logger.info(message)
        self.info_set("Log", message)
        if notify:
            self.notification(message, tray=tray)

    def log_debug(self, message, notify=False, tray=False):
        self.logger.debug(message)
        if notify:
            self.notification(message, tray=tray)

    def log_error(self, message, exception=None, notify=False, tray=False):
        self.logger.error(message, exception)
        if exception is not None:
            if len(exception.args) > 0:
                message += exception.args[0]
            else:
                message += str(exception)
        self.info_set("Error", message)
        if notify:
            self.notification(message, error=True, tray=tray)

    def notification(self, message, title=None, error=False, tray=False):
        from ok.gui import app
        communicate.notification.emit(app.tr(message), app.tr(title), error, tray)

    @property
    def enabled(self):
        return self._enabled

    def info_clear(self):
        self.info.clear()

    def info_incr(self, key, inc=1):
        # If the key is in the dictionary, get its value. If not, return 0.
        value = self.info.get(key, 0)
        # Increment the value
        value += inc
        # Store the incremented value back in the dictionary
        self.info[key] = value

    def info_add_to_list(self, key, item):
        value = self.info.get(key, [])
        if isinstance(item, list):
            value += item
        else:
            value.append(item)
        self.info[key] = value

    def info_set(self, key, value):
        self.info[key] = value

    def info_add(self, key, count=1):
        self.info[key] = self.info.get(key, 0) + count

    def load_config(self):
        self.config = Config(self.__class__.__name__, self.default_config, validator=self.validate)

    def validate(self, key, value):
        message = self.validate_config(key, value)
        if message:
            return False, message
        else:
            return True, None

    def validate_config(self, key, value):
        pass

    def disable(self):
        self._enabled = False
        communicate.task.emit(self)

    @property
    def hwnd_title(self):
        if self.executor.device_manager.hwnd:
            return self.executor.device_manager.hwnd.hwnd_title
        else:
            return ""

    def run(self):
        pass

    def trigger(self):
        return True

    def on_destroy(self):
        pass

    def on_create(self):
        pass

    def set_executor(self, executor):
        self.feature_set = executor.feature_set
        self.load_config()
        self.on_create()


class TaskDisabledException(Exception):
    pass


class CannotFindException(Exception):
    pass


class FinishedException(Exception):
    pass


class WaitFailedException(Exception):
    pass


cdef class TaskExecutor:
    cdef object frame_stats
    cdef object _frame
    cdef public bint paused
    cdef double pause_start
    cdef double pause_end_time
    cdef double _last_frame_time
    cdef double wait_until_timeout
    cdef public object device_manager
    cdef public object feature_set
    cdef double wait_until_check_delay
    cdef double wait_until_before_delay
    cdef double wait_scene_timeout
    cdef public object exit_event
    cdef public bint debug_mode
    cdef public bint debug
    cdef public object global_config
    cdef public object ocr
    cdef public object current_task
    cdef str config_folder
    cdef int trigger_task_index
    cdef public list trigger_tasks
    cdef public list onetime_tasks
    cdef object thread

    def __init__(self, device_manager,
                 wait_until_timeout=10, wait_until_before_delay=1, wait_until_check_delay=0,
                 exit_event=None, trigger_tasks=[], onetime_tasks=[], feature_set=None,
                 ocr=None,
                 config_folder=None, debug=False, global_config=None):
        global executor
        executor = self
        device_manager.executor = self
        self.pause_start = time.time()
        self.pause_end_time = time.time()
        self._last_frame_time = 0
        self.paused = True
        self.device_manager = device_manager
        self.feature_set = feature_set
        self.frame_stats = StreamStats()
        self.wait_until_check_delay = wait_until_check_delay
        self.wait_until_before_delay = wait_until_before_delay
        self.wait_scene_timeout = wait_until_timeout
        self.exit_event = exit_event
        self.debug_mode = False
        self.debug = debug
        self.global_config = global_config
        self.ocr = ocr
        self.current_task = None
        self.config_folder = config_folder or "config"
        self.trigger_task_index = -1

        self.trigger_tasks = self.init_tasks(trigger_tasks)
        self.onetime_tasks = self.init_tasks(onetime_tasks)
        self.thread = threading.Thread(target=self.execute, name="TaskExecutor")
        self.thread.start()

    def init_tasks(self, task_classes):
        tasks = []
        for task_class in task_classes:
            task = init_class_by_name(task_class[0], task_class[1])
            task.set_executor(self)
            tasks.append(task)
        return tasks

    @property
    def interaction(self):
        return self.device_manager.interaction

    @property
    def method(self):
        return self.device_manager.capture_method

    def nullable_frame(self):
        return self._frame

    def check_frame_and_resolution(self, supported_ratio, min_size, time_out=8.0):
        if supported_ratio is None or min_size is None:
            return True, '0x0'
        logger.info(f'start check_frame_and_resolution')
        self.device_manager.update_resolution_for_hwnd()
        cdef double start = time.time()
        cdef object frame = None
        while frame is None and (time.time() - start) < time_out:
            frame = self.method.get_frame()
            time.sleep(0.1)
        if frame is None:
            logger.error(f'check_frame_and_resolution failed can not get frame after {time_out} {time.time() - start}')
            return False, '0x0'
        cdef int width = self.method.width
        cdef int height = self.method.height
        cdef actual_ratio = 0
        if height == 0:
            actual_ratio = 0
        else:
            actual_ratio = width / height

        # Parse the supported ratio string
        supported_ratio = [int(i) for i in supported_ratio.split(':')]
        supported_ratio = supported_ratio[0] / supported_ratio[1]

        # Calculate the difference between the actual and supported ratios
        difference = abs(actual_ratio - supported_ratio)
        support = difference <= 0.01 * supported_ratio
        if not support:
            logger.error(f'resolution error {width}x{height} {frame is None}')
        if not support and frame is not None:
            communicate.screenshot.emit(frame, "resolution_error")
        # Check if the difference is within 1%
        if support and min_size is not None:
            if width < min_size[0] or height < min_size[1]:
                support = False
        return support, f"{width}x{height}"

    def can_capture(self):
        return self.method is not None and self.interaction is not None and self.interaction.should_capture()

    def next_frame(self):
        self.reset_scene()
        while not self.exit_event.is_set():
            if self.can_capture():
                self._frame = self.method.get_frame()
                if self._frame is not None:
                    self._last_frame_time = time.time()
                    height, width = self._frame.shape[:2]
                    if height <= 0 or width <= 0:
                        logger.warning(f"captured wrong size frame: {width}x{height}")
                        self._frame = None
                    return self._frame
            self.sleep(0.00001)
        raise FinishedException()

    def is_executor_thread(self):
        return self.thread == threading.current_thread()

    def connected(self):
        return self.method is not None and self.method.connected()

    @property
    def frame(self):
        while self.paused and not self.debug_mode:
            self.sleep(1)
        if self.exit_event.is_set():
            logger.info("frame Exit event set. Exiting early.")
            sys.exit(0)
        if self._frame is None:
            return self.next_frame()
        else:
            return self._frame

    cpdef sleep(self, double timeout):
        """
        Sleeps for the specified timeout, checking for an exit event every 100ms, with adjustments to prevent oversleeping.

        :param timeout: The total time to sleep in seconds.
        """
        if self.current_task and not self.current_task.enabled:
            self.current_task = None
            raise TaskDisabledException()
        self.reset_scene()
        if timeout <= 0:
            return
        if self.debug_mode:
            time.sleep(timeout)
            return
        self.frame_stats.add_sleep(timeout)
        self.pause_end_time = time.time() + timeout
        while True:
            if self.exit_event.is_set():
                logger.info("sleep Exit event set. Exiting early.")
                sys.exit(0)
            if not (self.paused or (
                    self.current_task is not None and self.current_task.paused) or self.interaction is None or not self.interaction.should_capture()):
                to_sleep = self.pause_end_time - time.time()
                if to_sleep <= 0:
                    return
                time.sleep(to_sleep)
            time.sleep(0.1)

    def pause(self, task=None):
        if task is not None:
            if self.current_task != task:
                raise Exception(f"Can only pause current task {self.current_task}")
        elif not self.paused:
            self.paused = True
            communicate.executor_paused.emit(self.paused)
            self.reset_scene()
            self.pause_start = time.time()
            return True

    def start(self):
        if self.paused:
            self.paused = False
            communicate.executor_paused.emit(self.paused)
            self.pause_end_time += self.pause_start - time.time()

    def wait_condition(self, condition, time_out=0, pre_action=None, post_action=None, wait_until_before_delay=-1,
                       wait_until_check_delay=-1,
                       raise_if_not_found=False):
        if wait_until_before_delay == -1:
            wait_until_before_delay = self.wait_until_before_delay
        if wait_until_check_delay == -1:
            wait_until_check_delay = self.wait_until_check_delay
        if wait_until_before_delay > 0:
            self.reset_scene()
        start = time.time()
        if time_out == 0:
            time_out = self.wait_scene_timeout
        while not self.exit_event.is_set():
            if pre_action is not None:
                pre_action()
            self.sleep(wait_until_before_delay)
            self.next_frame()
            result = condition()
            result_str = list_or_obj_to_str(result)
            if result:
                logger.debug(
                    f"found result {result_str} {(time.time() - start):.3f} delay {wait_until_before_delay} {self.wait_until_check_delay}")
                return result
            if post_action is not None:
                post_action()
            if time.time() - start > time_out:
                logger.info(f"wait_until timeout {condition} {time_out} seconds")
                break
            self.sleep(wait_until_check_delay)
        if raise_if_not_found:
            raise WaitFailedException()
        return None

    def reset_scene(self):
        self._frame = None

    cdef tuple next_task(self):
        if self.exit_event.is_set():
            logger.error(f"next_task exit_event.is_set exit")
            return None, False
        cycled = False
        for onetime_task in self.onetime_tasks:
            if onetime_task.enabled:
                return onetime_task, True
        if len(self.trigger_tasks) > 0:
            if self.trigger_task_index >= len(self.trigger_tasks) - 1:
                self.trigger_task_index = -1
                cycled = True
            self.trigger_task_index += 1
            task = self.trigger_tasks[self.trigger_task_index]
            if task.enabled and task.should_trigger():
                return task, cycled
        return None, cycled

    def active_trigger_task_count(self):
        return len([x for x in self.trigger_tasks if x.enabled])

    def execute(self):
        logger.info(f"start execute")
        cdef object task
        cdef bint cycled
        while not self.exit_event.is_set():
            if self.paused:
                logger.info(f'executor is paused sleep')
                self.sleep(1)
            task, cycled = self.next_task()
            if not task:
                time.sleep(1)
                continue
            if cycled:
                self.next_frame()
            elif time.time() - self._last_frame_time > 0.1:
                self.next_frame()
            try:
                if task.trigger():
                    self.current_task = task
                    self.current_task.running = True
                    self.current_task.start_time = time.time()
                    communicate.task.emit(self.current_task)
                    if cycled or self._frame is None:
                        self.next_frame()
                    if isinstance(task, TriggerTask):
                        self.current_task.run()
                    else:
                        prevent_sleeping(True)
                        self.current_task.run()
                        prevent_sleeping(False)
                        task.disable()
                    self.current_task = None
                    communicate.task.emit(task)
                if self.current_task is not None:
                    self.current_task.running = False
                    if not isinstance(self.current_task, TriggerTask):
                        communicate.task.emit(self.current_task)
                    self.current_task = None
            except TaskDisabledException:
                logger.info(f"TaskDisabledException, continue {task}")
                from ok.gui import ok
                communicate.notification.emit(QCoreApplication.translate("app", 'Stopped'), ok.app.tr(task.name), False,
                                              True)
                continue
            except FinishedException:
                logger.info(f"FinishedException, breaking")
                break
            except Exception as e:
                if isinstance(e, CaptureException):
                    communicate.capture_error.emit()
                name = task.name
                task.disable()
                from ok.gui import ok
                error = str(e)
                communicate.notification.emit(error, ok.app.tr(name), True, True)
                task.info_set(QCoreApplication.tr('app', 'Error'), error)
                logger.error(f"{name} exception", e)
                if self._frame is not None:
                    communicate.screenshot.emit(self.frame, name)
                self.current_task = None
                communicate.task.emit(None)

        logger.debug(f'exit_event is set, destroy all tasks')
        for task in self.onetime_tasks:
            task.on_destroy()
        for task in self.trigger_tasks:
            task.on_destroy()

    def stop(self):
        logger.info('stop')
        self.exit_event.set()

    def wait_until_done(self):
        self.thread.join()

    def get_all_tasks(self):
        return self.onetime_tasks + self.trigger_tasks

    def get_task_by_class_name(self, class_name):
        for onetime_task in self.onetime_tasks:
            if onetime_task.__class__.__name__ == class_name:
                return onetime_task
        for trigger_task in self.trigger_tasks:
            if trigger_task.__class__.__name__ == class_name:
                return trigger_task

def list_or_obj_to_str(val):
    if val is not None:
        if isinstance(val, list):
            return ', '.join(str(obj) for obj in val)
        else:
            return str(val)
    else:
        return None

def prevent_sleeping(yes=True):
    # Prevent the system from sleeping
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 if yes else 0x80000000)

cdef class ExecutorOperation:
    cdef double last_click_time

    def __init__(self):
        self.last_click_time = 0

    def exit_is_set(self):
        return self.executor.exit_event.is_set()

    def box_in_horizontal_center(self, box, off_percent=0.02):
        if box is None:
            return False

        center = self.executor.method.width / 2
        box_center = box.x + box.width / 2

        offset = abs(box_center - center)

        if offset / self.executor.method.width < off_percent:
            return True
        else:
            return False

    @property
    def executor(self):
        return executor

    @property
    def debug(self):
        return self.executor.debug

    def is_scene(self, the_scene):
        return isinstance(self.executor.current_scene, the_scene)

    def reset_scene(self):
        self.executor.reset_scene()

    def click(self, x: int | Box | List[Box] = -1, y=-1, move_back=False, name=None, interval=-1, move=True,
              down_time=0.01, after_sleep=0):
        if isinstance(x, Box) or isinstance(x, list):
            return self.click_box(x, move_back=move_back, down_time=down_time, after_sleep=after_sleep)
        if not self.check_interval(interval):
            self.executor.reset_scene()
            return False
        communicate.emit_draw_box("click", [Box(max(0, x - 10), max(0, y - 10), 20, 20, name="click")], "green",
                                  frame=self.executor.nullable_frame())
        self.executor.interaction.click(x, y, move_back, name=name, move=move, down_time=down_time)
        if name:
            self.logger.info(f'click {name} {x, y} after_sleep {after_sleep}')
        if after_sleep > 0:
            self.sleep(after_sleep)
        self.executor.reset_scene()
        return True

    def middle_click(self, x: int | Box = -1, y=-1, move_back=False, name=None, interval=-1, down_time=0.01):
        if not self.check_interval(interval):
            self.executor.reset_scene()
            return False
        communicate.emit_draw_box("middle_click", [Box(max(0, x - 10), max(0, y - 10), 20, 20, name="click")], "green")
        self.executor.interaction.middle_click(x, y, move_back, name=name, down_time=down_time)
        self.executor.reset_scene()
        return True

    def check_interval(self, interval):
        if interval <= 0:
            return True
        now = time.time()
        if now - self.last_click_time < interval:
            return False
        else:
            self.last_click_time = now
            return True

    def mouse_down(self, x=-1, y=-1, name=None, key="left"):
        frame = self.executor.nullable_frame()
        communicate.emit_draw_box("mouse_down", [Box(max(0, x - 10), max(0, y - 10), 20, 20, name="click")], "green",
                                  frame)
        self.executor.reset_scene()
        self.executor.interaction.mouse_down(x, y, name=name, key=key)

    def mouse_up(self, name=None, key="left"):
        frame = self.executor.nullable_frame()
        communicate.emit_draw_box("mouse_up", self.box_of_screen(0.5, 0.5, width=0.01, height=0.01, name="click"),
                                  "green",
                                  frame)
        self.executor.interaction.mouse_up(key=key)
        self.executor.reset_scene()

    def right_click(self, x=-1, y=-1, move_back=False, name=None):
        communicate.emit_draw_box("right_click", [Box(max(0, x - 10), max(0, y - 10), 20, 20, name="right_click")],
                                  "green")
        self.executor.reset_scene()
        self.executor.interaction.right_click(x, y, move_back, name=name)

    def swipe_relative(self, from_x, from_y, to_x, to_y, duration=0.5):
        self.swipe(int(self.width * from_x), int(self.height * from_y), int(self.width * to_x),
                   int(self.height * to_y), duration)

    @property
    def hwnd(self):
        return self.executor.device_manager.hwnd

    def scroll_relative(self, x, y, count):
        self.scroll(int(self.width * x), int(self.height * y), count)

    def scroll(self, x, y, count):
        frame = self.executor.nullable_frame()
        communicate.emit_draw_box("scroll", [
            Box(x, y, 10, 10,
                name="scroll")], "green", frame)
        # ms = int(duration * 1000)
        self.executor.interaction.scroll(x, y, count)
        self.executor.reset_scene()
        # self.sleep(duration)

    def swipe(self, from_x, from_y, to_x, to_y, duration=0.5):
        frame = self.executor.nullable_frame()
        communicate.emit_draw_box("swipe", [
            Box(min(from_x, to_x), min(from_y, to_y), max(abs(from_x - from_x), 10), max(abs(from_y - to_y), 10),
                name="swipe")], "green", frame)
        ms = int(duration * 1000)
        self.executor.reset_scene()
        self.executor.interaction.swipe(from_x, from_y, to_x, to_y, ms)
        self.sleep(duration)

    def screenshot(self, name=None, frame=None):
        if name is None:
            raise ValueError('screenshot name cannot be None')
        communicate.screenshot.emit(self.frame if frame is None else frame, name)

    def click_box_if_name_match(self, boxes, names, relative_x=0.5, relative_y=0.5):
        """
        Clicks on a box from a list of boxes if the box's name matches one of the specified names.
        The box to click is selected based on the order of names provided, with priority given
        to the earliest match in the names list.

        Parameters:
        - boxes (list): A list of box objects. Each box object must have a 'name' attribute.
        - names (str or list): A string or a list of strings representing the name(s) to match against the boxes' names.
        - relative_x (float, optional): The relative X coordinate within the box to click,
                                        as a fraction of the box's width. Defaults to 0.5 (center).
        - relative_y (float, optional): The relative Y coordinate within the box to click,
                                        as a fraction of the box's height. Defaults to 0.5 (center).

        Returns:
        - box: the matched box

        The method attempts to find and click on the highest-priority matching box. If no matches are found,
        or if there are no boxes, the method returns False. This operation is case-sensitive.
        """
        to_click = find_box_by_name(boxes, names)
        if to_click is not None:
            self.logger.info(f"click_box_if_name_match found {to_click}")
            self.click_box(to_click, relative_x, relative_y)
            return to_click

    def box_of_screen(self, x, y, to_x= 1.0, to_y=1.0, width = 0.0, height = 0.0, name=None,
                      hcenter=False):
        if name is None:
            name = f"{x} {y} {width} {height}"
        if self.out_of_ratio():
            should_width = self.executor.device_manager.supported_ratio * self.height
            return self.box_of_screen_scaled(should_width, self.height,
                                             x_original=x * should_width,
                                             y_original=self.height * y,
                                             to_x=to_x * should_width,
                                             to_y=to_y * self.height, width_original=width * should_width,
                                             height_original=self.height * height,
                                             name=name, hcenter=hcenter)
        else:
            return relative_box(self.executor.method.width, self.executor.method.height, x, y,
                                to_x=to_x, to_y=to_y, width=width, height=height, name=name)

    def out_of_ratio(self):
        return self.executor.device_manager.supported_ratio and abs(
            self.width / self.height - self.executor.device_manager.supported_ratio) > 0.01

    def box_of_screen_scaled(self, original_screen_width, original_screen_height, x_original, y_original,
                             to_x = 0, to_y = 0, width_original=0, height_original=0,
                             name=None, hcenter=False):
        if width_original == 0:
            width_original = to_x - x_original
        if height_original == 0:
            height_original = to_y - y_original
        x, y, w, h, scale = adjust_coordinates(x_original, y_original, width_original, height_original,
                                               self.screen_width, self.screen_height, original_screen_width,
                                               original_screen_height, hcenter=hcenter)
        return Box(x, y, w, h, name=name)

    def height_of_screen(self, percent):
        return int(percent * self.executor.method.height)

    @property
    def screen_width(self):
        return self.executor.method.width

    @property
    def screen_height(self):
        return self.executor.method.height

    def width_of_screen(self, percent):
        return int(percent * self.executor.method.width)

    def click_relative(self, x, y, move_back=False, hcenter=False, move=True, after_sleep=0, name=None):
        if self.out_of_ratio():
            should_width = self.executor.device_manager.supported_ratio * self.height
            x, y, w, h, scale = adjust_coordinates(x * should_width, y * self.height, 0, 0,
                                                   self.screen_width, self.screen_height, should_width,
                                                   self.height, hcenter=hcenter)
        else:
            x, y = int(self.width * x), int(self.height * y)
        self.click(x, y, move_back, name=name, move=move, after_sleep=after_sleep)

    def middle_click_relative(self, x, y, move_back=False, down_time=0.01):
        self.middle_click(int(self.width * x), int(self.height * y), move_back,
                          name=f'relative({x:.2f}, {y:.2f})', down_time=down_time)

    @property
    def height(self):
        return self.executor.method.height

    @property
    def width(self):
        return self.executor.method.width

    def move_relative(self, x, y):
        self.move(int(self.width * x), int(self.height * y))

    def move(self, x, y):
        self.executor.interaction.move(x, y)
        self.executor.reset_scene()

    def click_box(self, box: Box | List[Box] = None, relative_x=0.5, relative_y=0.5, raise_if_not_found=False,
                  move_back=False, down_time=0.01, after_sleep=1):
        if isinstance(box, list):
            if len(box) > 0:
                box = box[0]

        if not box:
            self.logger.error(f"click_box box is None")
            if raise_if_not_found:
                raise Exception(f"click_box box is None")
            return
        x, y = box.relative_with_variance(relative_x, relative_y)
        return self.click(x, y, name=box.name, move_back=move_back, down_time=down_time, after_sleep=after_sleep)

    def wait_scene(self, scene_type=None, time_out=0, pre_action=None, post_action=None):
        return self.executor.wait_scene(scene_type, time_out, pre_action, post_action)

    def sleep(self, timeout):
        self.executor.sleep(timeout)
        return True

    def send_key(self, key, down_time=0.02, interval=-1, after_sleep=0):
        if not self.check_interval(interval):
            self.executor.reset_scene()
            return False
        frame = self.executor.nullable_frame()
        communicate.emit_draw_box("send_key", [Box(max(0, 0), max(0, 0), 20, 20, name="send_key_" + str(key))], "green",
                                  frame)
        self.executor.reset_scene()
        self.executor.interaction.send_key(key, down_time)
        if after_sleep > 0:
            self.sleep(after_sleep)
        return True

    def get_global_config(self, option: ConfigOption):
        return self.executor.global_config.get_config(option)

    def send_key_down(self, key):
        self.executor.reset_scene()
        self.executor.interaction.send_key_down(key)

    def send_key_up(self, key):
        self.executor.reset_scene()
        self.executor.interaction.send_key_up(key)

    def wait_until(self, condition, time_out=0, pre_action=None, post_action=None, wait_until_before_delay=-1,
                   wait_until_check_delay=-1,
                   raise_if_not_found=False):
        return self.executor.wait_condition(condition, time_out, pre_action, post_action, wait_until_before_delay,
                                            wait_until_check_delay,
                                            raise_if_not_found=raise_if_not_found)

    def wait_click_box(self, condition, time_out=0, pre_action=None, post_action=None, raise_if_not_found=False):
        target = self.wait_until(condition, time_out, pre_action, post_action)
        self.click_box(box=target, raise_if_not_found=raise_if_not_found)
        return target

    def next_frame(self):
        self.executor.next_frame()
        return self.frame

    @property
    def scene(self):
        return self.executor.current_scene

    @property
    def frame(self):
        return self.executor.frame

    @staticmethod
    def draw_boxes(feature_name=None, boxes=None, color="red"):
        communicate.emit_draw_box(feature_name, boxes, color)

    def calculate_color_percentage(self, color, box: Box):
        percentage = calculate_color_percentage(self.frame, color, box)
        box.confidence = percentage
        self.draw_boxes(box.name, box)
        return percentage

    def adb_shell(self, *args, **kwargs):
        return self.executor.device_manager.shell(*args, **kwargs)

cdef class TriggerTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.default_config['_enabled'] = False
        self.trigger_interval = 0

    def on_create(self):
        self._enabled = self.config.get('_enabled', False)

    def get_status(self):
        if self.enabled:
            return "Enabled"
        else:
            return "Disabled"

    def enable(self):
        super().enable()
        self.config['_enabled'] = True

    def disable(self):
        super().disable()
        self.config['_enabled'] = False
