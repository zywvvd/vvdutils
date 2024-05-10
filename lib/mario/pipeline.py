#
# Pipeline mode
#
from _thread import start_new_thread
from re import T
import time
import json

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from .. import strong_printing
from .. import try_exc_handler
from .. import is_integer
from .. import is_bool
from .. import MyEncoder
# Flask packages
#
from flask import Flask
from flask import request, jsonify
import time
app = Flask(__name__)

from .register import apply_module, register_module
from .databus import DataBus
from .hook import Hook

from func_timeout import func_set_timeout, FunctionTimedOut


class Pipeline(object):
    def __init__(self,
                gpu_id=0,
                running_mode='sequence',
                max_worker_num=10,
                developer_mode=False,
                http_mode=False,
                timeout=None,
                databus=DataBus,
                debug=False,
                **kwargs):
        """
        Arguments:
            - gpu_id: ID of gpu to be used during inference
        """

        self.input_check(gpu_id, running_mode, max_worker_num, developer_mode, http_mode, timeout, databus, debug)

        self.max_worker_num = max_worker_num
        self.developer_mode = developer_mode
        self.running_mode = running_mode
        self.DataBus = databus
        self.http_mode = http_mode
        self.timeout = timeout
        self.debug = debug
        assert hasattr(databus, '__slots__'), "The DataBus class must define variable '__slots__'"

        if self.running_mode == 'multiprocessing':
            self.process_pool = Pool(self.max_worker_num)
        elif self.running_mode == 'multithreading':
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_worker_num)

        self._hooks = []
        self.gpu_id = gpu_id
        self.add_property(**kwargs)

        self.initialize(self.gpu_id)

        if self.http_mode:
            self._start_server()

    def input_check(self, gpu_id, running_mode, max_worker_num, developer_mode, http_mode, timeout, databus, debug):
        assert is_integer(gpu_id), f"input gpu_id {gpu_id} must be a integer"
        assert running_mode in ['sequence', 'multiprocessing', 'multithreading'],\
            f"running_mode {running_mode} should be one of ['sequence', 'multiprocessing', 'multithreading']."
        assert is_integer(max_worker_num) and max_worker_num > 0, f"input gpu_id {max_worker_num} must be a positive  integer"
        assert is_bool(developer_mode), f"developer_mode {developer_mode} should be a bool"
        assert is_bool(http_mode), f"http_mode {http_mode} should be a bool"
        assert is_bool(debug), f"debug {debug} should be a bool"
        assert timeout is None or timeout >= 0, f"timeout {timeout} should be None or float >= 0"
        assert issubclass(databus, DataBus), f"input databus {databus} must be a subclass of DataBus"

        if developer_mode:
            strong_printing("under developer mode, all exception handlings are truned off !!!")

    def add_property(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize(self, gpu_id):
        ## empty submodule
        def empty(*args, **kwargs):
            output = 'empty_test'
            print("@@ empty module walked through.")
            return output

        self.register_module('empty', empty)

    @staticmethod
    def register_module(name, func):
        register_module(name, func)

    @staticmethod
    def apply_module(url, **kwargs):
        apply_module(url, **kwargs)

    def _start_server(self):
        #
        # AI algorithm modules
        #
        @app.route('/infer/', methods=['POST'])
        def http_infer():
            if request.method == 'POST':
                def try_func():
                    input_string = request.headers['input_string']
                    inputs = json.loads(input_string)
                    results = self.infer(inputs)
                    json_str = json.dumps(results, cls=MyEncoder)
                    jsoned_results = json.loads(json_str)
                    return jsonify(jsoned_results)
                def exc_func(e):
                    print(e)
                    return self._global_exception_result(str(e), None)
                result = try_exc_handler(try_func, exc_func, self.developer_mode)
                return result

        def start_servers():
            app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

        def try_func():
            start_new_thread(start_servers, ())
            time.sleep(0.01)
            print("[*] Servers are running in background")

        def exc_func(e):
            print(e)
            print("!! Start server failed")

        try_exc_handler(try_func, exc_func, self.developer_mode)


    def _infer_single_task(self, databus):
        def try_func():
            assert isinstance(databus, DataBus), f"databus {databus} must be an instance of DataBus"
            self.call_hook('before_single_infer', self, databus)
            databus.outputs = self.infer_single_task(databus.inputs)
            self.call_hook('after_single_infer', self, databus)
            return databus.outputs
        exc_func = lambda e : self._single_exception_result(str(e), databus.inputs)
        databus.outputs = try_exc_handler(try_func, exc_func, self.developer_mode)
        return databus

    def infer_single_task(self, inputs):
        # core function
        return self.apply_module('empty')

    @func_set_timeout(0, allowOverride=True)
    def timer_multi_process(self, databus_list, **kwargs):
        return self.multi_process(databus_list)

    def multi_process(self, databus_list):
        """
            "Pool.map results are ordered. If you need order, great; if you don't,
            Pool.imap_unordered may be a useful optimization. Note that while the
            order in which you receive the results from Pool.map is fixed, the
            order in which they are computed is arbitrary."
        -- https://stackoverflow.com/questions/41273960/python-3-does-pool-keep-the-original-order-of-data-passed-to-map
        """
        self.call_hook('before_global_infer', self, databus_list)

        running_mode = self.running_mode
        core_function = self._infer_single_task
        num_tasks = len(databus_list)

        if None: pass
        elif running_mode == 'multiprocessing':
            # * multiprocessing
            pool = self.process_pool
            databus_list = pool.map(core_function, databus_list)
            pool.close()
            pool.join()

        elif running_mode == 'multithreading':
            # * multi threading
            databus_list = list(self.thread_pool.map(core_function, databus_list))

        elif running_mode == 'sequence':
            # * sequence
            databus_list = list(map(core_function, databus_list))
        else:
            raise TypeError

        assert isinstance(databus_list, list), f"databus_list {databus_list} must be a list"
        self.call_hook('after_global_infer', self, databus_list)

        # check the output order
        if len(databus_list) != num_tasks:
            raise RuntimeError(f"Error: Something fatal happened during inference!!!!\
                len(databus_list) {len(databus_list)} != len(databus_list) {len(databus_list)}")

        databus = self.outputs_fusion(databus_list)
        assert isinstance(databus, DataBus), f"outputs_fusion must return an instance of DataBus {databus}"
        return databus

    def _global_exception_result(self, exception_str, inputs):
        try_func = lambda : self.global_exception_result(exception_str, inputs)
        def exc_func(e):
            print("!! global_exception_result crashed !")
            print(e)
            return 'global_exception_result crashed !'

        result = try_exc_handler(try_func=try_func, exc_func=exc_func, developer_mode=self.developer_mode)
        return result


    def global_exception_result(self, exception_str, inputs):
        # overwrite to make exception data for your project
        return exception_str


    def _single_exception_result(self, exception_str, inputs):
        try_func = lambda : self.single_exception_result(exception_str, inputs)
        def exc_func(e):
            print("!! single_exception_result crashed !")
            print(e)
            return 'single_exception_result crashed !'

        return try_exc_handler(try_func=try_func, exc_func=exc_func, developer_mode=self.developer_mode)


    def single_exception_result(self, exception_str, inputs):
        # overwrite to make exception data for your project
        return exception_str

    def inputs_division(self, databus):
        return [self.DataBus(inputs) for inputs in databus.inputs]

    def outputs_fusion(self, databus_list):
        return self.DataBus(
            inputs  = [x.inputs  for x in databus_list],
            outputs = [x.outputs for x in databus_list],
        )

    def infer(self, inputs, **kwargs):
        def try_func():
            if self.debug:
                print("@@ debug info: inputs")
                print(inputs)
                print()
            databus = self.DataBus(inputs)
            self.call_hook('before_run', self, databus)
            databus_list = self.inputs_division(databus)
            if self.timeout is not None:
                databus = self.timer_multi_process(databus_list, forceTimeout=self.timeout)
            else:
                databus = self.multi_process(databus_list)
            self.call_hook('after_run', self, databus)
            return databus.outputs
        exc_func = lambda e : self._global_exception_result(str(e), inputs)

        results = try_exc_handler(try_func, exc_func, self.developer_mode)
        return results

    def register_hook(self, hook, priority=50, before_priority=None, after_priority=None):
        """
        register hook module for pipeline
        priority num bigger the priority higher (default)
        before_priority: priority that decides the running order of the before_xxx hooks
        after_priority: priority that decides the running order of the aftere_xxx hooks
        if set, before_priority/after_priority overrides priority
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'before_priority') and hasattr(hook, 'after_priority'):
            raise ValueError('!! "before_priority/after_priority" are reserved attributes for hooks')

        if before_priority is None:
            hook.before_priority = priority
        else:
            hook.before_priority = before_priority

        if after_priority is None:
            hook.after_priority = priority
        else:
            hook.after_priority = after_priority

        self._hooks.append(hook)

    def deregister_hook(self, hook_name:str):
        """ Deregister hook by their class name """
        find_hook_flag = False
        for hook_id, hook in enumerate(self._hooks):
            if hook_name in str(hook):
                find_hook_flag = True
                break
        if find_hook_flag:
            self._hooks.pop(hook_id)
        else:
            raise RuntimeError("No hook called '{}' found registered in pipeline".format(hook_name))


    def call_hook(self, fn_name, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        if fn_name.startswith('before_'):
            hooks = sorted(self._hooks, key=lambda x: x.before_priority, reverse=True)
        else:
            hooks = sorted(self._hooks, key=lambda x: x.after_priority,  reverse=True)

        if self.developer_mode:
            for index, hook in enumerate(hooks):
                getattr(hook, fn_name)(*args, **kwargs)
        else:
            for index, hook in enumerate(hooks):
                try:
                    getattr(hook, fn_name)(*args, **kwargs)
                except Exception as e:
                    print(f"!! The {index}th hook {str(type(hooks[index]))} running failed under processing of {fn_name}.")
                    print(e)
                    raise RuntimeError(e)

    __call__ = infer
