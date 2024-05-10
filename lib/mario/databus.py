from typing import Dict

class SlotsMeta(type):
    def __new__(cls, *args, **kwargs):
        super_class = args[1]
        if super_class:
            if getattr(super_class[0], "__slots__", None):
                __attr = args[2]
                if not __attr.get("__slots__"):
                    __attr["__slots__"] = ()
                    args = (args[0], args[1], __attr)
        return super().__new__(cls, *args, **kwargs)

class DataBus(metaclass=SlotsMeta):
    __slots__ = ('inputs', 'outputs')

    def __init__(self, inputs=None, outputs=None):
        self.inputs  = inputs
        self.outputs = outputs
