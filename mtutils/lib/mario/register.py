## registers for AI modules

_MT_AI_modules = dict()

def register_module(func_name, func):
    assert func_name not in _MT_AI_modules, "{} already registered.".format(func_name)
    _MT_AI_modules.update({func_name: func})


def apply_module(url, **kwargs):
    module_func = _MT_AI_modules[url]
    return module_func(**kwargs)
