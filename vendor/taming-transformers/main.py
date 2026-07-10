"""Minimal entrypoint shim retained for the instantiate_from_config helper
used by taming.models. The full training main() was trimmed for V1."""
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    try:
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
            return getattr(module_imp, cls)
        return getattr(importlib.import_module(module, package=None), cls)
    except AttributeError:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
        return getattr(module_imp, cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
