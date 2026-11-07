import os, sys, importlib.util
_dir = os.path.dirname(os.path.abspath(__file__))
for _mod in ("stage3_analysis", "stage3_figures"):
    _p = os.path.join(_dir, f"{_mod}.py")
    _full = f"{__name__}.{_mod}"
    if _full not in sys.modules and os.path.exists(_p):
        _spec = importlib.util.spec_from_file_location(_full, _p)
        _m = importlib.util.module_from_spec(_spec)
        sys.modules[_full] = _m
        _spec.loader.exec_module(_m)
        globals()[_mod] = _m
