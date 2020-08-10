from importlib import import_module

from .log import get_logger
logger = get_logger()

# For < Python 3.6
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError


def preset_import(name):
    """A unified importer for both regular and ``google3`` modules, according
    to specified presets/profiles (e.g., ignoring ``ModuleNotFoundError``).
    """
    if name in ('cv2', 'opencv'):
        try:
            # BUILD dep:
            # "//third_party/py/cvx2",
            from cvx2 import latest as mod
            # Or
            # BUILD dep:
            # "//third_party/OpenCVX:cvx2",
            # from google3.third_party.OpenCVX import cvx2 as cv2
        except ModuleNotFoundError:
            mod = import_module_404ok('cv2')
        # TODO: Below is cleaner, but doesn't work
        # mod = import_module_404ok('cvx2.latest')
        # if mod is None:
        #    mod = import_module_404ok('cv2')
        return mod

    if name in ('tf', 'tensorflow'):
        mod = import_module_404ok('tensorflow')
        return mod

    if name == 'gfile':
        # BUILD deps:
        # "//pyglib:gfile",
        # "//file/colossus/cns",
        mod = import_module_404ok('google3.pyglib.gfile')
        return mod

    if name in ('bpy', 'bmesh', 'OpenEXR', 'Imath'):
        # BUILD deps:
        # "//third_party/py/Imath",
        # "//third_party/py/OpenEXR",
        mod = import_module_404ok(name)
        return mod

    if name in ('Vector', 'Matrix', 'Quaternion'):
        mod = import_module_404ok('mathutils')
        cls = _get_module_class(mod, name)
        return cls

    if name == 'BVHTree':
        mod = import_module_404ok('mathutils.bvhtree')
        cls = _get_module_class(mod, name)
        return cls

    raise NotImplementedError(name)


def import_module_404ok(*args, **kwargs):
    """Returns ``None`` (instead of failing) in the case of
    ``ModuleNotFoundError``.
    """
    try:
        mod = import_module(*args, **kwargs)
    except (ModuleNotFoundError, ImportError) as e:
        mod = None
        logger.debug("Ignored: %s", str(e))
    return mod


def _get_module_class(mod, clsname):
    if mod is None:
        return None
    return getattr(mod, clsname)
