# For blaze on Google's infrastructure

# TODO: remove unnecessary deps (go/py-strict-deps)

py_library(
    name = "xiuminglib",
    srcs = glob([
        "*.py",
    ]),
    srcs_version = "PY3",
    deps = [
        # bpy not in third_party, so has to be wrapped into an MPM package
        "//experimental/users/xiuming/xiuminglib/data",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/blender",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/geometry",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/io",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/launch",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/vis",
        "//pyglib:gfile",
        "//file/colossus/cns",
        "//third_party/py/cvx2",
        "//third_party/py/matplotlib",
        "//third_party/py/mpl_toolkits/axes_grid1",
        "//third_party/py/mpl_toolkits/mplot3d",
        "//third_party/py/numpy",
        "//third_party/py/PIL:pil",
        "//third_party/py/scipy",
        "//third_party/py/tqdm",
    ],
)

py_library(
    name = "exr",
    srcs = glob([
        "*.py",
    ]),
    srcs_version = "PY3",
    deps = [
        # bpy not in third_party, so has to be wrapped into an MPM package
        "//experimental/users/xiuming/xiuminglib/data",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/blender",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/geometry",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/io",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/launch",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/vis",
        "//pyglib:gfile",
        "//file/colossus/cns",
        "//third_party/py/cvx2",
        "//third_party/py/matplotlib",
        "//third_party/py/mpl_toolkits/axes_grid1",
        "//third_party/py/mpl_toolkits/mplot3d",
        "//third_party/py/numpy",
        "//third_party/py/PIL:pil",
        "//third_party/py/scipy",
        "//third_party/py/tqdm",
        # FIXME: Listing OpenEXR and Imath as deps here segfaults bpy
        "//third_party/py/Imath",
        "//third_party/py/OpenEXR",
    ],
)
