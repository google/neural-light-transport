# For blaze on Google's infrastructure

py_library(
    name = "xiuminglib",
    srcs_version = "PY3",
    deps = [
        "//experimental/users/xiuming/xiuminglib/data",
        "//experimental/users/xiuming/xiuminglib/xiuminglib",
    ],
)

py_binary(
    name = "test",
    srcs = ["test.py"],
    python_version = "PY3",
    deps = [
        "//experimental/users/xiuming/xiuminglib",
        # "//file/colossus/cns",
        # "//pyglib:gfile",
        # "//pyglib:resources",
        # "//research/vale/tools/sstable",
        # "//research/vision/loco/python_utils:tf_example_io",
        "//third_party/py/absl:app",
        "//third_party/py/IPython:ipython-libs",
        # "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        # "//third_party/py/tqdm",
    ],
)
