from setuptools import setup, Extension

setup(
    name="minitorch",
    version="0.1",
    packages=[
        "minitorch"
    ],
    package_data={"minitorch": []},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],

    ext_modules=[
    	Extension('minitorch.minitorch_c', sources=['minitorch/minitorch_c.c'])
    ]
)
