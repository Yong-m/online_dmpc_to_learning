from setuptools import setup, find_packages

setup(
    name="policyflow",
    version="0.0.0",
    packages=find_packages(),
    license="BSD-3",
    description="Continuous normalizing flow policy optimization implemented in pytorch",
    python_requires=">=3.10",
    install_requires=[
        "gym",
        "numpy",
        "onnx",
        # "tensorboard",
        # "torch",
        # "torchvision",
        "wandb",
        "einops",
        "GitPython",
        "onnx",
    ],
)
