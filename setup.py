from setuptools import setup, find_packages

setup(
    name="me-ecu-agent-mlflow",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mlflow>=2.9.0",
        "me-ecu-agent @ git+https://github.com/YiZheHong/me-ecu-agent.git@main",
    ],
)