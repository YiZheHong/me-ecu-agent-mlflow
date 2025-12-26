import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# 加载模型
model = mlflow.pyfunc.load_model("models:/ECUAgent/latest")

# 使用
result = model.predict({"query": "How do you enable the NPU on the ECU-850b?"})
print(result)
