ECU Agent MLflow Management System
This project provides an integrated RAG (Retrieval-Augmented Generation) lifecycle management system for ECU (Electronic Control Unit) documentation. It uses MLflow for experiment tracking and FastAPI for training, registering, and serving models.

🚀 Quick Start with Docker
Assuming you have already cloned the repository and installed Docker/Docker Compose.

1. Environment Setup
Create a .env file in the root directory and add your API keys:

Code snippet

DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com
2. Launch the Services
Run the following command from the project root:

Bash

docker compose -f docker/docker-compose.yml up --build
This will start two services:

MLflow UI: http://localhost:5000

API Service: http://localhost:8000 (Handles Train/Register/Predict)

🛠️ Workflow Lifecycle
You can manage the model entirely through the API. Use the Swagger UI at http://localhost:8000/docs or the following commands:

Step 1: Run Grid Search Training
Trigger the optimization experiment to find the best RAG parameters.

Bash

curl -X POST http://localhost:8000/train
Response: A list of run_ids with their corresponding pass_rate.

Step 2: Register the Best Model
Pick a run_id from the training results and register it to the production registry.

Bash

curl -X POST http://localhost:8000/register \
     -H "Content-Type: application/json" \
     -d '{"run_id": "YOUR_BEST_RUN_ID"}'
Step 3: Reload & Predict
Refresh the active model instance and start asking questions.

Bash

# Reload the service to use the newly registered model
curl -X POST http://localhost:8000/reload

# Perform inference
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"query": "Which ECU model is robust in harsh environments?"}'

📁 Project Structure
/src: Core logic and MLflow model wrappers.

/data: Technical manuals and test questions.

/experiments: Local storage for MLflow runs and artifacts.

/docker: Dockerfile and Compose configurations.