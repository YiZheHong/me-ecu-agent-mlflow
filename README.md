# ECU Agent MLflow Management System

> Built on top of [me-ecu-agent](https://github.com/YiZheHong/me-ecu-agent)

An integrated RAG (Retrieval-Augmented Generation) lifecycle management system for ECU (Electronic Control Unit) documentation. Uses MLflow for experiment tracking and FastAPI for training, registering, and serving models.

## Quick Start

### 1. Setup Environment

Create a `.env` file in the root directory:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com
```

### 2. Launch Services

```bash
docker compose -f docker/docker-compose.yml up --build
```

**Services:**
- MLflow UI: http://localhost:5000
- API Service: http://localhost:8000

## Usage

Access the Swagger UI at http://localhost:8000/docs

### Step 1: Train Model

Click **POST /train** → Execute

This runs grid search to find optimal RAG parameters. Returns run IDs with their pass rates.

### Step 2: Register Best Model

Click **POST /register** → Enter the best `run_id` from training → Execute

Example request body:
```json
{
  "run_id": "3b2a80bc62894e72aca80dcc49f3acac",
  "model_name": "ECUAgent"
}
```

### Step 3: Reload & Predict

Click **POST /reload** → Execute (refreshes the active model)

Click **POST /predict** → Enter your query → Execute

Example request body:
```json
{
  "query": "What is ECU-850b",
  "session_id": "default"
}
```

## Project Structure

- `/src` - Core logic and MLflow model wrappers
- `/data` - Technical manuals and test questions
- `/experiments` - MLflow runs and artifacts storage
- `/docker` - Docker configurations