# ECU Agent MLflow Management System

> **Built on**: [me-ecu-agent](https://github.com/YiZheHong/me-ecu-agent) - Core RAG engine for ECU documentation

MLflow-based lifecycle management for ECU documentation RAG models. Train, register, and serve models through simple FastAPI endpoints.

## Quick Start

### 1. Setup Environment

Create `.env` file:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com
```

### 2. Launch Services

```bash
docker compose -f docker/docker-compose.yml up --build
```

**Access:**
- MLflow UI: http://localhost:5000
- API: http://localhost:8000/docs

## Workflow

### Step 1: Train

**POST /train** - Run grid search to find best parameters

Returns run IDs with pass rates:
```json
{
  "status": "success",
  "results": [
    {
      "run_id": "3b2a80bc...",
      "pass_rate": 0.95
    }
  ]
}
```

### Step 2: Register

**POST /register** - Register best model to production

```json
{
  "run_id": "3b2a80bc...",
  "model_name": "ECUAgent"
}
```

### Step 3: Reload & Predict

**POST /reload** - Load registered model

**POST /predict** - Query the model

```json
{
  "query": "What is ECU-850b max temperature?",
  "session_id": "default"
}
```

## What Gets Optimized

| Parameter | What It Does | Range |
|-----------|--------------|-------|
| `chunk_size` | Max chars per chunk | 1000-2000 |
| `chunk_overlap` | Overlap between chunks | 50-300 |
| `default_top_k` | Documents per query | 3-7 |
| `generic_top_k` | Docs for generic queries | 3-7 |

## Testing & Validation

**Evaluation Dataset**: `test-questions.csv`
- Covers single model, comparison, and spec comparison queries
- Includes expected answers and criteria

**Metrics**:
- Pass Rate: % of correct answers
- Avg Response Time: Latency per query

**Validation Flow**:
1. Train runs automated eval on all test cases
2. Best model selected by pass rate
3. Register to production
4. Monitor via MLflow UI

## Deployment

**Local Development**:
```bash
docker compose up --build
```

**Production**:
- Use registered model from MLflow registry
- Load via `mlflow.pyfunc.load_model('models:/ECUAgent/latest')`
- Deploy FastAPI service with model artifacts

## Limitations & Future Work

**Current Limitations**:
- Fixed embedding model (sentence-transformers/all-MiniLM-L6-v2)
- Manual test dataset curation required
- No real-time model updates

**Planned Improvements**:
- [ ] A/B testing framework for models
- [ ] Automatic test case generation
- [ ] Fine-tuned domain embeddings
- [ ] Multi-language support
- [ ] Streaming responses
