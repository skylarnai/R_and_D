# RAG Evaluation Suite (RAGAS Framework)

Streamlined evaluation engine for testing the **Medical RAG Chatbot** (Gemma + pgvector + LangGraph CRAG).

Evaluates queries using the **4 Core RAGAS Metrics**:
1. **Faithfulness** ($\ge 0.85$): Factual consistency against retrieved chunks (hallucination resistance).
2. **Answer Relevance** ($\ge 0.80$): Directness and adherence to the user query.
3. **Context Precision** ($\ge 0.75$): Ranking quality and signal-to-noise ratio of retrieved knowledge.
4. **Context Recall** ($\ge 0.75$): Extent to which retrieved contexts contain the ground-truth facts.

---

## Production Testing via Postman / REST API

FastAPI exposes a single on-the-fly evaluation endpoint:

### `POST /evaluate/single`

* **URL**: `http://127.0.0.1:8000/evaluate/single`
* **Method**: `POST`
* **Headers**: `Content-Type: application/json`
* **Body** (`raw JSON`):
  ```json
  {
    "question": "What is the patient's Hemoglobin level?",
    "reference": "The patient's Hemoglobin is 14.5 g/dL (reference range: 13.0 - 16.5 g/dL).",
    "session_id": "8350581e-f557-494c-b918-6dbed5356eba",
    "top_k": 5
  }
  ```

* **Response**:
  ```json
  {
    "status": "success",
    "session_id": "8350581e-f557-494c-b918-6dbed5356eba",
    "question": "What is the patient's Hemoglobin level?",
    "reference": "The patient's Hemoglobin is 14.5 g/dL...",
    "generated_answer": "According to the report, the Hemoglobin is 14.5 g/dL...",
    "answer_source": "documents",
    "retrieved_contexts_count": 5,
    "scores": {
      "faithfulness": 1.0,
      "answer_relevancy": 0.94,
      "context_precision": 1.0,
      "context_recall": 1.0
    },
    "threshold_status": [
      {
        "name": "Faithfulness",
        "score": 1.0,
        "threshold": 0.85,
        "passed": true
      }
    ]
  }
  ```
