
## Getting Started
### Prerequisites
- Python 3.12+
- MySQL Database

### Installation

1. Navigate to the `expert-python-api` directory:
   ```bash
   cd expert-python-api
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your MySQL database is running and configured (check `database.py` for connection details).

### Running the API

Start the server using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000/docs`.

### API Endpoints

#### 1. Trigger Training
**POST** `/api/train`

#### 2. Get Recommendation
**POST** `/api/recommend`

fetches the best matching expert for a given user query.

**Request Body:**
```json
{
  "question": "I need help with machine learning models."
}
```

**Response:**
```json
{
  "expert_name": "Alice Smith",
  "category_name": "Data Science",
  "confidence_score": 0.95,
  "recommendation_reason": "Matched based on expertise in Data Science."
}
```


Manually triggers the model training process. This runs in the background.

#### 3. Health Check
**GET** `/`

Returns a status message indicating the API is running.

