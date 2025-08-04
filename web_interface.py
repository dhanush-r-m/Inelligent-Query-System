from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
from intelligent_query_system import IntelligentQuerySystem
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Initialize the query system
gemini_api_key = os.environ.get("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

query_system = IntelligentQuerySystem(
    cache_dir="./query_system_cache",
    model="gemini-1.0-pro"
)

# Load knowledge base on startup
try:
    query_system.load_knowledge_base()
except Exception as e:
    document_paths = [str(p) for p in Path("Documents").glob("*")]
    if document_paths:
        print("Building knowledge base from documents...")
        query_system.build_knowledge_base(document_paths)
    else:
        print("No documents found to build knowledge base.")

app = FastAPI(title="Intelligent Query-Retrieval System", version="1.0.0")
security = HTTPBearer()
API_TOKEN = "f4bd0df7f630c9d41dac1108ebed049061fdb6b329ef7106bec52e1be260f77e"

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

class QueryRequest(BaseModel):
    questions: List[str]
    n_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    token: str = Depends(verify_token)
    ):
    """Query the retrieval system"""
    try:
        answers = []
        for question in request.questions:
            response = query_system.query(question)
            answers.append(response.answer if hasattr(response, 'answer') else str(response))
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)