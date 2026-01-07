from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from database import engine, Base, get_db
from models import Base as ModelBase
from schemas import UserQuery, ExpertListResponse, ExpertMatch
from ml_service import expert_service
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

ModelBase.metadata.create_all(bind=engine)

app = FastAPI(title="Expert Recommendation API", description="AI-powered expert finding service")

chat_sessions = {}

# Chatbot Models
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    escalate: bool = False

@app.on_event("startup")
def startup_event():
    from database import SessionLocal
    db = SessionLocal()
    try:
        expert_service.initialize_model(db)
        print("Model initialized successfully on startup.")
    except Exception as e:
        print(f"Startup: Failed to initialize model. Error: {e}")
    finally:
        db.close()

# ========================
# RECOMMEND ENDPOINT - FULLY FIXED
# ========================
@app.post("/api/recommend", response_model=ExpertListResponse)
def get_expert_recommendation(query: UserQuery, db: Session = Depends(get_db)):
    try:
        if not expert_service.is_trained:
            count = expert_service.train_model(db)
            if count == 0:
                raise HTTPException(status_code=503, detail="No experts available")

        raw_recommendations = expert_service.get_recommendations(query.question)
        
        print(f"DEBUG: Raw recommendations type: {type(raw_recommendations)}")
        if raw_recommendations:
            print(f"DEBUG: Sample: {raw_recommendations[0] if isinstance(raw_recommendations, (list, tuple)) else raw_recommendations}")

        # Fix: raw_recommendations tuple hai → uska [0] list of dicts hai
        if isinstance(raw_recommendations, tuple) and len(raw_recommendations) > 0:
            recommendations = raw_recommendations[0]
        elif isinstance(raw_recommendations, list):
            recommendations = raw_recommendations
        else:
            recommendations = []

        expert_matches = []
        for rec in recommendations:
            if not isinstance(rec, dict):
                continue  # Skip invalid
            expert_matches.append(ExpertMatch(
                expert_id=rec.get('id', 0),
                expert_name=rec.get('name', 'Unknown Expert'),
                category_id=rec.get('category_id', 0),
                category_name=rec.get('category', 'General'),
                confidence_score=rec.get('score', 0.0),
                reason=f"Matched in '{rec.get('category', 'General')}' expertise."
            ))

        # Agar ExpertListResponse mein isGeneralCategory field hai to add kar do, warna schemas.py se hata do
        return ExpertListResponse(
            recommendations=expert_matches,
            total_found=len(expert_matches),
            message="Top experts found." if expert_matches else "No specific match, connecting to general expert."
            # Agar schemas mein isGeneralCategory hai to ye add kar:
            # isGeneralCategory=True if not expert_matches else False
        )

    except Exception as e:
        print(f"Recommend endpoint error: {e}")
        import traceback
        traceback.print_exc()
        # Safe fallback
        return ExpertListResponse(
            recommendations=[],
            total_found=0,
            message="Service temporarily unavailable. Connecting you to an expert."
        )

@app.post("/api/train")
def trigger_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    background_tasks.add_task(expert_service.train_model, db)
    return {"message": "Training started in background."}

@app.get("/")
def root():
    return {"message": "Expert Recommendation API is running."}

# ========================
# CHAT ENDPOINT (already working fine)
# ========================
SYSTEM_PROMPT = """
You are AskExpert Assistant, a friendly and professional chatbot.
Your goal is to deeply understand the user's issue by asking clarifying questions.
Examples:
- Which version of the software are you using?
- What operating system?
- Exact error message?
- What have you tried?

Be polite and concise.
After 3–4 exchanges, you MUST say:
"Thanks for the details! I'll connect you to a specialist now."

Do NOT give final solutions yourself — always escalate.
"""

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.session_id or request.session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        bot_turns = 0
    else:
        session_id = request.session_id
        session = chat_sessions[session_id]
        history = session["history"]
        bot_turns = session["bot_turns"]

    history.append({"role": "user", "content": request.user_message})

    if bot_turns >= 6:
        escalate_msg = "Thanks for the details! I'll connect you to a specialist now."
        chat_sessions[session_id] = {
            "history": history + [{"role": "assistant", "content": escalate_msg}],
            "bot_turns": bot_turns + 1
        }
        return ChatResponse(session_id=session_id, response=escalate_msg, escalate=True)

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=history,
            temperature=0.7,
            max_tokens=500,
        )
        bot_response = completion.choices[0].message.content.strip()
    except Exception as e:
        bot_response = "Sorry, technical issue. Connecting you to a human expert."
        print(f"Groq error: {e}")

    history.append({"role": "assistant", "content": bot_response})
    bot_turns += 1
    chat_sessions[session_id] = {"history": history, "bot_turns": bot_turns}

    escalate_keywords = ["expert", "specialist", "connect", "human"]
    escalate = any(keyword in bot_response.lower() for keyword in escalate_keywords)

    return ChatResponse(
        session_id=session_id,
        response=bot_response,
        escalate=escalate or (bot_turns >= 6)
    )