from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from database import engine, Base, get_db
from models import Base as ModelBase
from schemas import UserQuery, ExpertListResponse, ExpertMatch
from ml_service import expert_service

# Create Tables (for dev purposes - in prod use Alembic)
ModelBase.metadata.create_all(bind=engine)

app = FastAPI(title="Expert Recommendation API", description="AI-powered expert finding service")

@app.on_event("startup")
def startup_event():
    """
    On startup, we attempt to train the model.
    We need a DB session for this.
    """
    # Create a new session specifically for startup training
    from database import SessionLocal
    db = SessionLocal()
    try:
        expert_service.initialize_model(db)
    except Exception as e:
        print(f"Startup: Failed to initialize model. Error: {e}")
    finally:
        db.close()

@app.post("/api/recommend", response_model=ExpertListResponse)
def get_expert_recommendation(query: UserQuery, db: Session = Depends(get_db)):
    """
    Receives a user question and returns all matching Experts.
    """
    if not expert_service.is_trained:
        # Attempt to train if not trained
        count = expert_service.train_model(db)
        if count == 0:
             raise HTTPException(status_code=503, detail="No experts available in database to recommend.")

    recommendations, is_fallback = expert_service.get_recommendations(query.question)

    expert_matches = []
    for rec in recommendations:
        expert_matches.append(ExpertMatch(
            expert_id=rec['id'],
            expert_name=rec['name'],
            category_id=rec.get('category_id', 0),
            category_name=rec['category'],
            confidence_score=rec['score'],
            reason=f"Strong match in '{rec['category']}' expertise. Score based on keyword overlap in category, bio, and description." if not is_fallback else "General category expert recommended."
        ))

    if is_fallback:
        msg = "No specific matches found. Showing General experts."
    elif len(expert_matches) > 0:
        msg = "Top matching experts based on your query"
    else:
        msg = "No matching experts found for your query."

    return ExpertListResponse(
        recommendations=expert_matches,
        total_found=len(expert_matches),
        isGeneralCategory=is_fallback,
        message=msg
    )

@app.post("/api/train")
def trigger_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Manually trigger model retraining (e.g., after adding new experts).
    Runs in background to avoid blocking.
    """
    background_tasks.add_task(expert_service.train_model, db)
    return {"message": "Training started in background."}

@app.get("/")
def root():
    return {"message": "Expert Recommendation API is running. Schema updated."}
