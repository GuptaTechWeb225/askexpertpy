from pydantic import BaseModel
from typing import Optional, List

# --- Request Models ---
class UserQuery(BaseModel):
    question: str

# --- Response Models ---
class ExpertMatch(BaseModel):
    expert_id: int
    expert_name: str
    category_id: int
    category_name: str
    confidence_score: float
    reason: str

    class Config:
        from_attributes = True

class ExpertListResponse(BaseModel):
    recommendations: List[ExpertMatch]
    total_found: int
    isGeneralCategory: Optional[bool] = None  # <-- Optional + default None
    message: str