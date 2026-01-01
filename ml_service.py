from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from models import Expert, ExpertCategory
import logging
import joblib
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseExpertModel(ABC):
    """
    Abstract Base Class for Expert Recommendation Models.
    Allows easy swapping of algorithms (e.g., TF-IDF vs BERT).
    """
    
    @abstractmethod
    def train(self, experts_data: List[Dict[str, Any]]):
        """
        Train the model with a list of expert dictionaries.
        Each dict should minimally contain 'id', 'name', 'category', 'description'.
        """
        pass

    @abstractmethod
    def predict(self, query: str) -> List[Dict[str, Any]]:
        """
        Predict the best experts for a given query.
        Returns: List of Expert Dicts with scores
        """
        pass

    @abstractmethod
    def get_by_category(self, category_name: str) -> List[Dict[str, Any]]:
        """
        Return all experts belonging to a specific category.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        pass

class TfidfExpertModel(BaseExpertModel):
    """
    Concrete implementation using TF-IDF and Cosine Similarity.
    Simple, fast, and effective for keyword-heavy matching.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.experts_data = []

    def _create_corpus(self, expert: Dict[str, Any]) -> str:
        """Combine relevant fields into a single text string."""
        # Weighted combination: Category matches are crucial, description provides context.
        # Repeating category name boosts its importance
        return f"{expert['category']} {expert['category']} {expert['name']} {expert.get('description', '')} {expert.get('bio', '')}"

    def train(self, experts_data: List[Dict[str, Any]]):
        logger.info(f"Training model with {len(experts_data)} experts...")
        self.experts_data = experts_data
        
        if not experts_data:
            logger.warning("No experts data found. Model will not be trained.")
            return

        corpus = [self._create_corpus(exp) for exp in experts_data]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info("Model training completed.")

    def save(self, path: str):
        logger.info(f"Saving model to {path}...")
        joblib.dump({
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'experts_data': self.experts_data
        }, path)
        logger.info("Model saved.")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        logger.info(f"Loading model from {path}...")
        try:
            data = joblib.load(path)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.experts_data = data['experts_data']
            logger.info("Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, query: str) -> List[Dict[str, Any]]:
        if self.tfidf_matrix is None or not self.experts_data:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get all scores above threshold
        results = []
        for idx, score in enumerate(similarities):
            if score >= 0.1:
                expert = self.experts_data[idx].copy()
                expert['score'] = float(score)
                results.append(expert)

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def get_by_category(self, category_name: str) -> List[Dict[str, Any]]:
        if not self.experts_data:
            return []
            
        results = []
        for expert in self.experts_data:
            # Flexible matching for category name
            if category_name.lower() in expert.get('category', '').lower():
                exp = expert.copy()
                exp['score'] = 0.0  # Fallback score
                results.append(exp)
        return results

class ExpertService:
    """
    Service layer to handle Database interactions and Model management.
    Singleton-like behavior can be managed by FastAPI's dependency injection.
    """
    def __init__(self):
        # We can easily swap this out for a different model implementation
        self.model: BaseExpertModel = TfidfExpertModel()
        self.is_trained = False
        self.model_path = "expert_model.pkl"

    def initialize_model(self, db: Session):
        """Try loading from disk, otherwise train."""
        if self.model.load(self.model_path):
            self.is_trained = True
            logger.info("Initialized model from disk.")
            return
        
        logger.info("No saved model found. Training from DB...")
        self.train_model(db)

    def train_model(self, db: Session):
        """Fetch data from DB, train, and save."""
        # Query Experts joined with Categories
        experts = db.query(Expert).join(ExpertCategory).filter(Expert.status == 'approved', Expert.is_active == True).all()
        
        data = []
        for expert in experts:
            # Flatten the data structure for the model
            data.append({
                "id": expert.id,
                "name": expert.name,
                # primary_specialty
                # secondary_specialty
                "category": expert.category.name if expert.category else "General",
                "category_id": expert.category.id if expert.category else None,
                "description": expert.category.description if expert.category else "",
                "bio": expert.bio if expert.bio else ""
            })
        
        self.model.train(data)
        self.model.save(self.model_path)
        self.is_trained = True
        return len(data)

    def get_recommendations(self, query: str) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Returns a tuple: (list_of_experts, is_fallback)
        """
        if not self.is_trained:
            # Fallback or auto-trigger training could happen here
            logger.warning("Model queried before training!")
            return [], False
            
        recommendations = self.model.predict(query)
        if recommendations:
            return recommendations, False
            
        # Fallback to General category
        return self.model.get_by_category("General"), True

# Global Instance
expert_service = ExpertService()
