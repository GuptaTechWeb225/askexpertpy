from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, DECIMAL
from sqlalchemy.orm import relationship
from database import Base

class ExpertCategory(Base):
    __tablename__ = "expert_categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    icon = Column(String(255))
    primary_specialty = Column(String(255))
    description = Column(Text)
    # Additional fields from SQL
    is_refundable = Column(Boolean, default=False)
    monthly_subscription_fee = Column(DECIMAL(10, 2))
    expert_fee = Column(DECIMAL(10, 2))

    # Relationship to experts
    experts = relationship("Expert", back_populates="category")

class Expert(Base):
    __tablename__ = "experts"

    id = Column(Integer, primary_key=True, index=True)
    f_name = Column(String(255), nullable=False)
    l_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(255))
    
    # Foreign Key matches table 'expert_categories'
    category_id = Column(Integer, ForeignKey("expert_categories.id"))
    
    # Specialties acting as bio/skills
    primary_specialty = Column(String(255))
    secondary_specialty = Column(String(255))
    
    experience = Column(Integer)
    status = Column(String(255), default="pending")
    is_active = Column(Boolean, default=False)
    
    # Relationship to category
    category = relationship("ExpertCategory", back_populates="experts")

    @property
    def name(self):
        return f"{self.f_name} {self.l_name}"

    @property
    def bio(self):
        return f"{self.primary_specialty or ''} {self.secondary_specialty or ''}"
