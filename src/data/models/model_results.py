from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True)
    prepared_question_id = Column(Integer, ForeignKey("prepared_questions.id"))
    model_name = Column(String, nullable=False)
    creation_date = Column(DateTime, default=datetime.utcnow)
    execution_date = Column(DateTime)
    status = Column(
        String, default="initialized"
    )  # initialized, running, completed, failed
    score = Column(Float)

    estimated_in_tokens = Column(Integer)
    estimated_out_tokens = Column(Integer)
    estimated_in_cost = Column(Float)
    estimated_out_cost = Column(Float)

    response = Column(String)
    actual_in_tokens = Column(Integer)
    actual_out_tokens = Column(Integer)
    actual_in_cost = Column(Float)
    actual_out_cost = Column(Float)
    execution_time = Column(Float)

    prepared_question = relationship("PreparedQuestion", back_populates="model_results")

    def __repr__(self):
        return f"<ModelResult(id={self.id}, model_name='{self.model_name}', status='{self.status}')>"
