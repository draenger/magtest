from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime


class PreparedQuestion(Base):
    __tablename__ = "prepared_questions"

    id = Column(Integer, primary_key=True)
    creation_date = Column(DateTime, default=datetime.utcnow)
    test_session_id = Column(String, nullable=False)
    benchmark_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    query = Column(String, nullable=False)
    correct_answer = Column(String, nullable=False)

    model_results = relationship("ModelResult", back_populates="prepared_question")

    def __repr__(self):
        return f"<PreparedQuestion(id={self.id}, benchmark_name='{self.benchmark_name}', category='{self.category}')>"
