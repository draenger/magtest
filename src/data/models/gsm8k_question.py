from sqlalchemy import Column, Integer, String
from .base import Base


class GSM8KQuestion(Base):
    __tablename__ = "gsm8k_questions"

    id = Column(Integer, primary_key=True)
    question = Column(String, nullable=False)
    full_solution = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    category = Column(String, nullable=False)  # 'socratic', 'standard', 'unknown'
    data_type = Column(String, nullable=False)  # 'train', 'test', 'example'

    def __repr__(self):
        return f"<GSM8KQuestion(id={self.id}, category='{self.category}', data_type='{self.data_type}')>"
