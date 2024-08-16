from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import Config
from .models import Base

class Database:
    def __init__(self):
        config = Config()
        self.engine = create_engine(config.get_database_url())
        self.Session = sessionmaker(bind=self.engine)

    def create_all_tables(self):
        Base.metadata.create_all(self.engine)

    def get_session(self):
        return self.Session()
