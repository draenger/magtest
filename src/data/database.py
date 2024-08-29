import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.models import Base
from dotenv import load_dotenv


class Database:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv("DATABASE_CONNECTION_STRING"))
        self.Session = sessionmaker(bind=self.engine)

    def create_all_tables(self):
        Base.metadata.create_all(self.engine)

    def get_session(self):
        return self.Session()
