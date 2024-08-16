import os
import json
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        
        with open('./config/default_config.json', 'r') as json_file:
            default_values = json.load(json_file)

        for key, default_value in default_values.items():
            setattr(self, key, os.getenv(key, default_value))

    def get_database_url(self):
        return f"{self.DATABASE_ENGINE}:///{self.DATABASE_NAME}"

# Przykład użycia
if __name__ == "__main__":
    print(f"Database Name: {config.DATABASE_NAME}")
    print(f"Database Engine: {config.DATABASE_ENGINE}")
    print(f"Some Setting: {config.SOME_SETTING}")
    print(f"Database URL: {config.get_database_url()}")
