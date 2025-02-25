from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env = os.getenv("FASTAPI_ENV", "local")
ENV_FILE_PATH = os.path.join(BASE_DIR, f'../../{env}.env')

# Load environment variables from the .env file
load_dotenv(ENV_FILE_PATH)

# Ensure the logs directory exists
LOGS_DIR = os.path.join(BASE_DIR, '../../logs')
os.makedirs(LOGS_DIR, exist_ok=True)


class Settings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str
    POSTGRES_PORT: str
    POSTGRES_DATABASE: str
    DATABASE_URL: Optional[str] = None

    API_PREFIX: str

    ## TODO: 본인 알고리즘 IN/OUTPUT Path 지정 
    W01_OUTPUT_PATH: str
    W03_INPUT_PATH: str
    W03_OUTPUT_PATH: str

    S01_MODEL_PATH: str
    S01_INPUT_PATH: str
    S01_OUTPUT_PATH: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"


settings = Settings()
