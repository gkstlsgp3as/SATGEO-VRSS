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

    M01_INPUT_PATH: str
    M01_OUTPUT_PATH: str

    M14_INPUT_PATH: str
    M14_OUTPUT_PATH: str

    N04_INPUT_PATH: str
    N04_OUTPUT_PATH: str
    N04_META_FILE: str
    N04_MARINE_ZONE_PATH: str
    N04_AIS_PATH: str
    N04_WHOLE_FACILITEIS_PATH: str
    N04_CONFIRMED_FACILITIES_PATH: str
    N04_VESSEL_DETECTION_PATH: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"


settings = Settings()
