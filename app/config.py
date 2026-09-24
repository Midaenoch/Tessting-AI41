# app/config.py
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    MONGO_URI: str = "mongodb+srv://ozchange2002_db_user:nor6BB5u3YhdInf9@cluster0.yqnvz3b.mongodb.net/ai4lassa?retryWrites=true&w=majority"
    MONGO_DB: str = "ai4lassa"
    JWT_SECRET: str = "change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Comma-separated list — each entry MUST end with a comma except the last.
    CORS_ORIGINS: str = (
        "http://localhost:3000,"
        "https://ai4lassa-phi.vercel.app,"
        "https://ai4lassa-test.vercel.app"
    )

    class Config:
        env_file = ".env"


settings = Settings()