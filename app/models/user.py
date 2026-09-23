from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class UserInDB(BaseModel):
    email: EmailStr
    full_name: str
    hashed_password: str
    role: str = "viewer"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserPublic(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: str
    created_at: datetime