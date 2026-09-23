from datetime import datetime
from typing import Literal
from pydantic import BaseModel, EmailStr, Field


# The canonical list of roles. Add here if you introduce new ones.
Role = Literal["admin", "health_officer", "viewer"]


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str
    role: Role = "viewer"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: Role


class UserInDB(BaseModel):
    email: EmailStr
    full_name: str
    hashed_password: str
    role: Role = "viewer"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: Role
    created_at: datetime