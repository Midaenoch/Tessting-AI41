from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app.database import get_db
from app.schemas.user import UserCreate, UserOut
from app.schemas.auth import Token
from app.auth.security import hash_password, verify_password, create_access_token
from app.auth.deps import get_current_user, require_roles

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
async def register(payload: UserCreate):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    existing = await db.users.find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    doc = {
        "email": payload.email.lower(),
        "full_name": payload.full_name,
        "hashed_password": hash_password(payload.password),
        "role": payload.role,
        "is_active": True,
        "created_at": datetime.utcnow(),
    }
    result = await db.users.insert_one(doc)

    return UserOut(
        id=str(result.inserted_id),
        email=doc["email"],
        full_name=doc["full_name"],
        role=doc["role"],
    )


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    user = await db.users.find_one({"email": form.username.lower()})
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token(
        subject=str(user["_id"]),
        role=user.get("role", "viewer"),
    )
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)):
    return UserOut(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
    )


# --- Example: role-restricted endpoints -----------------------------------
@router.get("/admin-only", dependencies=[Depends(require_roles("admin"))])
async def admin_only():
    return {"message": "You are an admin."}


@router.get("/officer-area", dependencies=[Depends(require_roles("admin", "health_officer"))])
async def officer_area():
    return {"message": "Welcome, admin or health officer."}