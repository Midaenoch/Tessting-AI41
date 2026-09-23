from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client: AsyncIOMotorClient = None
db = None


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        socketTimeoutMS=20000,
    )
    db = client[settings.MONGO_DB]

    try:
        await client.admin.command("ping")
        await db.users.create_index("email", unique=True)
        await db.alerts.create_index([("reported_date", -1)])
        await db.cases.create_index([("year", 1), ("month", 1)])
        await db.cases.create_index([("state", 1), ("lga", 1)])
        await db.cases.create_index([("result", 1)])
        await db.lga_synthetic.create_index([("state", 1), ("lga", 1)])
        await db.lga_synthetic.create_index([("case_status", 1)])
        print(f"[db] Connected to MongoDB: {settings.MONGO_DB}")
    except Exception as e:
        print(f"[db] WARNING: MongoDB connection failed: {e}")


async def close_db():
    global client
    if client:
        client.close()
        client = None
        print("[db] MongoDB connection closed.")


def get_db():
    return db