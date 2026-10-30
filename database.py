import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    db_password = os.getenv("SUPEBASE_PASSWORD") or os.getenv("SUPABASE_PASSWORD") or ""
    encoded_pw = quote_plus(db_password)
    db_user = os.getenv("DB_USER", "postgres.yspgeqqsqrzirlvjessn")
    db_host = os.getenv("DB_HOST", "aws-0-eu-west-1.pooler.supabase.com")
    db_port = os.getenv("DB_PORT", "6543")
    db_name = os.getenv("DB_NAME", "postgres")
    DATABASE_URL = f"postgresql+psycopg2://{db_user}:{encoded_pw}@{db_host}:{db_port}/{db_name}"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
