"""
Database engine and session management for Finance Market Intelligence.
Supports SQLite (zero-config local) and MySQL/PostgreSQL via DATABASE_URL.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_URL
from api.models import Base

# Setup connect_args for SQLite
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables in the database if they don't already exist."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables verified and initialized")


def get_db():
    """Dependency that yields a database session and safely closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
