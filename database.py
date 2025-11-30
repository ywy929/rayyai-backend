from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator, Optional
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConfigurationError
from google.cloud.sql.connector import Connector
load_dotenv()

# Environment detection
ENVIRONMENT = "local"  # Options: "local", "cloud"

# PostgreSQL configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Google Cloud SQL configuration
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")

# MongoDB configuration (optional, for chat messages)
MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

# Create database engine based on environment
if ENVIRONMENT == "cloud" and INSTANCE_CONNECTION_NAME:
    # Google Cloud SQL Connector for production
    def getconn():
        connector = Connector()
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME
        )
        return conn

    engine = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_size=12,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1000
    )
    print(f"Connected to Google Cloud SQL: {INSTANCE_CONNECTION_NAME}")
else:
    # Local PostgreSQL connection for development
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(
        DATABASE_URL,
        pool_size=12,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1000
    )
    print(f"Connected to local PostgreSQL: {DB_HOST}:{DB_PORT}/{DB_NAME}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# MongoDB setup
mongo_client: Optional[MongoClient] = None
mongo_db = None

if MONGODB_ATLAS_CLUSTER_URI:
    mongo_client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    if MONGODB_DB_NAME:
        mongo_db = mongo_client[MONGODB_DB_NAME]
    else:
        try:
            mongo_db = mongo_client.get_default_database()
        except ConfigurationError:
            mongo_db = None


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_mongo_client() -> MongoClient:
    """Return the shared MongoDB client instance."""
    if mongo_client is None:
        raise RuntimeError("MongoDB client is not configured. Set MONGODB_ATLAS_CLUSTER_URI in the environment.")
    return mongo_client


def get_mongo_db():
    """Return the configured MongoDB database instance."""
    if mongo_db is None:
        raise RuntimeError("MongoDB database is not configured. Set MONGODB_DB_NAME or include a default database in the URI.")
    return mongo_db