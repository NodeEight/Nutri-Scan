import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st 

load_dotenv()
# PostgreSQL Database Configuration
DATABASE_URL = (
        st.secrets.get("DATABASE_URL") or 
        os.getenv("DATABASE_URL") or 
        "your_cloud_name"
    )

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Table for malnourished data
class Malnurish_data(Base):
    __tablename__ = 'malnurish'
    id = Column(Integer, primary_key=True, index=True)
    age = Column(Float)  # Changed to Float for more precise measurements
    weight = Column(Float)
    height = Column(Float)
    mid_lower_hand_circumference = Column(Float)
    skin_type = Column(String)
    hair_type = Column(String)
    eyes_type = Column(String)
    date_created = Column(DateTime, default=datetime.now)
    type_of_malnutrition = Column(String)
    oedema = Column(String)
    angular_stomatitis = Column(String)
    cheilosis = Column(String)
    bowlegs = Column(String)
    location = Column(String)
    face_image_url = Column(String)
    hair_image_url = Column(String)
    hands_image_url = Column(String)
    leg_image_url = Column(String)

# Table for nourished data
class Nurish_data(Base):
    __tablename__ = 'nurish'
    id = Column(Integer, primary_key=True, index=True)
    age = Column(Float)  # Changed to Float for more precise measurements
    weight = Column(Float)
    height = Column(Float)
    mid_lower_hand_circumference = Column(Float)
    skin_type = Column(String)
    hair_type = Column(String)
    eyes_type = Column(String)
    date_created = Column(DateTime, default=datetime.now)
    oedema = Column(String)
    angular_stomatitis = Column(String)
    cheilosis = Column(String)
    bowlegs = Column(String)
    location = Column(String)
    face_image_url = Column(String)
    hair_image_url = Column(String)
    hands_image_url = Column(String)
    leg_image_url = Column(String)

def init_db():
    """Initialize the database, creating all tables."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()