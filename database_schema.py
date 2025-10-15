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
# data to be stored in the database
# form_data = {
#     'age': age if age is not None else None,
#     'weight': weight if weight is not None else None,
#     'height': height if height is not None else None,
#     'mid_lower_hand_circumference': hand_circumference if hand_circumference is not None else None,
#     'location': location,
#     'name': name,
#     "gender": gender,
#     "caregiver_name": caregiver_name,
#     "phone_number": phone_number,
#     "secondary_contact": secondary_contact,
#     "region": region,
#     "town": town,
#     "community": community,
#     "custom_notes": custom_notes,
#     "health_facility": health_facility,
#     'oedema': 'Yes' if st.session_state.get('Swelling in feet/ankles (edema)') else 'No',
#     'angular_stomatitis': 'Yes' if st.session_state.get('Angular cracks on mouth (cheilitis)') else 'No',
#     'cheilosis': 'Yes' if st.session_state.get('Glossy or pale tongue') else 'No',
#     'potbelly': 'Yes' if st.session_state.get('  Potbelly appearance') else 'No',
#     'skin_type': 'Dry, scaly skin' if st.session_state.get('Dry, scaly skin') else 'Normal',
#     'hair_type': 'Brittle / Discolored hair' if st.session_state.get('Brittle / Discolored hair') else 'Normal',
#     'eyes_type': 'Pale conjunctiva' if st.session_state.get('Pale conjunctiva (eyes)') else 'Normal',
#     'face_image_url': face_image_url,
#     'front_view_image_url': front_view_image_url,
#     'arm_muac_image_url': arm_muac_image_url,
#     'hands_image_url': hands_image_url,
#     'leg_image_url': leg_image_url,
#     'back_view_image_url': back_view_image_url,
#     'side_profile_image_url': side_profile_image_url
# }

# Table for malnourished data
class Malnurish_data(Base):
    __tablename__ = 'malnurish'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Float)  # Changed to Float for more precise measurements
    weight = Column(Float)
    height = Column(Float)
    hemoglobin = Column(Float)
    mid_lower_hand_circumference = Column(Float)
    town = Column(String)
    community = Column(String)
    gender = Column(String)
    caregiver_name = Column(String)
    phone_number = Column(String)
    secondary_contact = Column(String)
    region = Column(String)
    health_facility = Column(String)   
    skin_type = Column(String)
    hair_type = Column(String)
    eyes_type = Column(String)
    custom_notes = Column(String)
    date_created = Column(DateTime, default=datetime.now)
    type_of_malnutrition = Column(String)
    oedema = Column(String)
    angular_stomatitis = Column(String)
    cheilosis = Column(String)
    bowlegs = Column(String)
    location = Column(String)
    face_image_url = Column(String)
    front_view_image_url = Column(String)
    hands_image_url = Column(String)
    leg_image_url = Column(String)
    arm_muac_image_url = Column(String)
    back_view_image_url = Column(String)
    side_profile_image_url = Column(String)

# Table for nourished data
class Nurish_data(Base):
    __tablename__ = 'nurish'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Float)  # Changed to Float for more precise measurements
    weight = Column(Float)
    height = Column(Float)
    hemoglobin = Column(Float)
    mid_lower_hand_circumference = Column(Float)
    skin_type = Column(String)
    hair_type = Column(String)
    eyes_type = Column(String)
    town = Column(String)
    community = Column(String)
    gender = Column(String)
    caregiver_name = Column(String)
    phone_number = Column(String)
    secondary_contact = Column(String)
    custom_notes = Column(String)
    region = Column(String)
    health_facility = Column(String)
    date_created = Column(DateTime, default=datetime.now)
    oedema = Column(String)
    angular_stomatitis = Column(String)
    cheilosis = Column(String)
    bowlegs = Column(String)
    location = Column(String)
    face_image_url = Column(String)
    front_view_image_url = Column(String)
    hands_image_url = Column(String)
    leg_image_url = Column(String)
    arm_muac_image_url = Column(String)
    back_view_image_url = Column(String)
    side_profile_image_url = Column(String)

def init_db():
    """Initialize the database, creating all tables."""
    # Base.metadata.drop_all(bind=engine)  # Drop existing tables for a fresh start
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()