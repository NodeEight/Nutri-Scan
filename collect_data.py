import streamlit as st
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from datetime import datetime
from database_schema import *
from cloudinary_utils import upload_image_to_cloudinary, test_cloudinary_connection
import os
from dotenv import load_dotenv
from sqlalchemy import text

# Load environment variables
load_dotenv()

# Initialize database
try:
    init_db()
    print("✅ Database initialized successfully")
except Exception as e:
    print(f"❌ Error initializing database: {str(e)}")

def check_db_connection():
    """Test the database connection"""
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True, "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="Nutri-Scan Data Collection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-header {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# --- The following SQLite setup is NOT needed when using PostgreSQL ---
# SQLALCHEMY_DATABASE_URL = 'sqlite:///./malnutrition.db'
# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# Base.metadata.create_all(bind=engine)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# ---------------------------------------------------------------------

def validate_required_fields(data):
    """Validate that all required fields are filled and have valid values"""
    required_fields = {
        'age': 'Age',
        'weight': 'Weight',
        'height': 'Height',
        'mid_lower_hand_circumference': 'Hand Circumference',
        'location': 'Location'
    }
    
    missing_fields = []
    invalid_fields = []
    
    for field, label in required_fields.items():
        # Special handling for numeric fields
        if field in ['age', 'weight', 'height', 'mid_lower_hand_circumference']:
            value = data.get(field)
            if value is None:
                missing_fields.append(label)
            elif value <= 0:
                invalid_fields.append(label)
        # Handle string fields
        elif not data.get(field):
            missing_fields.append(label)
    
    errors = []
    if missing_fields:
        errors.append(f"Please fill in the following required fields: {', '.join(missing_fields)}")
    if invalid_fields:
        errors.append(f"Please enter values greater than 0 for: {', '.join(invalid_fields)}")
    
    return errors

def create_image_preview_section():
    """Create a section to preview all uploaded images"""
    st.markdown("### 📸 Image Preview")
    cols = st.columns(4)
    
    images = {
        'Face': st.session_state.get('face_image'),
        'Hair': st.session_state.get('hair_image'),
        'Hands': st.session_state.get('hands_image'),
        'Legs': st.session_state.get('leg_image')
    }
    
    for idx, (label, image) in enumerate(images.items()):
        with cols[idx]:
            st.markdown(f"**{label}**")
            if image:
                st.image(image, width=150, caption=f'{label} image')
            else:
                st.markdown("*No image uploaded*")

def main():
    # Initialize session state for progress tracking
    if 'form_progress' not in st.session_state:
        st.session_state.form_progress = 0
    
    # Sidebar
    with st.sidebar:
        st.image('images/img.PNG', use_container_width=True)
        st.title('Nutri-Scan')
        st.markdown('---')
        
        # Check database connection
        db_status, db_message = check_db_connection()
        if not db_status:
            st.error("⚠️ Database connection error")
            st.info(f"Please check your database configuration: {db_message}")
        else:
            st.success("✅ Database connected")
        
        # Check Cloudinary connection
        connection_status, message = test_cloudinary_connection()
        if not connection_status:
            st.warning("⚠️ Cloudinary connection not configured")
            st.info("Configure credentials in .streamlit/secrets.toml")
        
        classes = ['Malnourish', 'Nourished']
        selected_class = st.radio('👥 Select Patient Category', classes)
        
        st.markdown('---')
        st.markdown(f"**Selected Category:** {selected_class}")
        
        # Display progress
        st.markdown("### 📊 Form Progress")
        st.progress(st.session_state.form_progress)
        
    # Main content
    st.title("🏥 Malnutrition Data Collection")
    st.markdown(f"### Collecting data for {selected_class} patient")
    
    with st.form("user_form", clear_on_submit=True):
        # Basic Information Section
        st.markdown("## 📋 Basic Information")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(
                "Age (months)", 
                min_value=0.0, 
                max_value=150.0, 
                value=0.0, 
                step=0.5,
                help="Enter the patient's age in months"
            )
            
            weight = st.number_input(
                "Weight (kg)", 
                min_value=0.0, 
                max_value=100.0, 
                value=0.0, 
                step=0.1,
                help="Enter the patient's weight in kilograms"
            )
        
        with col2:
            height = st.number_input(
                "Height (cm)", 
                min_value=0.0, 
                max_value=200.0, 
                value=0.0, 
                step=0.1,
                help="Enter the patient's height in centimeters"
            )
            
            hand_circumference = st.number_input(
                "Mid Lower Hand Circumference (cm)", 
                min_value=0.0, 
                max_value=50.0, 
                value=0.0, 
                step=0.1,
                help="Measure and enter the mid-lower arm circumference"
            )
        
        location = st.text_input(
            "Location",
            help="Enter the location where data is being collected"
        )
        
        # Medical Indicators Section
        st.markdown("## 🔍 Medical Indicators")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            skin_type = st.selectbox(
                "Skin Condition",
                ['Dry and scaly', 'Rash'],
                help="Select the observed skin condition"
            )
            
            hair_type = st.selectbox(
                "Hair Condition",
                ['Dry flaky scalp', 'Thin sparse hair'],
                help="Select the observed hair condition"
            )
            
            eyes_type = st.selectbox(
                "Eye Condition",
                ['Jaundice', 'Dry sour eyes'],
                help="Select the observed eye condition"
            )
        
        with col2:
            oedema = st.selectbox(
                "Oedema Present",
                ['no', 'yes'],
                help="Select if oedema is present"
            )
            
            angular_stomatitis = st.selectbox(
                "Angular Stomatitis",
                ['no', 'yes'],
                help="Select if angular stomatitis is present"
            )
            
            cheilosis = st.selectbox(
                "Cheilosis",
                ['no', 'yes'],
                help="Select if cheilosis is present"
            )
            
            bowlegs = st.selectbox(
                "Bowlegs Present",
                ['no', 'yes'],
                help="Select if bowlegs condition is present"
            )
        
        if selected_class == 'Malnourish':
            type_of_malnutrition = st.text_input(
                "Type of Malnutrition",
                help="Specify the type of malnutrition observed"
            )
        
        # Image Upload Section
        st.markdown("## 📸 Image Upload")
        st.markdown("---")
        st.info("Please upload clear, well-lit images for accurate documentation")
        
        col1, col2 = st.columns(2)
        with col1:
            face_image = st.file_uploader(
                "Face Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear front-facing image",
                key="face_image"
            )
            
            hair_image = st.file_uploader(
                "Hair Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the scalp/hair",
                key="hair_image"
            )
        
        with col2:
            hands_image = st.file_uploader(
                "Hands Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of both hands",
                key="hands_image"
            )
            
            leg_image = st.file_uploader(
                "Leg Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the legs",
                key="leg_image"
            )
        
        # Preview section for all images
        if any([face_image, hair_image, hands_image, leg_image]):
            st.markdown("### Image Preview")
            st.markdown("---")
            create_image_preview_section()
        
        # Submit Button Section
        st.markdown("---")
        submitted = st.form_submit_button("💾 Save Patient Data", type='primary', use_container_width=True)
        
        if submitted:
            # Collect form data
            form_data = {
                'age': age if age is not None else None,
                'weight': weight if weight is not None else None,
                'height': height if height is not None else None,
                'mid_lower_hand_circumference': hand_circumference if hand_circumference is not None else None,
                'location': location,
                'skin_type': skin_type,
                'hair_type': hair_type,
                'eyes_type': eyes_type,
                'oedema': oedema,
                'angular_stomatitis': angular_stomatitis,
                'cheilosis': cheilosis,
                'bowlegs': bowlegs
            }
            
            # Validate required fields
            validation_errors = validate_required_fields(form_data)
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return
            
            # Validate image uploads
            if not all([face_image, hair_image, hands_image, leg_image]):
                st.error("Please upload all required images")
                return
            
            try:
                with st.spinner("📤 Uploading images and saving data..."):
                    # Upload images to Cloudinary
                    folder_prefix = "nutri-scan/malnourish" if selected_class == "Malnourish" else "nutri-scan/nourished"
                    
                    face_image_url = upload_image_to_cloudinary(face_image, folder=f"{folder_prefix}/face")
                    hair_image_url = upload_image_to_cloudinary(hair_image, folder=f"{folder_prefix}/hair")
                    hands_image_url = upload_image_to_cloudinary(hands_image, folder=f"{folder_prefix}/hands")
                    leg_image_url = upload_image_to_cloudinary(leg_image, folder=f"{folder_prefix}/legs")
                    
                    if not all([face_image_url, hair_image_url, hands_image_url, leg_image_url]):
                        st.error("Failed to upload one or more images. Please try again.")
                        return
                    
                    # Add image URLs to form data
                    form_data.update({
                        'face_image_url': face_image_url,
                        'hair_image_url': hair_image_url,
                        'hands_image_url': hands_image_url,
                        'leg_image_url': leg_image_url
                    })
                    
                    # Add type of malnutrition if applicable
                    if selected_class == 'Malnourish':
                        form_data['type_of_malnutrition'] = type_of_malnutrition
                    
                    # Save to database
                    with SessionLocal() as session:
                        if selected_class == 'Malnourish':
                            user = Malnurish_data(**form_data)
                        else:
                            user = Nurish_data(**form_data)
                        session.add(user)
                        session.commit()
                    
                    st.success("✅ Patient data saved successfully!")
                    st.balloons()
                    
                    # Reset form progress
                    st.session_state.form_progress = 0
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    main()