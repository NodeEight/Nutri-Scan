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

def yes_no(key):
    return 'Yes' if st.session_state.get(key) else 'No'


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
    
    images = {
        'Face Close-Up': st.session_state.get('face_image'),
        'Front View': st.session_state.get('front_view_image'),
        'Side Profile': st.session_state.get('side_profile_image'),
        'Arm (MUAC)': st.session_state.get('arm_muac_image')
    }
    if st.session_state.get('hands_image'):
        images['Hands/Nails'] = st.session_state.get('hands_image')
    if st.session_state.get('leg_image'):
        images['Legs/Feet'] = st.session_state.get('leg_image')
    if st.session_state.get('back_view_image'):
        images['Back View'] = st.session_state.get('back_view_image')

    cols = st.columns(len(images))
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
    st.title("🏥 Data Collection")
    st.markdown(f"### Collecting data for {selected_class} patient")
    
    with st.form("user_form", clear_on_submit=True):
        # Basic Information Section
        st.markdown("## 📋 Basic Information")
        st.markdown("---")
        
        name = st.text_input(
            "Patient Name",
            help="Enter the patient's full name"
        )
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
            gender = st.selectbox(
                "Gender",
                ['Male', 'Female', 'Other'],
                help="Select the patient's gender"
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
        
       # Patient Details Section
        st.markdown("## 🏷️ Patient Details")
        st.markdown("---")
        st.info("Please provide accurate information") 
        col1, col2 = st.columns(2)
        with col1:
            caregiver_name = st.text_input(
                "Primary Caregiver's Full Name",
                help="Enter the full name of the primary caregiver"
            )
            phone_number = st.text_input(
                "Primary Phone Number",
                help="This will be used as the unique patient ID"
            )
        with col2:
            secondary_contact = st.text_input(
                "Secondary Contact (Accountability Partner)",
                help="Alternative contact for follow-ups and support"
            )  

        # Patient Details Section
        st.markdown("## 🏷️ Patient Details")
        st.markdown("---")
        st.info("Please provide accurate information") 
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox(
                "Region",
                ["Ahafo", "Ashanti", "Bono", "Bono East", "Central", "Eastern", "Greater Accra", "North East", "Northern", "Oti", "Savannah", "Upper East", "Upper West", "Volta", "Western",  "Western North"],
                help="Enter the region"
            )
            town = st.text_input(
                "Town/City",
                help="Enter the town or city"
            )
        with col2:
            community = st.selectbox(
                "Community/Location",
                ["Urban Center", "Peri-Rural Community", "Rural Community", "Coastal Area Fishing Community", "Farming Community"],
                help="Enter the community or location"
            )  
            health_facility = st.selectbox(
                "Health Facility",
                ["CHPS Compound", "Regional Hospital", "District Hospital", "Tertiary (Teaching) Hospital",  "Health Center and Clinic", "Maternity Home", "Mobile Clinic", "Other"],
                help="Enter the health facility"
            )


        # Biochemical Indicators Section
        st.markdown("## 🔍 Biochemical Indicators")
        st.markdown("---")

        st.info("Enter precise measurements for accurate assessment")
       
        hemoglobin = st.number_input(
            "Hemoglobin (g/dL)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            help="Enter the hemoglobin level in grams per deciliter"
        )
        anemic = st.selectbox(
            "Anemic Status",
            ["Normal", "Anemic"],
            help="Select if the patient is anemic based on hemoglobin levels"
        )
        st.markdown("**Note:** Fields marked with * are mandatory")

        # Physical Signs Section
        st.markdown("### Physical Signs")
        st.markdown("---")
        st.info("Select all that apply")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Physical Signs")
            st.checkbox("Swelling in feet/ankles (edema)")
            st.checkbox("Pale conjunctiva (eyes)")
            st.checkbox("Angular cracks on mouth (cheilitis)")
            st.checkbox("Swollen gums, bleeding")
            st.checkbox("Dry, scaly skin")
            st.checkbox("Potbelly appearance")
            st.checkbox("Thin upper arms / visible ribs")
            st.checkbox("Goiter (neck swelling)")

        with col2:
            st.markdown("#### Vision & Development")
            st.checkbox("Bitot's spots / Night blindness")
            st.checkbox("Delayed developmental milestones")
            st.checkbox("Glossy or pale tongue")
            st.markdown("#### Behavioral Signs")
            st.checkbox("Lethargy, irritability")
            st.checkbox("Poor appetite / feeding refusal")
            st.checkbox("Brittle / Discolored hair")
            st.checkbox("Frequent diarrhea / infections")

        # Additional Observations Section
        st.markdown("### Additional Observations")
        st.markdown("---")
        custom_notes = st.text_area(
            "Custom Notes",
            help="Enter any additional observations or notes"
        )

        # Type of Malnutrition (if applicable)
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
            st.markdown("### Front View")
            st.markdown("Child facing camera, full body visible, arms at sides")
            st.markdown("**Guidelines:**")
            st.markdown("- Stand child 2-3 feet from camera")
            st.markdown("- Ensure good lighting")
            st.markdown("- Child should be relaxed")    
            front_view_image = st.file_uploader(
                "Front View Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear front-facing image",
                key="front_view_image"
            )

            st.markdown("### Face Close-up")
            st.markdown("Clear view of facial features and eyes")
            st.markdown("**Guidelines:**")
            st.markdown("- Focus on eyes and mouth")
            st.markdown("- Check for pale conjunctiva")
            st.markdown("- Look for angular cheilitis")
            face_image = st.file_uploader(
                "Face Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear Face Close-up image",
                key="face_image"
            )
            st.markdown("### Back View")
            st.markdown("Posterior view for spine and shoulder assessment")
            st.markdown("**Optional**")
            st.markdown("**Guidelines:**")
            st.markdown("- Child facing away from camera")
            st.markdown("- Check spine alignment")
            st.markdown("- Full body visible")
            back_view_image = st.file_uploader(
                "Back View Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear Back View image",
                key="back_view_image"
            )

            st.markdown("### Hands/Nails")
            st.markdown("Palm and nail examination")
            st.markdown("**Optional**")
            st.markdown("**Guidelines:**")
            st.markdown("- Both palms visible")
            st.markdown("- Check for pallor")
            st.markdown("- Nail condition assessment")
            hands_image = st.file_uploader(
                "Hands/Nails Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the Hands/Nails",
                key="hands_image"
            )

        with col2:
            st.markdown("### Side Profile")
            st.markdown("Side view showing posture and development")
            st.markdown("**Guidelines:**")
            st.markdown("- Profile view from left side")
            st.markdown("- Show body proportions")
            st.markdown("- Arms should be visible")
            side_profile_image = st.file_uploader(
                "Side Profile Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the Side Profile",
                key="side_profile_image"
            )

            st.markdown("###  Arm (MUAC)")
            st.markdown("Arm for MUAC measurement validation")
            st.markdown("**Required**")
            st.markdown("**Guidelines:**")
            st.markdown("- Left upper arm")
            st.markdown("- Show muscle mass")
            st.markdown("- Include measurement tape if available")
            arm_muac_image = st.file_uploader(
                "Arm (MUAC) Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the Arm (MUAC)",
                key="arm_muac_image"
            )
            
            st.markdown("### Legs/Feet")
            st.markdown("Lower extremities for edema check")
            st.markdown("**Optional**")
            st.markdown("**Guidelines:**")
            st.markdown("- Check for ankle swelling")
            st.markdown("- Assess muscle wasting")
            st.markdown("- Look for skin changes")
            leg_image = st.file_uploader(
                "Legs/Feet Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the legs/feet",
                key="leg_image"
            )
        
        # Preview section for all images
        if any([front_view_image, face_image, arm_muac_image, side_profile_image, hands_image, leg_image, back_view_image]):
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
                'name': name,
                "gender": gender,
                "caregiver_name": caregiver_name,
                "phone_number": phone_number,
                "secondary_contact": secondary_contact,
                "region": region,
                "town": town,
                "community": community,
                "custom_notes": custom_notes,
                "health_facility": health_facility,
                'hemoglobin': hemoglobin,
                'oedema': yes_no('Swelling in feet/ankles (edema)'),
                'angular_stomatitis': yes_no('Angular cracks on mouth (cheilitis)'),
                'cheilosis': yes_no('Glossy or pale tongue'),
                'bowlegs': yes_no('Potbelly appearance'),
                'skin_type': 'Dry, scaly skin' if st.session_state.get('Dry, scaly skin') else 'Normal',
                'hair_type': 'Brittle / Discolored hair' if st.session_state.get('Brittle / Discolored hair') else 'Normal',
                'eyes_type': 'Pale conjunctiva' if st.session_state.get('Pale conjunctiva (eyes)') else 'Normal',
                "anemic": anemic
            }

            # Validate required fields
            validation_errors = validate_required_fields(form_data)
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return
            
            # Validate image uploads
            if not all([front_view_image, face_image, arm_muac_image, hands_image, side_profile_image]):
                st.error("Please upload all required images")
                return
            
            try:
                with st.spinner("📤 Uploading images and saving data..."):
                    # Upload images to Cloudinary
                    folder_prefix = "nutri-scan/malnourish" if selected_class == "Malnourish" else "nutri-scan/nourished"
                    
                    face_image_url = upload_image_to_cloudinary(face_image, folder=f"{folder_prefix}/face")
                    front_view_image_url = upload_image_to_cloudinary(front_view_image, folder=f"{folder_prefix}/front_view")
                    arm_muac_image_url = upload_image_to_cloudinary(arm_muac_image, folder=f"{folder_prefix}/arms")
                    side_profile_image_url = upload_image_to_cloudinary(side_profile_image, folder=f"{folder_prefix}/side_profile")
                    hands_image_url = upload_image_to_cloudinary(hands_image, folder=f"{folder_prefix}/hands") if hands_image else None
                    leg_image_url = upload_image_to_cloudinary(leg_image, folder=f"{folder_prefix}/legs") if leg_image else None
                    back_view_image_url = upload_image_to_cloudinary(back_view_image, folder=f"{folder_prefix}/back") if back_view_image else None

                    if not all([face_image_url, front_view_image_url, side_profile_image_url, arm_muac_image_url]):
                        st.error("Failed to upload one or more images. Please try again.")
                        return
                    
                    # Add image URLs to form data
                    form_data.update({
                        'face_image_url': face_image_url,
                        'front_view_image_url': front_view_image_url,
                        'arm_muac_image_url': arm_muac_image_url,
                        'hands_image_url': hands_image_url,
                        'leg_image_url': leg_image_url,
                        'back_view_image_url': back_view_image_url,
                        'side_profile_image_url': side_profile_image_url
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