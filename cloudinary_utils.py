import cloudinary
import cloudinary.uploader
import streamlit as st
import os
from io import BytesIO

# Cloudinary configuration with fallback to environment variables
def get_cloudinary_config():
    """Get Cloudinary configuration from Streamlit secrets or environment variables"""
    cloud_name = (
        st.secrets.get("CLOUDINARY_CLOUD_NAME") or 
        os.getenv("CLOUDINARY_CLOUD_NAME") or 
        "your_cloud_name"
    )
    api_key = (
        st.secrets.get("CLOUDINARY_API_KEY") or 
        os.getenv("CLOUDINARY_API_KEY") or 
        "your_api_key"
    )
    api_secret = (
        st.secrets.get("CLOUDINARY_API_SECRET") or 
        os.getenv("CLOUDINARY_API_SECRET") or 
        "your_api_secret"
    )
    
    return cloud_name, api_key, api_secret

# Configure Cloudinary
cloud_name, api_key, api_secret = get_cloudinary_config()
cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret
)

def upload_image_to_cloudinary(image_file, folder="nutri-scan"):
    """
    Upload an image to Cloudinary and return the URL
    
    Args:
        image_file: StreamlitUploadedFile object
        folder: Cloudinary folder name
    
    Returns:
        str: Cloudinary URL of the uploaded image
    """
    try:
        if image_file is None:
            return None
            
        # Check if Cloudinary is properly configured
        if cloud_name == "your_cloud_name" or api_key == "your_api_key" or api_secret == "your_api_secret":
            st.error("Cloudinary is not properly configured. Please check your credentials in .streamlit/secrets.toml or environment variables.")
            return None
            
        # Read the image file
        image_bytes = image_file.read()
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            BytesIO(image_bytes),
            folder=folder,
            resource_type="image",
            public_id=f"{folder}/{image_file.name}_{image_file.type}"
        )
        
        # Return the secure URL
        return upload_result.get('secure_url')
        
    except Exception as e:
        st.error(f"Error uploading image to Cloudinary: {str(e)}")
        return None

def delete_image_from_cloudinary(public_id):
    """
    Delete an image from Cloudinary
    
    Args:
        public_id: Cloudinary public ID of the image
    """
    try:
        cloudinary.uploader.destroy(public_id)
    except Exception as e:
        st.error(f"Error deleting image from Cloudinary: {str(e)}")

def test_cloudinary_connection():
    """
    Test if Cloudinary is properly configured
    """
    if cloud_name == "your_cloud_name" or api_key == "your_api_key" or api_secret == "your_api_secret":
        return False, "Cloudinary credentials not configured"
    
    return True, "Cloudinary credentials configured" 