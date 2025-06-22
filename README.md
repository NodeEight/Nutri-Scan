# Nutri-Scan

Nutri-Scan is a data collection and visualization platform designed to assist in the analysis and monitoring of child malnutrition. It enables users to collect detailed health and demographic data, upload images, store information in a local database, and visualize trends and patterns through an interactive dashboard. The project also includes pipelines for machine learning using both PyTorch and TensorFlow.

---

## Features

- **Data Collection App**: Streamlit-based form for collecting malnutrition and nourishment data, including images (face, hair, hands, legs) uploaded to Cloudinary.
- **Database Storage**: Uses SQLite and SQLAlchemy to store structured data for both malnourished and nourished children.
- **Data Visualization Dashboard**: Interactive Streamlit dashboard for exploring, filtering, and analyzing collected data with charts, metrics, and raw data tables.
- **Image Management**: Integration with Cloudinary for secure image uploads and management.
- **Machine Learning Pipelines**: Notebooks for training models using PyTorch and TensorFlow (see respective folders).

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Nutri-Scan
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Cloudinary Setup:**
   - Add your Cloudinary credentials to `.streamlit/secrets.toml` or set them as environment variables:
     - `CLOUDINARY_CLOUD_NAME`
     - `CLOUDINARY_API_KEY`
     - `CLOUDINARY_API_SECRET`

---

## Usage

### 1. Data Collection
Run the data collection form to input new records:
```bash
streamlit run collect_data.py
```
- Fill in all required fields and upload images for each child.
- Images are uploaded to Cloudinary and URLs are stored in the database.

### 2. Data Visualization
Launch the dashboard to explore and analyze the data:
```bash
streamlit run visualize_data.py
```
- Filter by date, category, and location.
- View metrics, charts, and download data as CSV.

### 3. Machine Learning Pipelines
- Explore and run the notebooks in `pytorch pipeline/` and `tensorflow pipeline/` for model training and experimentation.

---

## Project Structure

```
Nutri-Scan/
├── cloudinary_utils.py         # Cloudinary image upload/delete utilities
├── collect_data.py             # Streamlit app for data collection
├── database_schema.py          # SQLAlchemy ORM models for database tables
├── images/                     # Sample images for the app UI
├── malnutrition.db             # SQLite database file
├── pytorch pipeline/           # PyTorch ML pipeline notebook(s)
├── tensorflow pipeline/        # TensorFlow ML pipeline notebook(s)
├── requirements.txt            # Python dependencies
├── visualize_data.py           # Streamlit dashboard for data visualization
└── README.md                   # Project documentation
```

---

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please open an issue on the repository.
