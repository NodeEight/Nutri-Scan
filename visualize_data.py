import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Nutri-Scan Data Visualization",
    page_icon="📊",
    layout="wide"
)

# Load environment variables
load_dotenv()

DATABASE_URL = (
        st.secrets.get("DATABASE_URL") or 
        os.getenv("DATABASE_URL") or 
        "your_cloud_name"
    )

engine = create_engine(DATABASE_URL)

def load_data():
    """Load data from both tables using SQLAlchemy engine"""
    # Load malnourished data
    malnourished_df = pd.read_sql_query(
        """
        SELECT *, 'Malnourished' as category 
        FROM malnurish
        """, engine
    )
    # Load nourished data
    nourished_df = pd.read_sql_query(
        """
        SELECT *, 'Nourished' as category 
        FROM nurish
        """, engine
    )
    # Combine datasets
    if not malnourished_df.empty and not nourished_df.empty:
        combined_df = pd.concat([malnourished_df, nourished_df], ignore_index=True)
    elif not malnourished_df.empty:
        combined_df = malnourished_df
    elif not nourished_df.empty:
        combined_df = nourished_df
    else:
        combined_df = pd.DataFrame()
    return combined_df, malnourished_df, nourished_df

def main():
    st.title("📊 Nutri-Scan Data Visualization Dashboard")
    st.markdown("---")
    
    # Load data
    combined_df, malnourished_df, nourished_df = load_data()
    
    if combined_df.empty:
        st.warning("⚠️ No data found in the database. Please collect some data first using the data collection form.")
        st.info("Run `streamlit run collect_data.py` to start collecting data.")
        return
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    if 'date_created' in combined_df.columns:
        combined_df['date_created'] = pd.to_datetime(combined_df['date_created'])
        min_date = combined_df['date_created'].min()
        max_date = combined_df['date_created'].max()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            combined_df = combined_df[
                (combined_df['date_created'].dt.date >= start_date) &
                (combined_df['date_created'].dt.date <= end_date)
            ]
    
    # Category filter
    categories = combined_df['category'].unique()
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=categories,
        default=categories
    )
    
    if selected_categories:
        combined_df = combined_df[combined_df['category'].isin(selected_categories)]
    
    # Location filter
    if 'location' in combined_df.columns and not combined_df['location'].isna().all():
        locations = ['All'] + list(combined_df['location'].dropna().unique())
        selected_location = st.sidebar.selectbox("Location", locations)
        
        if selected_location != 'All':
            combined_df = combined_df[combined_df['location'] == selected_location]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(combined_df))
    
    with col2:
        malnourished_count = len(combined_df[combined_df['category'] == 'Malnourished'])
        st.metric("Malnourished", malnourished_count)
    
    with col3:
        nourished_count = len(combined_df[combined_df['category'] == 'Nourished'])
        st.metric("Nourished", nourished_count)
    
    with col4:
        if len(combined_df) > 0:
            avg_age = combined_df['age'].mean()
            st.metric("Average Age (months)", f"{avg_age:.1f}")
    
    st.markdown("---")
    
    # Charts section
    st.header("📈 Data Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "👥 Demographics", "🏥 Medical Indicators", "📍 Geographic", "📅 Temporal"
    ])
    
    with tab1:
        st.subheader("Data Distribution Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution pie chart
            if len(combined_df) > 0:
                category_counts = combined_df['category'].value_counts()
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Distribution by Category",
                    color_discrete_map={'Malnourished': '#ff6b6b', 'Nourished': '#4ecdc4'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Age distribution histogram
            if len(combined_df) > 0:
                fig_age = px.histogram(
                    combined_df,
                    x='age',
                    color='category',
                    title="Age Distribution",
                    nbins=20,
                    color_discrete_map={'Malnourished': '#ff6b6b', 'Nourished': '#4ecdc4'}
                )
                fig_age.update_layout(xaxis_title="Age (months)", yaxis_title="Count")
                st.plotly_chart(fig_age, use_container_width=True)
    
    with tab2:
        st.subheader("Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weight vs Height scatter plot
            if len(combined_df) > 0:
                fig_scatter = px.scatter(
                    combined_df,
                    x='weight',
                    y='height',
                    color='category',
                    title="Weight vs Height",
                    color_discrete_map={'Malnourished': '#ff6b6b', 'Nourished': '#4ecdc4'}
                )
                fig_scatter.update_layout(xaxis_title="Weight (kg)", yaxis_title="Height (cm)")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Hand circumference distribution
            if len(combined_df) > 0:
                fig_hand = px.box(
                    combined_df,
                    x='category',
                    y='mid_lower_hand_circumference',
                    title="Hand Circumference by Category",
                    color='category',
                    color_discrete_map={'Malnourished': '#ff6b6b', 'Nourished': '#4ecdc4'}
                )
                fig_hand.update_layout(xaxis_title="Category", yaxis_title="Hand Circumference (cm)")
                st.plotly_chart(fig_hand, use_container_width=True)
    
    with tab3:
        st.subheader("Medical Indicators Analysis")
        
        # Medical indicators
        medical_indicators = ['skin_type', 'hair_type', 'eyes_type', 'oedema', 'angular_stomatitis', 'cheilosis', 'bowlegs']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Medical indicators comparison
            if len(combined_df) > 0:
                # Create a summary of medical indicators
                medical_data = []
                for indicator in medical_indicators:
                    if indicator in combined_df.columns:
                        indicator_counts = combined_df.groupby(['category', indicator]).size().reset_index(name='count')
                        medical_data.append(indicator_counts)
                
                if medical_data:
                    medical_summary = pd.concat(medical_data, ignore_index=True)
                    
                    # Create a heatmap-like visualization
                    fig_medical = px.bar(
                        medical_summary,
                        x='category',
                        y='count',
                        color=medical_summary.columns[1],  # The indicator column
                        title="Medical Indicators by Category",
                        barmode='group'
                    )
                    st.plotly_chart(fig_medical, use_container_width=True)
        
        with col2:
            # Specific medical condition analysis
            if len(combined_df) > 0 and 'oedema' in combined_df.columns:
                oedema_data = combined_df.groupby(['category', 'oedema']).size().reset_index(name='count')
                fig_oedema = px.bar(
                    oedema_data,
                    x='category',
                    y='count',
                    color='oedema',
                    title="Oedema by Category",
                    color_discrete_map={'yes': '#ff6b6b', 'no': '#4ecdc4'}
                )
                st.plotly_chart(fig_oedema, use_container_width=True)
    
    with tab4:
        st.subheader("Geographic Analysis")
        
        if 'location' in combined_df.columns and not combined_df['location'].isna().all():
            col1, col2 = st.columns(2)
            
            with col1:
                # Location distribution
                location_counts = combined_df['location'].value_counts().head(10)
                fig_location = px.bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    title="Top 10 Locations",
                    labels={'x': 'Location', 'y': 'Count'}
                )
                fig_location.update_xaxes(tickangle=45)
                st.plotly_chart(fig_location, use_container_width=True)
            
            with col2:
                # Location by category
                location_category = combined_df.groupby(['location', 'category']).size().reset_index(name='count')
                fig_location_cat = px.bar(
                    location_category,
                    x='location',
                    y='count',
                    color='category',
                    title="Data by Location and Category",
                    color_discrete_map={'Malnourished': '#ff6b6b', 'Nourished': '#4ecdc4'}
                )
                fig_location_cat.update_xaxes(tickangle=45)
                st.plotly_chart(fig_location_cat, use_container_width=True)
        else:
            st.info("No location data available for geographic analysis.")
    
    with tab5:
        st.subheader("Temporal Analysis")
        
        if 'date_created' in combined_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily data collection trend
                daily_counts = combined_df.groupby(combined_df['date_created'].dt.date).size().reset_index(name='count')
                daily_counts['date'] = pd.to_datetime(daily_counts['date_created'])
                
                fig_trend = px.line(
                    daily_counts,
                    x='date',
                    y='count',
                    title="Daily Data Collection Trend",
                    labels={'date': 'Date', 'count': 'Records Collected'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Monthly distribution
                combined_df['month'] = combined_df['date_created'].dt.month
                monthly_counts = combined_df.groupby(['month', 'category']).size().reset_index(name='count')
                
                fig_monthly = px.bar(
                    monthly_counts,
                    x='month',
                    y='count',
                    color='category',
                    title="Monthly Distribution",
                    color_discrete_map={'Malnourished': '#ff6b6b', 'Nourished': '#4ecdc4'}
                )
                fig_monthly.update_xaxes(tickmode='array', tickvals=list(range(1, 13)), ticktext=[
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                ])
                st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            st.info("No date information available for temporal analysis.")
    
    # Data table section
    st.markdown("---")
    st.header("📋 Raw Data")
    
    # Show data table with filters
    st.subheader("Data Records")
    
    # Add search functionality
    search_term = st.text_input("Search in data (age, location, etc.)")
    
    if search_term:
        # Create a search mask
        search_mask = combined_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered_df = combined_df[search_mask]
    else:
        filtered_df = combined_df
    
    # Show the data table
    st.dataframe(
        filtered_df.drop(['face_image_url', 'hair_image_url', 'hands_image_url', 'leg_image_url'], axis=1, errors='ignore'),
        use_container_width=True
    )
    
    # Download functionality
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"nutri_scan_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main() 