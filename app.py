import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit.components.v1 as components  # Required for Windows fix

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & SECURE AUTH
# ---------------------------------------------------------
# layout="wide" allows the map to stretch across the whole screen
st.set_page_config(layout="wide", page_title="Liberia Forest Loss Tracker")

# Securely retrieve the service account credentials
import json

def initialize_ee():
    """Initialize Earth Engine with service account or local credentials."""
    # Try service account first (for Hugging Face deployment)
    service_account_json = os.environ.get("GEE_SERVICE_ACCOUNT")
    
    if service_account_json:
        try:
            # Parse the JSON credentials from environment variable
            service_account_info = json.loads(service_account_json)
            credentials = ee.ServiceAccountCredentials(
                email=service_account_info['client_email'],
                key_data=service_account_json
            )
            ee.Initialize(credentials=credentials)
            return True
        except Exception as e:
            st.error(f"Service account auth failed: {e}")
            return False
    else:
        # Fallback for local development
        try:
            ee.Initialize()
            return True
        except Exception:
            st.error("Please set GEE_SERVICE_ACCOUNT environment variable with your service account JSON")
            st.stop()
            return False

initialize_ee()

# ---------------------------------------------------------
# 2. DATA PROCESSING
# ---------------------------------------------------------
@st.cache_data
def get_forest_data():
    """Calculates stats once and caches them."""
    liberia = ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'Liberia'))
    counties = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM0_NAME', 'Liberia'))
    
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    lossyear = hansen.select('lossyear')
    treecover = hansen.select('treecover2000')
    
    forest2000 = treecover.gte(30)
    area_img = forest2000.multiply(ee.Image.pixelArea()).divide(10000)
    
    baseline_stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=liberia.geometry(),
        scale=1000, 
        maxPixels=1e9
    )
    total_forest_2000_ha = baseline_stats.get('treecover2000').getInfo()
    
    stats_list = []
    accumulated_area = 0
    
    for year in range(2001, 2025):
        loss_val = year - 2000
        yearly_loss = lossyear.eq(loss_val).And(forest2000)
        stat = yearly_loss.multiply(ee.Image.pixelArea()).divide(10000).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=liberia.geometry(),
            scale=1000, 
            maxPixels=1e9
        )
        val = stat.get('lossyear').getInfo()
        accumulated_area += (val if val else 0)
        pct = (accumulated_area / total_forest_2000_ha) * 100
        stats_list.append({'Year': year, 'Loss_Ha': accumulated_area, 'Percent': pct})
        
    df = pd.DataFrame(stats_list)
    return df, liberia, counties, total_forest_2000_ha

@st.cache_data
def get_county_forest_loss():
    """Calculates cumulative forest loss per county for all years."""
    counties = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM0_NAME', 'Liberia'))
    
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    lossyear = hansen.select('lossyear')
    treecover = hansen.select('treecover2000')
    forest2000 = treecover.gte(30)
    
    # Get county names
    county_list = counties.aggregate_array('ADM1_NAME').getInfo()
    
    # Dictionary to store yearly data per county
    county_yearly_data = {county: [] for county in county_list}
    
    for county_name in county_list:
        county_geom = counties.filter(ee.Filter.eq('ADM1_NAME', county_name)).geometry()
        
        # Get baseline forest area for this county
        area_img = forest2000.multiply(ee.Image.pixelArea()).divide(10000)
        baseline = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=county_geom,
            scale=1000,
            maxPixels=1e9
        )
        county_forest_2000 = baseline.get('treecover2000').getInfo() or 0
        
        accumulated_loss = 0
        for year in range(2001, 2025):
            loss_val = year - 2000
            yearly_loss = lossyear.eq(loss_val).And(forest2000)
            stat = yearly_loss.multiply(ee.Image.pixelArea()).divide(10000).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=county_geom,
                scale=1000,
                maxPixels=1e9
            )
            val = stat.get('lossyear').getInfo() or 0
            accumulated_loss += val
            pct = (accumulated_loss / county_forest_2000 * 100) if county_forest_2000 > 0 else 0
            county_yearly_data[county_name].append({
                'Year': year,
                'County': county_name,
                'Loss_Ha': accumulated_loss,
                'Baseline_Ha': county_forest_2000,
                'Percent': pct
            })
    
    # Flatten to DataFrame
    all_data = []
    for county, data in county_yearly_data.items():
        all_data.extend(data)
    
    return pd.DataFrame(all_data)

# Load Data
with st.spinner("Loading Earth Engine Data..."):
    df, liberia_boundary, liberia_counties, total_forest_2000 = get_forest_data()

with st.spinner("Loading County-Level Data..."):
    county_df = get_county_forest_loss()

# ---------------------------------------------------------
# 3. SIDEBAR (Controls & Chart)
# ---------------------------------------------------------
st.sidebar.title("🌲 Controls")

# A. Slider
selected_year = st.sidebar.slider("Year", 2001, 2024, 2024)

# B. Metrics
current_data = df[df['Year'] == selected_year].iloc[0]
loss_ha = current_data['Loss_Ha']
loss_pct = current_data['Percent']

st.sidebar.metric(
    label="Cumulative Loss", 
    value=f"{loss_ha:,.0f} Ha", 
    delta=f"-{loss_pct:.1f}%",
    delta_color="inverse"
)

# C. Charts in Expanders (Collapsible)
st.sidebar.markdown("---")

# National Loss Trend Chart
with st.sidebar.expander("📈 National Loss Trend", expanded=True):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(df['Year'], df['Loss_Ha'], color='#d3d3d3', linewidth=1.5)
    active_data = df[df['Year'] <= selected_year]
    ax.plot(active_data['Year'], active_data['Loss_Ha'], color='#ff4b4b', linewidth=2.5)
    ax.scatter([selected_year], [loss_ha], color='red', s=50, zorder=5)
    
    ax.set_ylabel("Hectares", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    plt.close(fig)

# County Loss Chart
with st.sidebar.expander("📊 Loss by County", expanded=True):
    # Filter county data for selected year
    county_year_data = county_df[county_df['Year'] == selected_year].copy()
    county_year_data = county_year_data.sort_values('Loss_Ha', ascending=True)
    
    # Create horizontal bar chart
    fig2, ax2 = plt.subplots(figsize=(4, 5))
    colors = ['#ff4b4b' if loss > county_year_data['Loss_Ha'].median() else '#ff8080' 
              for loss in county_year_data['Loss_Ha']]
    bars = ax2.barh(county_year_data['County'], county_year_data['Loss_Ha'], color=colors)
    
    ax2.set_xlabel("Hectares", fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, val in zip(bars, county_year_data['Loss_Ha']):
        ax2.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, 
                 f'{val:,.0f}', va='center', fontsize=6)
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ---------------------------------------------------------
# 4. MAIN MAP (Full Width)
# ---------------------------------------------------------
st.subheader(f"Liberia Forest Loss: 2000–{selected_year}")

# Create Map (Safe Initialization)
m = geemap.Map(center=[6.5, -9.5], zoom=7)

# Add Basemap manually (Safest method)
try:
    m.add_basemap("HYBRID") # Tries Google Hybrid first
except:
    m.add_basemap("SATELLITE") # Fallback
# Define Layers
hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
lossyear = hansen.select('lossyear')
treecover = hansen.select('treecover2000')
forest2000 = treecover.gte(30).clip(liberia_boundary)

# 1. Forest 2000 (Green)
m.addLayer(forest2000.updateMask(forest2000), {'palette': ['006400'], 'opacity': 0.6}, 'Forest 2000 (Base)')

# 2. Cumulative Loss (Red)
loss_val = selected_year - 2000
cumulative_loss = lossyear.lte(loss_val).And(lossyear.neq(0)).And(forest2000)
m.addLayer(cumulative_loss.selfMask(), {'palette': ['red']}, 'Cumulative Loss')

# 3. Counties (Overlay)
style = {'color': 'white', 'weight': 1, 'fillOpacity': 0}
m.add_geojson(geemap.ee_to_geojson(liberia_counties), style=style, layer_name="Counties")

# 4. Add Layer Control (To switch basemaps)
# This adds the button to toggle between Esri and OSM
m.add_basemap('OpenStreetMap') # Add OSM as an option
m.add_layer_control()

# ---------------------------------------------------------
# 5. RENDER MAP (Windows Fix Applied)
# ---------------------------------------------------------
# Save to static HTML
m.to_html("map_render.html")

# Read and display manually
with open("map_render.html", "r", encoding='utf-8') as f:
    map_html = f.read()

# Render full width (height=850 for better visibility)
components.html(map_html, height=850, scrolling=True)