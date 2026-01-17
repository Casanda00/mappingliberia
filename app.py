import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit.components.v1 as components
import json
import hashlib

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & SECURE AUTH
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Liberia Forest Loss Tracker")

def initialize_ee():
    """Initialize Earth Engine with service account or local credentials."""
    # Check if already initialized by trying a simple operation
    try:
        ee.Number(1).getInfo()
        return True  # Already initialized
    except:
        pass  # Not initialized yet, continue
        
    service_account_json = os.environ.get("GEE_SERVICE_ACCOUNT")
    
    if service_account_json:
        try:
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
        try:
            ee.Initialize()
            return True
        except Exception:
            st.error("Please set GEE_SERVICE_ACCOUNT environment variable with your service account JSON")
            st.stop()
            return False

initialize_ee()

# ---------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = 2024
if 'map_needs_update' not in st.session_state:
    st.session_state.map_needs_update = False

# ---------------------------------------------------------
# 3. DATA PROCESSING (Cached)
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
    
    county_list = counties.aggregate_array('ADM1_NAME').getInfo()
    county_yearly_data = {county: [] for county in county_list}
    
    for county_name in county_list:
        county_geom = counties.filter(ee.Filter.eq('ADM1_NAME', county_name)).geometry()
        
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
    
    all_data = []
    for county, data in county_yearly_data.items():
        all_data.extend(data)
    
    return pd.DataFrame(all_data)

# ---------------------------------------------------------
# 4. CACHE MAP HTML BY YEAR (Key optimization)
# ---------------------------------------------------------
@st.cache_data
def generate_map_html(year: int, _liberia_boundary, _liberia_counties) -> str:
    """Generate and cache the map HTML for each year."""
    m = geemap.Map(center=[6.5, -9.5], zoom=7)
    
    try:
        m.add_basemap("HYBRID")
    except:
        m.add_basemap("SATELLITE")
    
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    lossyear = hansen.select('lossyear')
    treecover = hansen.select('treecover2000')
    forest2000 = treecover.gte(30).clip(_liberia_boundary)
    
    # Forest 2000 (Green)
    m.addLayer(forest2000.updateMask(forest2000), {'palette': ['006400'], 'opacity': 0.6}, 'Forest 2000 (Base)')
    
    # Cumulative Loss (Red)
    loss_val = year - 2000
    cumulative_loss = lossyear.lte(loss_val).And(lossyear.neq(0)).And(forest2000)
    m.addLayer(cumulative_loss.selfMask(), {'palette': ['red']}, 'Cumulative Loss')
    
    # Counties (Overlay)
    style = {'color': 'white', 'weight': 1, 'fillOpacity': 0}
    m.add_geojson(geemap.ee_to_geojson(_liberia_counties), style=style, layer_name="Counties")
    
    m.add_basemap('OpenStreetMap')
    m.add_layer_control()
    
    # Generate HTML string directly
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        temp_path = f.name
    
    m.to_html(temp_path)
    
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    os.unlink(temp_path)
    
    # Inject view persistence JavaScript
    view_persistence_js = """
<script>
(function() {
    var KEY_CENTER = 'liberia_map_center';
    var KEY_ZOOM = 'liberia_map_zoom';
    
    function findLeafletMap() {
        for (var key in window) {
            try {
                if (window[key] && typeof window[key].getCenter === 'function' && 
                    typeof window[key].getZoom === 'function' && 
                    window[key]._container) {
                    return window[key];
                }
            } catch(e) {}
        }
        return null;
    }
    
    function initMapPersistence() {
        var map = findLeafletMap();
        if (!map) {
            setTimeout(initMapPersistence, 50);
            return;
        }
        
        // Restore saved view immediately
        try {
            var savedCenter = localStorage.getItem(KEY_CENTER);
            var savedZoom = localStorage.getItem(KEY_ZOOM);
            
            if (savedCenter && savedZoom) {
                var center = JSON.parse(savedCenter);
                var zoom = parseInt(savedZoom);
                map.setView(center, zoom, {animate: false, duration: 0});
            }
        } catch(e) {
            console.log('Could not restore map view:', e);
        }
        
        // Save view on any movement
        function saveView() {
            try {
                var center = map.getCenter();
                localStorage.setItem(KEY_CENTER, JSON.stringify([center.lat, center.lng]));
                localStorage.setItem(KEY_ZOOM, map.getZoom().toString());
            } catch(e) {}
        }
        
        map.on('moveend', saveView);
        map.on('zoomend', saveView);
    }
    
    // Start immediately, don't wait for DOMContentLoaded
    if (document.readyState === 'complete') {
        initMapPersistence();
    } else {
        document.addEventListener('DOMContentLoaded', initMapPersistence);
    }
})();
</script>
"""
    
    html_content = html_content.replace('</body>', view_persistence_js + '</body>')
    return html_content

# Load Data (cached)
with st.spinner("Loading Earth Engine Data..."):
    df, liberia_boundary, liberia_counties, total_forest_2000 = get_forest_data()

with st.spinner("Loading County-Level Data..."):
    county_df = get_county_forest_loss()

# ---------------------------------------------------------
# 5. SIDEBAR (Controls & Charts)
# ---------------------------------------------------------
st.sidebar.title("🌲 Controls")

# Year slider with on_change callback to minimize reruns
def on_year_change():
    st.session_state.map_needs_update = True

selected_year = st.sidebar.slider(
    "Year", 
    2001, 
    2024, 
    st.session_state.selected_year,
    key="year_slider",
    on_change=on_year_change
)

# Update session state
st.session_state.selected_year = selected_year

# Metrics
current_data = df[df['Year'] == selected_year].iloc[0]
loss_ha = current_data['Loss_Ha']
loss_pct = current_data['Percent']

st.sidebar.metric(
    label="Cumulative Loss", 
    value=f"{loss_ha:,.0f} Ha", 
    delta=f"-{loss_pct:.1f}%",
    delta_color="inverse"
)

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
    county_year_data = county_df[county_df['Year'] == selected_year].copy()
    county_year_data = county_year_data.sort_values('Loss_Ha', ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(4, 5))
    colors = ['#ff4b4b' if loss > county_year_data['Loss_Ha'].median() else '#ff8080' 
              for loss in county_year_data['Loss_Ha']]
    bars = ax2.barh(county_year_data['County'], county_year_data['Loss_Ha'], color=colors)
    
    ax2.set_xlabel("Hectares", fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, county_year_data['Loss_Ha']):
        ax2.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, 
                 f'{val:,.0f}', va='center', fontsize=6)
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ---------------------------------------------------------
# 6. MAIN MAP (Cached HTML per year + View Persistence)
# ---------------------------------------------------------
st.subheader(f"Liberia Forest Loss: 2000–{selected_year}")

# Get cached map HTML for this year
map_html = generate_map_html(selected_year, liberia_boundary, liberia_counties)

# Render the map
components.html(map_html, height=850, scrolling=True)