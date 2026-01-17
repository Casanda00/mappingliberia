import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit.components.v1 as components
import json

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & SECURE AUTH
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Liberia Forest Loss Tracker")

# Remove default padding to stretch map edge-to-edge
st.markdown("""
<style>
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    iframe {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

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
def generate_map_html(year: int, _liberia_boundary, _liberia_counties, county_stats: dict) -> str:
    """Generate and cache the map HTML for each year with interactive county selection."""
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
    
    # Convert counties to GeoJSON and enrich with stats
    counties_geojson = geemap.ee_to_geojson(_liberia_counties)
    
    # Add forest loss stats to each county feature
    for feature in counties_geojson.get('features', []):
        county_name = feature.get('properties', {}).get('ADM1_NAME', 'Unknown')
        if county_name in county_stats:
            stats = county_stats[county_name]
            feature['properties']['loss_ha'] = stats.get('Loss_Ha', 0)
            feature['properties']['baseline_ha'] = stats.get('Baseline_Ha', 0)
            feature['properties']['loss_pct'] = stats.get('Percent', 0)
    
    # Counties (Overlay) - nearly transparent fill to allow click detection, yellow border on hover
    style = {'color': 'white', 'weight': 1, 'fillColor': '#000000', 'fillOpacity': 0.01}
    hover_style = {'color': 'yellow', 'weight': 2, 'fillColor': '#000000', 'fillOpacity': 0.01}
    m.add_geojson(counties_geojson, style=style, hover_style=hover_style, layer_name="Counties", info_mode=None)
    
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
    
    # Inject view persistence + county selection JavaScript
    interactive_js = """
<script>
(function() {
    var KEY_CENTER = 'liberia_map_center';
    var KEY_ZOOM = 'liberia_map_zoom';
    var selectedLayer = null;
    var originalStyle = {color: 'white', weight: 1, fillOpacity: 0};
    var selectedStyle = {color: '#00ffff', weight: 3, fillOpacity: 0};
    
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
    
    function formatNumber(num) {
        return num.toLocaleString('en-US', {maximumFractionDigits: 0});
    }
    
    function initMapFeatures() {
        var map = findLeafletMap();
        if (!map) {
            setTimeout(initMapFeatures, 50);
            return;
        }
        
        // Restore saved view
        try {
            var savedCenter = localStorage.getItem(KEY_CENTER);
            var savedZoom = localStorage.getItem(KEY_ZOOM);
            if (savedCenter && savedZoom) {
                var center = JSON.parse(savedCenter);
                map.setView(center, parseInt(savedZoom), {animate: false, duration: 0});
            }
        } catch(e) {}
        
        // Save view on movement
        function saveView() {
            try {
                var center = map.getCenter();
                localStorage.setItem(KEY_CENTER, JSON.stringify([center.lat, center.lng]));
                localStorage.setItem(KEY_ZOOM, map.getZoom().toString());
            } catch(e) {}
        }
        map.on('moveend', saveView);
        map.on('zoomend', saveView);
        
        // Find GeoJSON layers and add click handlers
        map.eachLayer(function(layer) {
            if (layer.feature && layer.feature.properties && layer.feature.properties.ADM1_NAME) {
                layer.on('click', function(e) {
                    // Reset previous selection
                    if (selectedLayer && selectedLayer !== layer) {
                        selectedLayer.setStyle(originalStyle);
                    }
                    
                    // Highlight clicked county
                    layer.setStyle(selectedStyle);
                    selectedLayer = layer;
                    
                    // Get county stats
                    var props = layer.feature.properties;
                    var name = props.ADM1_NAME || 'Unknown';
                    var lossHa = props.loss_ha || 0;
                    var baselineHa = props.baseline_ha || 0;
                    var lossPct = props.loss_pct || 0;
                    
                    // Create popup content with period
                    var selectedYear = SELECTED_YEAR_PLACEHOLDER;
                    var periodYears = selectedYear - 2000;
                    var content = '<div style="font-family: Arial, sans-serif; min-width: 200px;">' +
                        '<h4 style="margin: 0 0 8px 0; color: #333; border-bottom: 2px solid #ff4b4b; padding-bottom: 5px;">' + 
                        name + ' County</h4>' +
                        '<p style="margin: 0 0 8px 0; font-size: 11px; color: #888;">Period: 2001 – ' + selectedYear + ' (' + periodYears + ' years)</p>' +
                        '<table style="width: 100%; font-size: 12px;">' +
                        '<tr><td style="color: #666;">Baseline (2000):</td><td style="text-align: right; font-weight: bold;">' + 
                        formatNumber(baselineHa) + ' Ha</td></tr>' +
                        '<tr><td style="color: #666;">Cumulative Loss:</td><td style="text-align: right; font-weight: bold; color: #ff4b4b;">' + 
                        formatNumber(lossHa) + ' Ha</td></tr>' +
                        '<tr><td style="color: #666;">Loss Percentage:</td><td style="text-align: right; font-weight: bold; color: #ff4b4b;">' + 
                        lossPct.toFixed(1) + '%</td></tr>' +
                        '</table></div>';
                    
                    // Show popup without auto-panning
                    layer.bindPopup(content, {maxWidth: 250, autoPan: false}).openPopup();
                    
                    L.DomEvent.stopPropagation(e);
                });
            }
        });
        
        // Click on map (not county) to deselect
        map.on('click', function(e) {
            if (selectedLayer) {
                selectedLayer.setStyle(originalStyle);
                selectedLayer.closePopup();
                selectedLayer = null;
            }
        });
    }
    
    if (document.readyState === 'complete') {
        initMapFeatures();
    } else {
        document.addEventListener('DOMContentLoaded', initMapFeatures);
    }
})();
</script>
"""
    
    # Replace year placeholder and inject script
    interactive_js = interactive_js.replace('SELECTED_YEAR_PLACEHOLDER', str(year))
    html_content = html_content.replace('</body>', interactive_js + '</body>')
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

# Year slider
selected_year = st.sidebar.slider(
    "Year", 
    2001, 
    2024, 
    st.session_state.selected_year,
    key="year_slider"
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

# Build county stats dictionary for selected year
county_year_stats = county_df[county_df['Year'] == selected_year].set_index('County').to_dict('index')

# Get cached map HTML for this year
map_html = generate_map_html(selected_year, liberia_boundary, liberia_counties, county_year_stats)

# Render the map
components.html(map_html, height=850, scrolling=True)