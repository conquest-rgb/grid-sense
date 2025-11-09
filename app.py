import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import qrcode
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# PAGE CONFIGURATION
st.set_page_config(
    page_title="GridSense - Outage Risk Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styles
dark_mode_css = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    [data-testid="stMetricValue"] {
        color: #fafafa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
</style>
"""

light_mode_css = """
<style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
</style>
"""


# LOAD MODELS AND DATA
@st.cache_resource
def load_models_and_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    model = joblib.load('models/rf_tuned_final_v20251107_0700.pkl')
    preprocessor = joblib.load('models/preprocessor_v20251107_0700.pkl')
    
    with open('streamlit_app/data/counties.json', 'r') as f:
        counties = json.load(f)
    
    return model, preprocessor, config, counties


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('Data/final_df.csv')
    df['hour'] = pd.to_datetime(df['hour'], errors='coerce')
    return df


# HELPER FUNCTIONS
def to_risk_level(prob):
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def generate_qr_code(url):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def get_shareable_url(county, persona):
    """Generate shareable URL with query parameters"""
    base_url = st.get_option("browser.serverAddress")
    port = st.get_option("browser.serverPort")
    
    persona_encoded = persona.replace(' ', '_').replace('/', '%2F')
    
    # Check if deployed or local
    if not base_url or base_url == "localhost" or base_url == "0.0.0.0":
        return f"http://localhost:8501/?county={county}&persona={persona_encoded}"
    else:
        # Deployed app URL
        if port and port != 80 and port != 443:
            return f"https://{base_url}:{port}/?county={county}&persona={persona_encoded}"
        else:
            return f"https://{base_url}/?county={county}&persona={persona_encoded}"


def get_county_forecast(county, final_df, model, hours_ahead=48):
    """Generate forecast for a specific county"""
    county_data = final_df[final_df['county'] == county].copy()
    
    if len(county_data) == 0:
        st.warning(f"No data available for {county}")
        return pd.DataFrame()
    
    county_data = county_data.sort_values('hour')
    template = county_data.iloc[-1].copy()
    
    feature_cols = [col for col in county_data.columns 
                   if col not in ['planned_outage', 'y_outage_1h', 'y_outage_6h', 
                                 'y_outage_24h', 'y_outage_next_48h', 'hour', 'date']]
    
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    future_times = [start_time + timedelta(hours=i) for i in range(hours_ahead)]
    
    predictions = []
    for future_time in future_times:
        record = template[feature_cols].copy()
        
        if 'county' in feature_cols:
            record['county'] = county
        if 'hour_of_day' in record.index:
            record['hour_of_day'] = future_time.hour
        if 'day_of_week' in record.index:
            record['day_of_week'] = future_time.weekday()
        if 'is_weekend' in record.index:
            record['is_weekend'] = 1 if future_time.weekday() >= 5 else 0
        if 'month' in record.index:
            record['month'] = future_time.month
        
        predictions.append(record)
    
    X = pd.DataFrame(predictions)
    
    try:
        probabilities = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        probabilities = np.zeros(len(X))
    
    forecast = pd.DataFrame({
        'hour': future_times,
        'county': county,
        'risk_probability': probabilities,
        'risk_level': [to_risk_level(p) for p in probabilities]
    })
    
    return forecast


def recommend_actions(forecast_df, persona):
    """Generate persona-specific recommendations"""
    if len(forecast_df) == 0:
        return ["No forecast data available"]
    
    max_prob = forecast_df['risk_probability'].max()
    max_risk_level = to_risk_level(max_prob)
    avg_prob = forecast_df['risk_probability'].mean()
    
    high_risk_hours = forecast_df[forecast_df['risk_level'] == 'HIGH']
    peak_hours_str = ', '.join(high_risk_hours['hour'].dt.strftime('%I %p').tolist()[:3]) if len(high_risk_hours) > 0 else "None identified"
    
    recommendations = [] 
    
    if max_risk_level == "HIGH":
        recommendations.append(f"🚨 **CRITICAL ALERT:** HIGH risk window detected for {peak_hours_str}")
    elif max_risk_level == "MEDIUM":
        recommendations.append(f"⚠️ **CAUTION:** MEDIUM risk expected for {peak_hours_str}")
    else:
        recommendations.append("✅ **ALL CLEAR:** Low grid stress expected in next 48 hours")
    
    # Persona-specific recommendations
    if persona == "SME / Factory":
        if max_risk_level == "HIGH":
            recommendations.extend([
                "🏭 **Production:** Complete all high-energy operations before peak risk window",
                "🔋 **Equipment:** Fully charge all battery-operated devices and forklifts",
                "📋 **Planning:** Reschedule critical production runs outside high-risk period",
                "⚡ **Backup:** Test generator and ensure adequate fuel supply"
            ])
    
    elif persona == "Clinic / Cold-Chain":
        if max_risk_level == "HIGH":
            recommendations.extend([
                "🏥 **IMMEDIATE ACTION:** Test backup generator NOW and verify fuel levels",
                "💉 **Cold Chain:** Consolidate vaccines into primary battery-backed units",
                "🚑 **Operations:** Postpone non-urgent elective procedures during risk window",
                "🌡️ **Monitoring:** Activate temperature logging on all refrigeration units"
            ])
    
    elif persona == "Telecom Site":
        if max_risk_level == "HIGH":
            recommendations.extend([
                "📡 **CRITICAL:** Verify battery bank charge status across all sites",
                "⛽ **Fueling:** Priority refuel for sites below 24h generator runtime",
                "🎯 **Monitoring:** Alert NOC to watch county alarms during risk window",
                "🔧 **Maintenance:** Position field crews in county for rapid response"
            ])
    
    elif persona == "Household":
        if max_risk_level == "HIGH":
            recommendations.extend([
                "🔌 **Charging:** Charge all phones, laptops, and power banks immediately",
                "💡 **Lighting:** Prepare flashlights and check batteries",
                "🍳 **Meals:** Cook and store meals before risk window",
                "🌡️ **Temperature:** Pre-cool refrigerator, minimize door openings"
            ])
    
    recommendations.append(f"\n📊 **Average Risk Level:** {avg_prob:.1%} over next 48 hours")
    return recommendations


# ==================== MAIN APP ====================

# Load resources
model, preprocessor, config, counties = load_models_and_config()
final_df = load_data()

# Initialize session state from URL parameters
query_params = st.query_params

if 'selected_county' not in st.session_state:
    if 'county' in query_params:
        url_county = query_params['county'].lower()
        st.session_state['selected_county'] = url_county if url_county in counties else None
    else:
        st.session_state['selected_county'] = None  # ← CHANGED TO None

if 'selected_persona' not in st.session_state:
    persona_options = ["SME / Factory", "Clinic / Cold-Chain", "Telecom Site", "Household"]
    if 'persona' in query_params:
        url_persona = query_params['persona'].replace('_', ' ').replace('%2F', '/')
        st.session_state['selected_persona'] = url_persona if url_persona in persona_options else None
    else:
        st.session_state['selected_persona'] = None  # ← CHANGED TO None


# ==================== SIDEBAR ====================

st.sidebar.title("⚡ GridSense")
st.sidebar.markdown("**48-Hour Grid Risk Forecaster**")
st.sidebar.markdown("---")

# County selection
selected_county = st.sidebar.selectbox(
    "🗺️ Select County",
    options=["-- Select County --"] + counties,  # ← ADD THIS
    index=0 if st.session_state['selected_county'] is None else counties.index(st.session_state['selected_county']) + 1  # ← CHANGE THIS
)

# Only update if a valid county is selected
if selected_county != "-- Select County --":  # ← ADD THIS CHECK
    if selected_county != st.session_state['selected_county']:
        st.session_state['selected_county'] = selected_county
        st.session_state.pop('qr_generated', None)
        st.session_state.pop('qr_buffer', None)
else:  # ← ADD THIS
    st.session_state['selected_county'] = None  # ← ADD THIS

# Persona selection
persona_options = ["SME / Factory", "Clinic / Cold-Chain", "Telecom Site", "Household"]
persona = st.sidebar.selectbox(
    "👤 Select Your Profile",
    options=["-- Select Profile --"] + persona_options,  # ← ADD THIS
    index=0 if st.session_state['selected_persona'] is None else persona_options.index(st.session_state['selected_persona']) + 1  # ← CHANGE THIS
)

# Only update if a valid persona is selected
if persona != "-- Select Profile --":  # ← ADD THIS CHECK
    if persona != st.session_state['selected_persona']:
        st.session_state['selected_persona'] = persona
        st.session_state.pop('qr_generated', None)
        st.session_state.pop('qr_buffer', None)
else:  # ← ADD THIS
    st.session_state['selected_persona'] = None  # ← ADD THIS

# Dark mode toggle
st.sidebar.markdown("---")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)
st.markdown(dark_mode_css if dark_mode else light_mode_css, unsafe_allow_html=True)

# Share section (only show if both county and persona are selected)
if st.session_state['selected_county'] and st.session_state['selected_persona']:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Share this Forecast")

    shareable_url = get_shareable_url(selected_county, persona)

    st.sidebar.text_input(
        "Shareable Link:",
        value=shareable_url,  # ← NOW IT'S INSIDE THE IF
        key="share_url",
        label_visibility="collapsed"
    )
    st.sidebar.caption("👆 Copy this link to share")

    # QR Code generation
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("📱 Generate QR", use_container_width=True):
            current_url = get_shareable_url(selected_county, persona)
            st.session_state['qr_buffer'] = generate_qr_code(current_url)
            st.session_state['qr_generated'] = True
            st.rerun()

    with col2:
        if st.session_state.get('qr_generated'):
            if st.button("❌ Clear", use_container_width=True):
                st.session_state.pop('qr_generated', None)
                st.session_state.pop('qr_buffer', None)
                st.rerun()

    # Display QR code
    if st.session_state.get('qr_generated') and st.session_state.get('qr_buffer'):
        st.sidebar.image(st.session_state['qr_buffer'], use_container_width=True)
        st.sidebar.caption(f"📍 {selected_county.title()} | 👤 {persona}")
        
        st.sidebar.download_button(
            label="💾 Download QR",
            data=BytesIO(st.session_state['qr_buffer'].getvalue()),
            file_name=f"gridsense_{selected_county}_{persona.replace(' ', '_').replace('/', '_')}.png",
            mime="image/png",
            use_container_width=True
        )
# ← THE IF STATEMENT ENDS HERE

st.sidebar.markdown("---")
st.sidebar.info(
    "GridSense uses machine learning to predict power grid stress "
    "48 hours ahead, helping you plan and prepare for potential outages."
)
st.sidebar.markdown("---")
st.sidebar.info(
    "GridSense uses machine learning to predict power grid stress "
    "48 hours ahead, helping you plan and prepare for potential outages."
)


# ==================== MAIN CONTENT ====================

st.title("⚡ GridSense: Power Outage Risk Forecaster")
st.markdown(f"### 48-Hour Forecast for **{selected_county.title()}** County")

# Generate forecast
with st.spinner("Generating forecast..."):
    forecast_df = get_county_forecast(selected_county, final_df, model, hours_ahead=48)

if len(forecast_df) == 0:
    st.error("Unable to generate forecast. No data available for selected county.")
    st.stop()

# Key metrics
max_risk = forecast_df['risk_probability'].max()
avg_risk = forecast_df['risk_probability'].mean()
high_risk_count = len(forecast_df[forecast_df['risk_level'] == 'HIGH'])
peak_hour = forecast_df.loc[forecast_df['risk_probability'].idxmax(), 'hour']

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Peak Risk", f"{max_risk:.1%}", delta=to_risk_level(max_risk), delta_color="inverse")
with col2:
    st.metric("Average Risk", f"{avg_risk:.1%}")
with col3:
    st.metric("High-Risk Hours", high_risk_count)
with col4:
    st.metric("Peak Time", peak_hour.strftime("%I %p"))

st.markdown("---")

# Risk timeline chart
st.subheader("📈 48-Hour Risk Timeline")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_df['hour'],
    y=forecast_df['risk_probability'],
    mode='lines+markers',
    line=dict(color='#0066cc', width=3),
    marker=dict(size=6),
    fill='tozeroy',
    fillcolor='rgba(0, 102, 204, 0.1)'
))

fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="HIGH Risk")
fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="MEDIUM Risk")

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Risk Probability",
    yaxis=dict(tickformat='.0%', range=[0, 1]),
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Risk distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Level Distribution")
    risk_counts = forecast_df['risk_level'].value_counts()
    
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'},
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("⏰ Hourly Risk Levels")
    forecast_df['hour_of_day'] = forecast_df['hour'].dt.hour
    forecast_df['day'] = forecast_df['hour'].dt.day_name()
    
    hourly_data = forecast_df.pivot_table(
        values='risk_probability',
        index='day',
        columns='hour_of_day',
        aggfunc='mean'
    )
    
    fig_heat = px.imshow(
        hourly_data,
        labels=dict(x="Hour of Day", y="Day", color="Risk"),
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# Recommendations
st.subheader(f"💡 Recommended Actions for {persona}")
recommendations = recommend_actions(forecast_df, persona)

for rec in recommendations:
    st.markdown(rec)

st.markdown("---")

# Detailed forecast table
with st.expander("📋 View Detailed Hourly Forecast"):
    display_df = forecast_df[['hour', 'risk_probability', 'risk_level']].copy()
    display_df['Time'] = display_df['hour'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['Risk %'] = (display_df['risk_probability'] * 100).round(1)
    display_df = display_df[['Time', 'Risk %', 'risk_level']]
    display_df.columns = ['Time', 'Risk Probability (%)', 'Risk Level']
    
    st.dataframe(display_df, use_container_width=True, height=400)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p><strong>GridSense</strong> | Powered by Machine Learning | Data updated hourly</p>
        <p>⚠️ This is a predictive tool. Always follow official utility communications.</p>
    </div>
    """,
    unsafe_allow_html=True
)