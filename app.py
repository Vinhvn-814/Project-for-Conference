import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import uuid

st.set_page_config(layout="wide", page_title="PlasticSource – Simulation")
st.title("PlasticSource (Simulated Ocean Model)")

# ============================================================
# 1) SIMULATED CURRENT FIELD (NO REAL DATA REQUIRED)
# ============================================================

def synthetic_current(lon, lat, t):
    """
    Fake ocean current model.
    Creates a rotating flow around Taiwan + random small noise.
    Always keeps particles offshore.
    lon, lat are numpy arrays.
    """

    # Center of Taiwan for rotational flow
    center_lon = 121.0
    center_lat = 23.8

    # Vector from center
    dx = lon - center_lon
    dy = lat - center_lat

    # Radial distance
    r = np.sqrt(dx**2 + dy**2) + 1e-6

    # Create circular flow around Taiwan
    u = -dy / r * 0.3    # east-west component
    v = dx / r * 0.3     # north-south component

    # Add small oscillating component based on “time”
    u += 0.05 * np.sin(t/30)
    v += 0.05 * np.cos(t/40)

    return u, v


# ============================================================
# 2) RUNGE–KUTTA 4 BACKTRACKING (USING THE SIMULATED CURRENT)
# ============================================================

def rk4_backtrack(lon, lat, t, dt):
    """One RK4 step backward in time using the simulated ocean model."""
    k1_u, k1_v = synthetic_current(lon, lat, t)
    k2_u, k2_v = synthetic_current(lon - 0.5*dt*k1_u, lat - 0.5*dt*k1_v, t - 0.5*dt)
    k3_u, k3_v = synthetic_current(lon - 0.5*dt*k2_u, lat - 0.5*dt*k2_v, t - 0.5*dt)
    k4_u, k4_v = synthetic_current(lon - dt*k3_u, lat - dt*k3_v, t - dt)

    u = (k1_u + 2*k2_u + 2*k3_u + k4_u)/6
    v = (k1_v + 2*k2_v + 2*k3_v + k4_v)/6

    new_lon = lon - u*dt
    new_lat = lat - v*dt

    return new_lon, new_lat


# ============================================================
# 3) SIMPLE “LAND PREVENTION”
# ============================================================

def avoid_taiwan(lon, lat):
    """Push particles outward if they enter Taiwan bounding box."""
    taiwan_box = {
        "lon_min": 120.0,
        "lon_max": 122.2,
        "lat_min": 21.7,
        "lat_max": 25.5,
    }

    inside = (
        (lon >= taiwan_box["lon_min"]) &
        (lon <= taiwan_box["lon_max"]) &
        (lat >= taiwan_box["lat_min"]) &
        (lat <= taiwan_box["lat_max"])
    )

    # Push outward slightly
    lon[inside] += (lon[inside] - 121.0) * 0.02
    lat[inside] += (lat[inside] - 23.8) * 0.02

    return lon, lat


# ============================================================
# 4) PARTICLE TRACK SIMULATION
# ============================================================

def simulate_track(start_lon, start_lat, days=10, dt_hours=1):
    """Simulate backward trajectory using synthetic ocean model."""

    dt = dt_hours * 3600   # seconds irrelevant here but consistent
    t = 0

    lons = [start_lon]
    lats = [start_lat]

    lon = start_lon
    lat = start_lat

    steps = int(days * 24 / dt_hours)

    for _ in range(steps):
        lon, lat = rk4_backtrack(lon, lat, t, 0.05)  # dt=0.05 for model time
        lon, lat = avoid_taiwan(np.array([lon]), np.array([lat]))
        lon = lon[0]; lat = lat[0]
        lons.append(lon)
        lats.append(lat)
        t -= 1

    df = pd.DataFrame({"lon": lons, "lat": lats})
    return df


# ============================================================
# 5) UI – SAME STYLE AS YOUR ORIGINAL CODE
# ============================================================

st.sidebar.header("Simulation Settings")

site_options = {
    "Hsinchu":  (120.9, 24.8),
    "Kaohsiung": (120.3, 22.6),
    "Taitung":  (121.2, 22.9),
    "Hualien":  (121.6, 24.0),
    "Keelung":  (121.75, 25.15),
}

site = st.sidebar.selectbox("Choose start site", list(site_options.keys()))
days = st.sidebar.slider("Backtracking Days", 5, 60, 20)

start_lon, start_lat = site_options[site]

if st.sidebar.button("Run Simulation"):
    df = simulate_track(start_lon, start_lat, days=days)
    
    # Create unique path ID
    df["id"] = str(uuid.uuid4())

    # Map layer
    line_layer = pdk.Layer(
        "PathLayer",
        df,
        pickable=True,
        get_path="[['lon','lat']]",
        get_color=[255, 50, 80],
        width_scale=3,
        width_min_pixels=2,
    )

    # View
    view = pdk.ViewState(
        longitude=start_lon,
        latitude=start_lat,
        zoom=6.5,
        pitch=30,
    )

    r = pdk.Deck(
        layers=[line_layer],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/light-v10",
    )

    st.pydeck_chart(r)

else:
    st.info("Click **Run Simulation** to start.")


