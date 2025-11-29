# Code.py - Procedural Mock Version (full file)
# UI identical to original Code.py; data engine replaced by procedural bundle generator
# - 5 release sites (TAIWAN_BEACHING_SITES)
# - Bundle of particles tightly around a curvy center track that drifts offshore
# - Some particles end inside LOCAL_BOX (local), some outside (external)
# ---------------------------------------------------------------

import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------- Page config (same UI)
st.set_page_config(layout="wide", page_title="PlasticSource")
st.title("PlasticSource")

# ---------------- 5 release sites (as you provided)
TAIWAN_BEACHING_SITES = {
    "North - Keelung/Yehliu": ((121.60, 25.43), (121.90, 24.98)),
    "East - Hualien":         ((121.55, 24.17), (121.75, 23.67)),
    "East - Taitung":         ((121.28, 22.95), (121.45, 22.70)),
    "South - Kaohsiung":      ((120.08, 22.80), (120.30, 22.45)),
    "West - Hsinchu":         ((120.70, 25.00), (120.90, 24.70)),
}

# ---------------- Constants and helpers
METERS_PER_DEG_LAT = 111320.0

def meters_to_deg_lon(m, lat):
    return m / (METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat)) + 1e-12)

def meters_to_deg_lat(m):
    return m / METERS_PER_DEG_LAT

def haversine(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1); lat1 = np.asarray(lat1)
    lon2 = np.asarray(lon2); lat2 = np.asarray(lat2)
    R = 6371000.0
    phi1 = np.deg2rad(lat1); phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2-lat1); dl = np.deg2rad(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Local classification box (same logic as original)
LOCAL_BOX = {
    "lon_min": 119.5, "lon_max": 122.5,
    "lat_min": 21.5,  "lat_max": 25.8
}

# ---------------- Center-track generators per site
# Each returns a center-line array of shape (T,2) that intentionally drifts offshore
def generate_center_track_for_site(site_name, total_steps, days, start_lon=None, start_lat=None):
    """
    Produce a curvy center track that moves away from Taiwan island (offshore).
    total_steps: number of steps (int)
    days: float total days
    start_lon/lat optional override (use midpoint of site box if provided)
    """
    t = np.linspace(0.0, days, total_steps+1)  # inclusive 0..total_steps
    # default midpoints if not provided:
    if start_lon is None or start_lat is None:
        # choose site midpoint if available:
        mid = None
        for name, box in TAIWAN_BEACHING_SITES.items():
            if name == site_name:
                mid = ((box[0][0]+box[1][0])/2.0, (box[0][1]+box[1][1])/2.0)
                break
        if mid:
            start_lon, start_lat = mid
        else:
            start_lon, start_lat = 120.5, 23.5

    # choose drift direction & curvature parameters per site so paths go offshore
    if "Kaohsiung" in site_name:
        # drift WSW (away from east coast), add meanders and vortices
        lon = start_lon - 0.02*t - 0.08*np.log1p(t) + 0.03*np.sin(t*2.2)
        lat = start_lat - 0.03*t - 0.01*np.cos(t*1.7) + 0.02*np.sin(t*0.5)
    elif "Taitung" in site_name:
    # Drift mạnh ra đông nam → external 100%
        lon = start_lon + 0.10 * t + 0.05 * np.sin(t * 0.8)         # lon tăng mạnh → ra biển
        lat = start_lat - 0.09 * t + 0.03 * np.cos(t * 1.7)         # lat giảm xuống < 21 → external
    elif "Hualien" in site_name:
        # similar to Taitung but milder
        lon = start_lon + 0.02*np.sin(t/1.1) + 0.005*np.cos(t*0.9)
        lat = start_lat - 0.035*t - 0.01*np.sin(t*1.3)
    elif "Keelung" in site_name:
    # Drift ra đông bắc thật xa khỏi local box
        lon = start_lon + 0.08 * t + 0.04 * np.sin(t * 1.3)         # tăng lon → đi ra biển đông
        lat = start_lat + 0.06 * t + 0.03 * np.cos(t * 1.1)         # tăng lat → vượt 25.8 → external
    elif "Hsinchu" in site_name:
        # drift westward slightly and then south-west offshore
        lon = start_lon - 0.015*t + 0.02*np.sin(t*1.3)
        lat = start_lat - 0.02*t + 0.01*np.cos(t*1.1)
    else:
        lon = start_lon - 0.01*t + 0.01*np.sin(t)
        lat = start_lat - 0.01*t + 0.01*np.cos(t)

    center = np.vstack([lon, lat]).T
    return center

# ---------------- The procedural run_backtracking (replacement engine)
def run_backtracking(release_sites, n_particles_per_site, n_steps, dt_minutes, prov=None):
    """
    release_sites: list of tuples (site_name, (lonA,latA), (lonB,latB)) OR (site_name, box)
    For compatibility with original Code.py we accept a list of (site_name, box)
    Returns DataFrame with columns like original Code.py
    """
    rng = np.random.default_rng(42)
    base_time = datetime.utcnow()
    all_rows = []
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    # Determine days from steps & dt (mirrors original)
    days = (n_steps * dt_minutes) / (24*60)

    # We will iterate through provided release_sites (UI passes a single selected site box)
    for site_entry in release_sites:
        # site_entry format: (site_name, site_box) where site_box is ((lon1,lat1),(lon2,lat2))
        if isinstance(site_entry, (list,tuple)) and len(site_entry) == 2:
            site_name, site_box = site_entry
        elif isinstance(site_entry, (list,tuple)) and len(site_entry) >= 3:
            site_name = site_entry[0]; site_box = site_entry[1]
        else:
            # fallback
            site_name = str(site_entry)
            site_box = TAIWAN_BEACHING_SITES.get(site_name, ((120.5,23.5),(120.5,23.5)))

        lon_center = (site_box[0][0] + site_box[1][0]) / 2.0
        lat_center = (site_box[0][1] + site_box[1][1]) / 2.0

        # center line (curvy) that drifts offshore by design
        center_track = generate_center_track_for_site(site_name, n_steps, days, lon_center, lat_center)
        T = center_track.shape[0]

        # offsets: very small (tight bundle). Use meters->deg scale approx
        # Allow slightly anisotropic noise along-track vs cross-track
        sigma_cross_deg = 0.0035  # ~ few hundred meters
        sigma_along_deg = 0.0008

        # We will create two populations so dataset includes both local and external:
        # - majority follow the center track (external)
        # - some fraction jitter less and remain near coast (local)
        local_fraction = 0.25  # ~25% particles will tend to remain 'local' by applying smaller radial offset and shifting slightly inward

        for p_idx in range(n_particles_per_site):
            pid = f"{site_name.replace(' ','_')}_{p_idx}"

            # choose if this particle will be biased local or external
            is_local_biased = rng.random() < local_fraction

            # draw base offsets that are constant along trajectory (keeps bundle shape)
            # cross-track offset larger, along-track small
            cross = rng.normal(0.0, sigma_cross_deg)
            along = rng.normal(0.0, sigma_along_deg)

            # if local-biased, reduce cross offset so it stays nearer coast and may end within LOCAL_BOX
            if is_local_biased:
                cross *= 0.35
                # slightly nudge toward island (inward) by moving small fraction opposite to offshore drift
                # implement by a small sign flip based on site
                # compute small inward vector:
                inward_factor = 0.007  # small deg shift toward island
                if "Taitung" in site_name or "Hualien" in site_name:
                    # coast on east -> nudge slightly west (toward lon -)
                    cross -= inward_factor * 0.4
                elif "Kaohsiung" in site_name:
                    cross += inward_factor * 0.2
                elif "Keelung" in site_name:
                    cross -= inward_factor * 0.2
                elif "Hsinchu" in site_name:
                    cross += inward_factor * 0.05

            # build particle track anchored to center line but with small offsets and time-varying micro-meander
            for step_idx in range(T):
                cx, cy = center_track[step_idx]
                # micro meander along time to add twists: combination of sinusoids scaled small
                micro_lon = 0.0015 * np.sin(step_idx * 0.25 + p_idx*0.11)
                micro_lat = 0.0012 * np.cos(step_idx * 0.18 + p_idx*0.13)

                # rotate offsets a little to simulate cross-track orientation varying
                theta = 0.02 * np.sin(step_idx * 0.07 + p_idx*0.2)
                dx = along * np.cos(theta) - cross * np.sin(theta)
                dy = along * np.sin(theta) + cross * np.cos(theta)

                lon = cx + dx + micro_lon
                lat = cy + dy + micro_lat

                t_str = (base_time - timedelta(minutes=step_idx*dt_minutes)).isoformat()

                all_rows.append({
                    "particle_id": pid,
                    "run_id": run_id,
                    "site_name": site_name,
                    "lon": float(lon),
                    "lat": float(lat),
                    "step_index": int(step_idx),
                    "time": t_str,
                    "status": "active"
                })

    df = pd.DataFrame(all_rows)
    diagnostics = {"total_particles": df["particle_id"].nunique()}
    return df, run_id, diagnostics

# ---------------- UI (sidebar) - keep same fields as original Code.py
st.sidebar.header("Input Parameters")

sample_sites = {name: ((box[0][0]+box[1][0])/2.0, (box[0][1]+box[1][1])/2.0)
                for name, box in TAIWAN_BEACHING_SITES.items()}

site_name = st.sidebar.selectbox("Select Start Point", list(sample_sites.keys()))
init_lon, init_lat = sample_sites[site_name]

n_particles = st.sidebar.slider("Number of Particles", 10, 1000, 100)
days = st.sidebar.selectbox("Backtracking Days", [7, 14, 30], index=1)
dt_mins = st.sidebar.slider("Step Size (min)", 10, 120, 10)
total_steps = int(days * 24 * 60 / dt_mins)

st.sidebar.info(f"Total Steps: {total_steps}")

# ---------------- Run (UI button)
# Keep same signature call but prov is unused in mock
if st.button("Start Backtracking"):
    with st.spinner("Calculating..."):
        # pass the selected site's pair (box) to run_backtracking for compatibility
        site_box = TAIWAN_BEACHING_SITES[site_name]
        df, run_id, diag = run_backtracking([(site_name, site_box)], n_particles, total_steps, dt_mins, prov=None)
    st.success("Done!")
    st.session_state["df"] = df
    st.session_state["run_id"] = run_id

# ---------------- Visualization & Stats (keep same format as original)
if "df" in st.session_state:
    df = st.session_state["df"]

    # compute per-particle metrics and final classification
    particles = df.groupby("particle_id")
    stats_direct = []
    stats_total = []
    source_results = {}  # pid -> color

    for pid, group in particles:
        g = group.sort_values("step_index")
        lons = g["lon"].values
        lats = g["lat"].values

        # direct distance
        d_direct = haversine(lons[0], lats[0], lons[-1], lats[-1])
        stats_direct.append(d_direct)

        if len(lons) > 1:
            d_segs = haversine(lons[:-1], lats[:-1], lons[1:], lats[1:])
            stats_total.append(np.sum(d_segs))
        else:
            stats_total.append(0.0)

        # classification based on final point (LOCAL_BOX)
        origin_lon = lons[-1]
        origin_lat = lats[-1]
        is_local = (
            LOCAL_BOX["lon_min"] <= origin_lon <= LOCAL_BOX["lon_max"] and
            LOCAL_BOX["lat_min"] <= origin_lat <= LOCAL_BOX["lat_max"]
        )
        source_results[pid] = [50, 255, 50, 200] if is_local else [255, 50, 50, 200]

    avg_direct = np.mean(stats_direct) if stats_direct else 0.0
    avg_total = np.mean(stats_total) if stats_total else 0.0

    # ratios
    n_local = sum(1 for c in source_results.values() if c[1] == 255)
    n_total = len(source_results)
    ratio_local = (n_local / n_total) * 100 if n_total > 0 else 0.0
    ratio_ext = 100.0 - ratio_local

    # Stats panel (same look)
    st.markdown("### 📊 Drift Statistics & Source Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Displacement", f"{avg_direct/1000:.1f} km")
    c2.metric("Total Distance", f"{avg_total/1000:.1f} km")
    c3.metric("Local Source", f"{ratio_local:.1f}%")
    c4.metric("External Source", f"{ratio_ext:.1f}%")
    st.divider()

    # Map rendering (PathLayer + Scatter)
    st.subheader("Trajectory Map")

    pids = df["particle_id"].unique()
    # sample up to 300 particles for performance (deterministic)
    sample_n = min(len(pids), 300)
    rng = np.random.default_rng(42)
    if len(pids) > sample_n:
        show_pids = rng.choice(pids, sample_n, replace=False)
    else:
        show_pids = pids

    df_show = df[df["particle_id"].isin(show_pids)]

    paths = []
    for pid in show_pids:
        d = df_show[df_show["particle_id"]==pid].sort_values("step_index")
        path = d[["lon","lat"]].values.tolist()
        color = source_results.get(pid, [200,200,200,200])
        # slight alpha variation to give thickness impression
        paths.append({"path": path, "color": color})

    starts = df_show[df_show["step_index"]==0]

    layer_path = pdk.Layer(
        "PathLayer", data=paths, get_path="path",
        get_color="color", width_min_pixels=2, opacity=0.9
    )
    layer_scatter = pdk.Layer(
        "ScatterplotLayer", data=starts, get_position=["lon","lat"],
        get_color=[255,100,0,255], get_radius=150
    )

    view_state = pdk.ViewState(
        latitude=starts["lat"].mean() if not starts.empty else init_lat,
        longitude=starts["lon"].mean() if not starts.empty else init_lon,
        zoom=6
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer_path, layer_scatter],
        initial_view_state=view_state,
        map_style="dark"
    ))

    # CSV download (same format)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "traj.csv")
