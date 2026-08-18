"""
Hurricane Watch — simplest version.

Run:
    pip install streamlit pydeck tropycal
    streamlit run app.py
"""

import streamlit as st
import pydeck as pdk
from tropycal import realtime

st.set_page_config(page_title="Hurricane Watch", layout="wide")
st.title("🌀 Hurricane Watch")


# Colour by wind speed (knots) -> Saffir-Simpson.
def color(vmax):
    v = vmax or 0
    if v >= 137: return [255, 45, 85]      # Cat 5
    if v >= 113: return [255, 96, 55]      # Cat 4
    if v >= 96:  return [255, 149, 0]      # Cat 3
    if v >= 83:  return [255, 214, 10]     # Cat 2
    if v >= 64:  return [255, 245, 120]    # Cat 1
    if v >= 34:  return [48, 209, 220]     # Tropical Storm
    return [100, 160, 255]                 # Depression


@st.cache_data(ttl=600, show_spinner="Loading active storms…")
def get_storms():
    rt = realtime.Realtime()
    out = []
    for sid in rt.list_active_storms():
        s = rt.get_storm(sid)
        track = [[float(lo), float(la)] for lo, la in zip(s.lon, s.lat)]
        out.append({
            "id": sid,
            "name": str(s.name).title(),
            "track": track,
            "pos": track[-1],
            "vmax": float(s.vmax[-1]),
        })
    return out


storms = get_storms()

if not storms:
    st.info("No active storms right now.")
    st.stop()

# One row per storm on the map.
paths = [{"path": s["track"], "color": color(s["vmax"])} for s in storms]
dots = [{"position": s["pos"], "color": color(s["vmax"]),
         "name": s["name"], "wind": f'{s["vmax"]:.0f} kt'} for s in storms]

st.pydeck_chart(pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    initial_view_state=pdk.ViewState(latitude=25, longitude=-55, zoom=2),
    layers=[
        pdk.Layer("PathLayer", paths, get_path="path", get_color="color",
                  width_min_pixels=3),
        pdk.Layer("ScatterplotLayer", dots, get_position="position",
                  get_fill_color="color", get_radius=60000,
                  radius_min_pixels=6, pickable=True),
    ],
    tooltip={"text": "{name}\n{wind}"},
))

# Pick a storm to read its full NHC discussion.
name = st.selectbox("Storm", [s["name"] for s in storms])
storm = next(s for s in storms if s["name"] == name)
st.metric("Max wind", f'{storm["vmax"]:.0f} kt')

with st.expander("📄 Full NHC discussion"):
    try:
        rt = realtime.Realtime()
        disc = rt.get_storm(storm["id"]).get_nhc_discussion(forecast=-1)
        st.text(disc["text"])
    except Exception:
        st.caption("No discussion available.")
