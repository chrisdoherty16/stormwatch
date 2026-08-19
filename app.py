"""
Hurricane Watch — global tropical-cyclone monitoring board.

Map centrepiece shows every active system worldwide (storms + invests),
with forecast track + cone of uncertainty on the live storms, and a grid
of summary cards for everything being monitored.

Run:
    uv run streamlit run app.py
"""

import math
import time
import threading
from datetime import timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    BERMUDA_TZ = ZoneInfo("Atlantic/Bermuda")
except Exception:
    BERMUDA_TZ = None

import numpy as np
import streamlit as st
import pydeck as pdk
from tropycal import realtime

# Optional AI layer. Absent library or key -> app runs unchanged.
try:
    from google import genai
except Exception:
    genai = None

# Preferred Flash models, best first. We pick whichever your key can access,
# so this survives Google renaming models.
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3.7-flash", "gemini-3.6-flash",
                 "gemini-3.5-flash", "gemini-2.5-flash"]

st.set_page_config(page_title="Hurricane Watch", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1rem; max-width: 100%;}
  .hw-legend span {display:inline-flex; align-items:center; margin-right:16px;
      font-size:.76rem; color:#c4cad6;}
  .hw-dot {width:11px; height:11px; border-radius:50%; margin-right:6px;
      display:inline-block;}
  .hw-card {background:#12151c; border:1px solid #222836; border-radius:12px;
      padding:14px 16px 12px; margin-bottom:12px; border-left:5px solid #444;}
  .hw-card h4 {margin:0 0 2px; font-size:1.02rem; color:#e9ecf2;}
  .hw-card .klass {font-size:.8rem; color:#aab2c0; margin-bottom:10px;}
  .hw-grid {display:flex; gap:16px; font-size:.8rem;}
  .hw-grid div span {display:block; color:#7b8496; font-size:.66rem;
      text-transform:uppercase; letter-spacing:.4px;}
  .hw-grid div b {font-size:.98rem; color:#e9ecf2;}
</style>
""", unsafe_allow_html=True)

st.title("🌀 Hurricane Watch")
st.caption("Live global tropical-cyclone monitor · NHC + JTWC via Tropycal")


# ---------------------------------------------------------------------------
# Classification + colour (Saffir-Simpson, winds in knots)
# ---------------------------------------------------------------------------
def category(vmax):
    v = vmax or 0
    if v >= 137: return 5
    if v >= 113: return 4
    if v >= 96:  return 3
    if v >= 83:  return 2
    if v >= 64:  return 1
    return 0


def color(vmax, invest=False):
    if invest:
        return [150, 160, 175]
    return {
        5: [255, 45, 85], 4: [255, 96, 55], 3: [255, 149, 0],
        2: [255, 214, 10], 1: [255, 245, 120],
    }.get(category(vmax), [48, 209, 220] if (vmax or 0) >= 34 else [100, 160, 255])


def classify(vmax, stype, basin, invest=False):
    if invest:
        return "Invest / Area of Interest"
    v = vmax or 0
    cat = category(v)
    special = {"EX": "Post-Tropical Cyclone", "SS": "Subtropical Storm",
               "SD": "Subtropical Depression", "LO": "Remnant Low",
               "DB": "Disturbance", "WV": "Tropical Wave"}
    if stype in special:
        return special[stype]
    if cat >= 1:
        if basin == "west_pacific":
            return "Super Typhoon" if v >= 130 else "Typhoon"
        if basin in ("north_indian", "south_indian", "australia", "south_pacific"):
            return f"Cyclone (Cat {cat})"
        return f"Hurricane (Cat {cat})"
    if v >= 34:
        return "Tropical Storm"
    return "Tropical Depression"


_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def movement(track, times):
    if len(track) < 2 or len(times) < 2:
        return "—"
    (lon1, lat1), (lon2, lat2) = track[-2], track[-1]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    brg = (math.degrees(math.atan2(y, x)) + 360) % 360
    comp = _COMPASS[round(brg / 22.5) % 16]
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    dist = 3440.065 * 2 * math.asin(math.sqrt(a))
    try:
        hrs = (times[-1] - times[-2]).total_seconds() / 3600
        return f"{comp} at {dist / hrs:.0f} kt" if hrs else comp
    except Exception:
        return comp


# ---------------------------------------------------------------------------
# Time formatting — convert a UTC fix time to Bermuda local (AST/ADT).
# ---------------------------------------------------------------------------
def fmt_bermuda(dt):
    """Return e.g. '19 Aug 3:00 AM AST'. dt is a UTC-aware/naive datetime."""
    if dt is None or not hasattr(dt, "strftime"):
        return "—"
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(BERMUDA_TZ) if BERMUDA_TZ else dt
        hour12 = local.hour % 12 or 12
        ampm = "AM" if local.hour < 12 else "PM"
        tz = local.tzname() or ("AST" if BERMUDA_TZ else "UTC")
        return f"{local.day} {local.strftime('%b')} {hour12}:{local.minute:02d} {ampm} {tz}"
    except Exception:
        return dt.strftime("%d %b %H:%MZ")


# ---------------------------------------------------------------------------
# Cone of uncertainty -> polygon ring for a PolygonLayer
# ---------------------------------------------------------------------------
def cone_polygon(forecast, basin):
    try:
        from tropycal import utils
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cone = utils.generate_nhc_cone(forecast, basin, cone_days=5)
        grid = np.asarray(cone["cone"], dtype=float)
        cs = plt.contour(np.asarray(cone["lon"]), np.asarray(cone["lat"]),
                         grid, levels=[0.5])
        best = max(cs.allsegs[0], key=len) if cs.allsegs[0] else None
        plt.close("all")
        if best is None or len(best) < 3:
            return None
        return [[float(x), float(y)] for x, y in best]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live data — NHC + JTWC, storms + invests, forecast + cone
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Forecast outlook — peak intensity, when first reached, and trend.
# Derived from Tropycal's structured forecast vmax array (no AI, reliable).
# ---------------------------------------------------------------------------
def forecast_outlook(fc, cur_vmax, basin):
    try:
        fhrs = list(fc.get("fhr", []))
        vmaxs = list(fc.get("vmax", []))
        pairs = [(h, v) for h, v in zip(fhrs, vmaxs) if v is not None]
        if not pairs:
            return None
        init = fc.get("init")
        peak_v = max(v for _, v in pairs)
        peak_klass = classify(peak_v, None, basin)

        # first forecast time the peak classification is reached
        by = ""
        for h, v in pairs:
            if classify(v, None, basin) == peak_klass:
                if init is not None:
                    try:
                        by = (init + timedelta(hours=int(h))).strftime("%a %HZ")
                    except Exception:
                        by = f"+{int(h)}h"
                else:
                    by = f"+{int(h)}h"
                break

        final_v = pairs[-1][1]
        if cur_vmax and peak_v >= cur_vmax + 5:
            trend = "↑ strengthening"
        elif cur_vmax and final_v <= cur_vmax - 5:
            trend = "↓ weakening"
        else:
            trend = "→ steady"
        return {"peak_klass": peak_klass, "peak_v": peak_v, "by": by, "trend": trend}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Optional Gemini brief — one plain sentence per storm. Fully cached and
# strictly additive: no key / any failure -> deterministic fallback sentence.
# ---------------------------------------------------------------------------
def gemini_key():
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def gemini_client():
    if genai is None or not gemini_key():
        return None
    try:
        return genai.Client(api_key=gemini_key())
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
@st.cache_data(ttl=3600, show_spinner=False)
def gemini_model():
    client = gemini_client()
    if client is None:
        return None
    try:
        available = [m.name.replace("models/", "") for m in client.models.list()]
    except Exception as e:
        _AI_ERROR["msg"] = f"models.list failed: {type(e).__name__}: {str(e)[:150]}"
        # Fall back to a widely-available model name rather than giving up.
        return "gemini-2.5-flash"
    for pref in GEMINI_MODELS:
        if pref in available:
            return pref
    flash = [n for n in available if "flash" in n
             and not any(x in n for x in ("image", "tts", "live", "lite"))]
    if flash:
        return sorted(flash, reverse=True)[0]
    _AI_ERROR["msg"] = f"No flash model found. Available: {available[:8]}"
    return "gemini-2.5-flash"


def storm_facts(s):
    bits = [s["name"], s["klass"]]
    if s["vmax"]:
        bits.append(f"{s['vmax']:.0f} kt winds")
    if s["mslp"]:
        bits.append(f"{s['mslp']:.0f} mb")
    if s["move"] and s["move"] != "—":
        bits.append(f"moving {s['move']}")
    fo = s.get("outlook")
    if fo:
        if fo["peak_klass"] != s["klass"] and fo["by"]:
            bits.append(f"forecast to reach {fo['peak_klass']} by {fo['by']}")
        bits.append(fo["trend"].split(" ", 1)[-1])
    return "; ".join(bits)


def fallback_brief(s):
    head = f"{s['name']} is a {s['klass'].lower()}"
    if s["vmax"]:
        head += f" with {s['vmax']:.0f} kt winds"
    parts = [head]
    if s["move"] and s["move"] != "—":
        parts.append(f"moving {s['move']}")
    fo = s.get("outlook")
    if fo and fo["peak_klass"] != s["klass"] and fo["by"]:
        parts.append(f"forecast to reach {fo['peak_klass'].lower()} by {fo['by']}")
    return ", ".join(parts) + "."


# Holds the last Gemini error so we can show it in a diagnostics expander
# instead of silently falling back. Cleared on each successful call.
_AI_ERROR = {"msg": ""}


@st.cache_data(ttl=1800, show_spinner=False)
def ai_brief(facts, discussion=None):
    """One-sentence brief. If the NHC forecast discussion is available, the AI
    reads that narrative (landfall threats, timing, confidence); otherwise it
    summarises the structured facts."""
    client, model = gemini_client(), gemini_model()
    if client is None or not model:
        _AI_ERROR["msg"] = "No client/model (key or library issue)."
        return None
    if discussion:
        prompt = ("You are briefing a reinsurance underwriter. Read this official "
                  "NHC forecast discussion and give ONE concise, plain sentence "
                  "capturing what the storm is doing now and the key forecast threat "
                  "(landfall/intensity/timing) the forecaster emphasises. No preamble, "
                  "no lists.\n\nDISCUSSION:\n" + discussion[:6000])
    else:
        prompt = ("You are briefing a reinsurance underwriter. In ONE concise, plain "
                  "sentence, state what this tropical system is doing now and its key "
                  "forecast. No preamble, no lists.\nFacts: " + facts)
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = (resp.text or "").strip()
        if not text:
            _AI_ERROR["msg"] = "Model returned empty text."
            return None
        _AI_ERROR["msg"] = ""  # success
        return text
    except Exception as e:
        _AI_ERROR["msg"] = f"{type(e).__name__}: {str(e)[:200]}"
        return None


def _attr(s, key):
    """Read a RealtimeStorm attribute whether it's exposed directly or in .attrs."""
    v = getattr(s, key, None)
    if v is None and hasattr(s, "attrs"):
        try:
            v = s.attrs.get(key)
        except Exception:
            v = None
    return v


def _formation_prob(s):
    """Return {'p2': int|None, 'p7': int|None, 'risk': str} for an invest.
    NHC populates 2-day and 5/7-day odds; JTWC returns 'N/A'."""
    def num(*keys):
        for k in keys:
            v = _attr(s, k)
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip().rstrip("%").isdigit():
                return int(v.strip().rstrip("%"))
        return None
    risk = ""
    for k in ("risk_7day", "risk_5day", "risk_2day"):
        v = _attr(s, k)
        if isinstance(v, str) and v not in ("", "N/A"):
            risk = v
            break
    return {"p2": num("prob_2day"), "p7": num("prob_7day", "prob_5day"), "risk": risk}


def _read(rt, want_cone, source="NHC"):
    systems = []
    for sid in rt.list_active_storms():
        try:
            s = rt.get_storm(sid)
            track = [[float(lo), float(la)] for lo, la in zip(s.lon, s.lat)]
            if not track:
                continue
            times = list(getattr(s, "date", []) or getattr(s, "time", []))
            vmax = float(s.vmax[-1]) if len(getattr(s, "vmax", [])) else 0.0
            stype = s.type[-1] if getattr(s, "type", None) is not None else None
            basin = getattr(s, "basin", None)
            invest = bool(getattr(s, "invest", False))
            mslp_raw = s.mslp[-1] if getattr(s, "mslp", None) is not None else None
            mslp = float(mslp_raw) if mslp_raw and not math.isnan(float(mslp_raw)) else None

            # forecast track + cone + outlook (live storms only, not invests)
            fc_track, cone, outlook = [], None, None
            if not invest:
                try:
                    fc = s.get_forecast_realtime()
                    fc_track = [[float(lo), float(la)]
                                for lo, la in zip(fc["lon"], fc["lat"])]
                    outlook = forecast_outlook(fc, vmax, basin)
                    if want_cone and fc_track:
                        cone = cone_polygon(fc, basin or "north_atlantic")
                except Exception:
                    pass

            # NHC forecast discussion text (for the AI brief). NHC/CPHC only.
            discussion = None
            if source == "NHC" and not invest:
                try:
                    d = s.get_nhc_discussion(forecast=-1)
                    discussion = d.get("text") if isinstance(d, dict) else None
                except Exception:
                    discussion = None

            # invest formation odds (for the outlook panel)
            prob = _formation_prob(s) if invest else None

            systems.append({
                "id": sid, "name": str(s.name).title(),
                "basin": basin, "invest": invest, "source": source,
                "track": track, "pos": track[-1],
                "fc_track": ([track[-1]] + fc_track) if fc_track else [],
                "cone": cone, "outlook": outlook,
                "discussion": discussion, "prob": prob,
                "vmax": vmax, "mslp": mslp,
                "klass": classify(vmax, stype, basin, invest),
                "cat": category(vmax),
                "move": movement(track, times),
                "time": fmt_bermuda(times[-1]) if times else "—",
            })
        except Exception:
            continue
    return systems


# NHC (Atlantic / E-Pac / C-Pac) — fast, shown immediately.
@st.cache_data(ttl=600, show_spinner="Loading storms (NHC)…")
def get_nhc():
    try:
        return _read(realtime.Realtime(), want_cone=True)
    except Exception:
        return []


# JTWC (rest of world) — slow (~2 min), loaded in a background thread so it
# never blocks the NHC map. Result lives in this shared object across reruns.
@st.cache_resource(ttl=600)
def jtwc_state():
    return {"data": None, "started": False}


def get_systems():
    nhc = get_nhc()
    state = jtwc_state()
    if not state["started"]:
        state["started"] = True

        def bg():
            try:
                state["data"] = _read(
                    realtime.Realtime(jtwc=True, jtwc_source="ucar"),
                    want_cone=True, source="JTWC")
            except Exception:
                state["data"] = []
        threading.Thread(target=bg, daemon=True).start()

    out = list(nhc)
    if state["data"]:
        have = {s["id"] for s in out}
        out += [s for s in state["data"] if s["id"] not in have]
    loading = state["data"] is None
    return out, loading


systems, jtwc_loading = get_systems()
if jtwc_loading:
    st.caption("⏳ Loading global (JTWC) systems in the background…")

if not systems:
    st.info("No active tropical systems anywhere right now.")
    st.stop()

# strongest first
systems.sort(key=lambda s: (s["invest"], -s["vmax"]))


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
_LEG = [("Cat 5", [255, 45, 85]), ("Cat 4", [255, 96, 55]),
        ("Cat 3", [255, 149, 0]), ("Cat 2", [255, 214, 10]),
        ("Cat 1", [255, 245, 120]), ("Trop. Storm", [48, 209, 220]),
        ("Depression", [100, 160, 255]), ("Invest", [150, 160, 175])]
st.markdown('<div class="hw-legend">' + "".join(
    f'<span><span class="hw-dot" style="background:rgb({r},{g},{b})"></span>{l}</span>'
    for l, (r, g, b) in _LEG) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAP — all systems at once, satellite basemap, forecast + cone
# ---------------------------------------------------------------------------
obs_paths, fc_paths, cones, dots = [], [], [], []
for s in systems:
    col = color(s["vmax"], s["invest"])
    if len(s["track"]) >= 2:
        obs_paths.append({"path": s["track"], "color": col})
    if len(s["fc_track"]) >= 2:
        fc_paths.append({"path": s["fc_track"], "color": [255, 255, 255]})
    if s["cone"]:
        cones.append({"polygon": s["cone"]})
    dots.append({
        "position": s["pos"], "color": col,
        "name": s["name"], "klass": s["klass"],
        "wind": f'{s["vmax"]:.0f} kt ({s["vmax"] * 1.15078:.0f} mph)'
                if s["vmax"] else "—",
        "pressure": f'{s["mslp"]:.0f} mb' if s["mslp"] else "N/A",
        "move": s["move"],
    })

st.pydeck_chart(pdk.Deck(
    map_provider="carto", map_style="dark",
    initial_view_state=pdk.ViewState(latitude=15, longitude=-40, zoom=1.2),
    layers=[
        pdk.Layer("PolygonLayer", cones, get_polygon="polygon",
                  get_fill_color=[255, 255, 255, 30],
                  get_line_color=[255, 255, 255, 110],
                  stroked=True, filled=True, line_width_min_pixels=1),
        pdk.Layer("PathLayer", obs_paths, get_path="path", get_color="color",
                  width_min_pixels=7, opacity=0.25),
        pdk.Layer("PathLayer", obs_paths, get_path="path", get_color="color",
                  width_min_pixels=2.5),
        pdk.Layer("PathLayer", fc_paths, get_path="path", get_color="color",
                  width_min_pixels=1.5, opacity=0.9),
        pdk.Layer("ScatterplotLayer", dots, get_position="position",
                  get_fill_color="color", get_radius=55000, radius_min_pixels=6,
                  stroked=True, get_line_color=[10, 12, 20],
                  line_width_min_pixels=2, pickable=True),
    ],
    tooltip={
        "html": "<b>{name}</b><br/>{klass}<br/>💨 {wind}<br/>"
                "🔽 {pressure}<br/>➡️ {move}",
        "style": {"backgroundColor": "#12151c", "color": "#e6e9ef",
                  "fontSize": "12px", "padding": "8px 10px",
                  "borderRadius": "6px", "border": "1px solid #2a2f3a"},
    },
), height=520)


# ---------------------------------------------------------------------------
# SUMMARY CARDS — everything being monitored
# ---------------------------------------------------------------------------
_ai_on = bool(gemini_key())
storms = [s for s in systems if not s["invest"]]
invests = [s for s in systems if s["invest"]]

st.subheader(f"Currently monitoring — {len(storms)} storm"
             f"{'s' if len(storms) != 1 else ''}")
st.caption("🧠 AI briefs on" if _ai_on else
           "AI briefs off — add GEMINI_API_KEY to .streamlit/secrets.toml to enable.")

cols = st.columns(3)
for i, s in enumerate(storms):
    r, g, b = color(s["vmax"], s["invest"])
    fo = s.get("outlook")

    # one-line brief. If key present: AI reads the NHC discussion when available,
    # else summarises the facts. No key: deterministic fallback sentence.
    disc = s.get("discussion")
    brief = ai_brief(storm_facts(s), disc) if _ai_on else None
    brief_color = "#ff8ac4" if brief else "#cdd3de"  # pink when AI-written
    read_tag = ("📄 from NHC discussion" if (brief and disc) else "")
    if not brief:
        brief = fallback_brief(s)

    # forecast-to-become badge (only when an intensity-class change is forecast)
    fc_badge = ""
    if fo and fo["peak_klass"] != s["klass"] and fo["by"]:
        fc_badge = (f'<span style="font-size:.72rem;color:#8fb7ff;'
                    f'background:#16203a;border-radius:20px;padding:2px 9px;">'
                    f'⏱ {fo["peak_klass"]} by {fo["by"]}</span>')
    trend_chip = ""
    if fo:
        tcol = {"↑": "#ff6b6b", "↓": "#5ec8ff", "→": "#9aa4b2"}[fo["trend"][0]]
        trend_chip = (f'<span style="font-size:.72rem;color:{tcol};'
                      f'margin-left:6px;">{fo["trend"]}</span>')

    with cols[i % 3]:
        st.markdown(f"""
<div class="hw-card" style="border-left-color:rgb({r},{g},{b})">
  <h4>{s['name']}</h4>
  <div class="klass">{s['klass']}</div>
  <div style="font-size:.82rem;color:{brief_color};font-style:italic;margin-bottom:4px">
    {brief}
  </div>
  <div style="font-size:.62rem;color:#6b7686;margin-bottom:8px">{read_tag}</div>
  <div style="margin-bottom:10px">{fc_badge}{trend_chip}</div>
  <div class="hw-grid">
    <div><span>Winds</span><b>{f'{s["vmax"]:.0f} kt' if s['vmax'] else '—'}</b></div>
    <div><span>Pressure</span><b>{f'{s["mslp"]:.0f} mb' if s['mslp'] else 'N/A'}</b></div>
    <div><span>Moving</span><b>{s['move']}</b></div>
  </div>
  <div style="color:#697083;font-size:.68rem;margin-top:8px">
    {s['pos'][1]:.1f}°, {s['pos'][0]:.1f}° · {s['time']}
  </div>
</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FORMATION OUTLOOK — areas of interest / invests and their development odds
# ---------------------------------------------------------------------------
def _prob_color(p):
    if p is None:
        return "#9aa4b2"
    if p >= 60:
        return "#ff6b6b"   # high
    if p >= 40:
        return "#ffb020"   # medium
    return "#5ec8ff"       # low


st.markdown("---")
st.subheader(f"🌱 Formation outlook — {len(invests)} area"
             f"{'s' if len(invests) != 1 else ''} of interest")
st.caption("Disturbances NHC is watching, with their chance of developing into a "
           "tropical cyclone. (NHC basins only; JTWC areas show no odds.)")

if not invests:
    st.caption("No areas of interest being monitored right now.")
else:
    ocols = st.columns(3)
    for i, s in enumerate(invests):
        pr = s.get("prob") or {}
        p2, p7, risk = pr.get("p2"), pr.get("p7"), pr.get("risk")
        c7 = _prob_color(p7)
        with ocols[i % 3]:
            st.markdown(f"""
<div class="hw-card" style="border-left-color:{c7}">
  <h4>{s['name']}</h4>
  <div class="klass">{risk + ' formation risk' if risk else 'Area of interest'}</div>
  <div class="hw-grid">
    <div><span>48-hour</span><b style="color:{_prob_color(p2)}">
        {f'{p2}%' if p2 is not None else '—'}</b></div>
    <div><span>7-day</span><b style="color:{c7}">
        {f'{p7}%' if p7 is not None else '—'}</b></div>
  </div>
  <div style="color:#697083;font-size:.68rem;margin-top:8px">
    {s['pos'][1]:.1f}°, {s['pos'][0]:.1f}° · {s['time']}
  </div>
</div>""", unsafe_allow_html=True)


# Background JTWC not ready yet -> wait briefly, then rerun to pick it up.
if jtwc_loading:
    time.sleep(4)
    st.rerun()
