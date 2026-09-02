from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import quote

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "cities_uzbekistan.csv"
ASSET_DIR = APP_DIR / "assets"
FEEDBACK_PATH = APP_DIR / "feedback_state.json"

FEATURE_COLUMNS = [
    "affordability",
    "grocery_affordability",
    "entertainment",
    "sunset_scenery",
    "safety",
    "jobs",
    "internet",
    "healthcare",
    "education",
    "heritage",
    "nature",
    "climate_comfort",
    "quietness",
    "mobility",
]

FEATURE_LABELS = {
    "affordability": "Affordable living",
    "grocery_affordability": "Affordable groceries",
    "entertainment": "Entertainment",
    "sunset_scenery": "Sunset & scenery",
    "safety": "Safety",
    "jobs": "Jobs & career",
    "internet": "Internet & remote work",
    "healthcare": "Healthcare",
    "education": "Education",
    "heritage": "History & culture",
    "nature": "Nature access",
    "climate_comfort": "Climate comfort",
    "quietness": "Quiet lifestyle",
    "mobility": "Transportation",
}

SYNONYMS = {
    "affordability": [
        "cheap", "cheapest", "affordable", "low cost", "low-cost", "budget",
        "inexpensive", "save money", "low rent", "reasonable rent",
    ],
    "grocery_affordability": [
        "cheap grocery", "cheap groceries", "grocery price", "food price",
        "affordable food", "low food cost", "market price", "cheap market",
    ],
    "entertainment": [
        "entertainment", "nightlife", "fun", "activities", "events", "concert",
        "cinema", "shopping", "restaurants", "social life", "things to do",
    ],
    "sunset_scenery": [
        "sunset", "sunsets", "scenic", "beautiful view", "view", "photography",
        "sky", "romantic", "landscape",
    ],
    "safety": ["safe", "safety", "secure", "low crime", "family friendly"],
    "jobs": ["job", "jobs", "career", "employment", "business", "salary", "startup"],
    "internet": [
        "internet", "wifi", "remote work", "digital nomad", "online work",
        "technology", "tech", "fast connection",
    ],
    "healthcare": ["hospital", "healthcare", "doctor", "medical", "clinic", "health"],
    "education": [
        "school", "education", "university", "college", "student", "children",
        "academic", "learning",
    ],
    "heritage": [
        "history", "historic", "heritage", "culture", "architecture", "museum",
        "old city", "silk road", "traditional",
    ],
    "nature": [
        "nature", "mountain", "mountains", "green", "park", "hiking", "outdoor",
        "river", "lake", "desert", "fresh air",
    ],
    "climate_comfort": [
        "comfortable weather", "mild climate", "climate", "weather", "not too hot",
        "not too cold", "pleasant weather",
    ],
    "quietness": ["quiet", "calm", "peaceful", "slow life", "less crowded", "relaxed"],
    "mobility": [
        "transport", "transportation", "metro", "bus", "airport", "walkable",
        "commute", "easy travel", "connected",
    ],
}

CARD_IMAGES = ["spot-forest.jpg", "spot-lake.jpg", "spot-meadow.jpg"]
DETAIL_IMAGES = ["detail-forest-1.jpg", "detail-lake-1.jpg", "detail-meadow-1.jpg"]


def image_data_uri(filename: str) -> str:
    path = ASSET_DIR / filename
    if not path.exists():
        return ""
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_data
def load_city_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(50).clip(0, 100)
    return df


def city_document(row: pd.Series) -> str:
    feature_words = " ".join(
        FEATURE_LABELS[column] for column in FEATURE_COLUMNS if float(row[column]) >= 72
    )
    return " ".join(
        [
            str(row.get("city", "")),
            str(row.get("region", "")),
            str(row.get("tags", "")),
            str(row.get("description", "")),
            feature_words,
        ]
    )


@st.cache_resource
def build_text_index(documents: Tuple[str, ...]) -> Tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def parse_preference_weights(query: str) -> Tuple[Dict[str, float], List[str]]:
    text = re.sub(r"\s+", " ", query.lower()).strip()
    weights = {feature: 0.0 for feature in FEATURE_COLUMNS}
    matches: List[str] = []

    for feature, phrases in SYNONYMS.items():
        for phrase in phrases:
            if phrase in text:
                weights[feature] += 1.4 if " " in phrase else 1.0
                matches.append(f"{FEATURE_LABELS[feature]} <- '{phrase}'")

    if "family" in text or "children" in text or "kids" in text:
        weights["safety"] += 1.1
        weights["education"] += 0.9
        weights["healthcare"] += 0.6
        matches.append("Family intent -> safety, education, healthcare")
    if "retire" in text or "retirement" in text:
        weights["affordability"] += 0.8
        weights["quietness"] += 1.0
        weights["healthcare"] += 0.8
        matches.append("Retirement intent -> affordability, quietness, healthcare")
    if "young professional" in text or "professional" in text:
        weights["jobs"] += 1.0
        weights["internet"] += 0.7
        weights["entertainment"] += 0.5
        matches.append("Professional intent -> jobs, internet, entertainment")
    if "tourist" in text or "tourism" in text or "visit" in text:
        weights["heritage"] += 0.9
        weights["entertainment"] += 0.5
        weights["mobility"] += 0.4
        matches.append("Tourism intent -> heritage, activities, transportation")

    total = sum(weights.values())
    if total <= 0:
        defaults = {
            "affordability": 1.0,
            "safety": 1.0,
            "healthcare": 0.8,
            "internet": 0.7,
            "mobility": 0.6,
            "entertainment": 0.5,
        }
        weights.update(defaults)
        matches.append("No strong keyword found -> balanced default profile")
        total = sum(weights.values())

    return {key: value / total for key, value in weights.items()}, matches


def combine_weights(text_weights: Dict[str, float], slider_weights: Dict[str, float]) -> Dict[str, float]:
    combined = {
        feature: text_weights.get(feature, 0.0) + slider_weights.get(feature, 0.0) / 5.0
        for feature in FEATURE_COLUMNS
    }
    total = sum(combined.values()) or 1.0
    return {feature: value / total for feature, value in combined.items()}


def load_feedback_state(cities: Iterable[str]) -> Dict[str, Dict[str, int]]:
    default = {str(city): {"likes": 0, "dislikes": 0} for city in cities}
    if not FEEDBACK_PATH.exists():
        return default
    try:
        stored = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        for city in default:
            values = stored.get(city, {})
            default[city]["likes"] = int(values.get("likes", 0))
            default[city]["dislikes"] = int(values.get("dislikes", 0))
    except (OSError, ValueError, TypeError):
        pass
    return default


def save_feedback_state(state: Dict[str, Dict[str, int]]) -> None:
    try:
        FEEDBACK_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def bandit_posterior(city: str, state: Dict[str, Dict[str, int]]) -> float:
    values = state.get(city, {"likes": 0, "dislikes": 0})
    likes = max(0, int(values.get("likes", 0)))
    dislikes = max(0, int(values.get("dislikes", 0)))
    return (likes + 1.0) / (likes + dislikes + 2.0)


def make_dnn_training_data(df: pd.DataFrame, n_samples: int = 1400) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    features = df[FEATURE_COLUMNS].to_numpy(dtype=float) / 100.0
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for _ in range(n_samples):
        city_vector = features[rng.integers(0, len(features))]
        preference = rng.dirichlet(np.ones(len(FEATURE_COLUMNS)) * 0.75)
        interaction = preference * city_vector
        utility = float(np.dot(preference, city_vector))

        idx = {name: FEATURE_COLUMNS.index(name) for name in FEATURE_COLUMNS}
        utility += 0.09 * min(
            preference[idx["heritage"]] * city_vector[idx["heritage"]],
            preference[idx["entertainment"]] * city_vector[idx["entertainment"]],
        )
        utility += 0.08 * min(
            preference[idx["affordability"]] * city_vector[idx["affordability"]],
            preference[idx["grocery_affordability"]] * city_vector[idx["grocery_affordability"]],
        )
        utility += 0.07 * min(
            preference[idx["jobs"]] * city_vector[idx["jobs"]],
            preference[idx["internet"]] * city_vector[idx["internet"]],
        )
        utility += 0.06 * min(
            preference[idx["nature"]] * city_vector[idx["nature"]],
            preference[idx["sunset_scenery"]] * city_vector[idx["sunset_scenery"]],
        )
        utility += rng.normal(0, 0.025)

        x_rows.append(np.concatenate([preference, city_vector, interaction]))
        like_probability = 1.0 / (1.0 + math.exp(-(utility - 0.73) / 0.055))
        y_rows.append(int(rng.random() < like_probability))

    return np.vstack(x_rows), np.asarray(y_rows, dtype=int)


@st.cache_resource
def train_demo_dnn(data_signature: str, _df: pd.DataFrame) -> Pipeline:
    del data_signature
    x_train, y_train = make_dnn_training_data(_df)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "dnn",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.002,
                    max_iter=220,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def dnn_like_probabilities(model: Pipeline, df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    preference = np.array([weights[feature] for feature in FEATURE_COLUMNS], dtype=float)
    city_features = df[FEATURE_COLUMNS].to_numpy(dtype=float) / 100.0
    x = np.hstack(
        [
            np.repeat(preference.reshape(1, -1), len(df), axis=0),
            city_features,
            city_features * preference,
        ]
    )
    return model.predict_proba(x)[:, 1]


def rank_cities(
    df: pd.DataFrame,
    query: str,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    text_weights, matches = parse_preference_weights(query)
    weights = combine_weights(text_weights, slider_weights)

    documents = tuple(city_document(row) for _, row in df.iterrows())
    vectorizer, city_matrix = build_text_index(documents)
    query_vector = vectorizer.transform([query or "balanced affordable safe city"])
    text_similarity = cosine_similarity(query_vector, city_matrix).ravel()

    feature_matrix = df[FEATURE_COLUMNS].to_numpy(dtype=float) / 100.0
    weight_vector = np.array([weights[column] for column in FEATURE_COLUMNS], dtype=float)
    preference_score = feature_matrix @ weight_vector

    signature = str(hash(tuple(np.round(feature_matrix.ravel(), 4))))
    dnn_model = train_demo_dnn(signature, df)
    dnn_score = dnn_like_probabilities(dnn_model, df, weights)
    posterior = np.array([bandit_posterior(city, feedback_state) for city in df["city"]], dtype=float)

    final_score = 0.68 * preference_score + 0.14 * text_similarity + 0.13 * dnn_score + 0.05 * posterior

    ranked = df.copy()
    ranked["preference_score"] = preference_score * 100
    ranked["text_similarity"] = text_similarity * 100
    ranked["dnn_like_probability"] = dnn_score * 100
    ranked["feedback_posterior"] = posterior * 100
    ranked["match_score"] = final_score * 100
    ranked = ranked.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    return ranked, weights, matches


def top_reasons(row: pd.Series, weights: Dict[str, float], n: int = 4) -> List[str]:
    contributions = []
    for feature in FEATURE_COLUMNS:
        contributions.append((weights.get(feature, 0.0) * float(row[feature]), feature, float(row[feature])))
    contributions.sort(reverse=True)
    return [f"{FEATURE_LABELS[feature]}: {value:.0f}/100" for _, feature, value in contributions[:n]]


def cluster_cities(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    output["cluster_id"] = model.fit_predict(df[FEATURE_COLUMNS])
    summaries = output.groupby("cluster_id")[FEATURE_COLUMNS].mean()
    labels: Dict[int, str] = {}
    for cluster_id, values in summaries.iterrows():
        if values["jobs"] + values["healthcare"] + values["education"] >= 230:
            label = "Career and services hub"
        elif values["heritage"] + values["entertainment"] >= 145:
            label = "Culture and tourism hub"
        elif values["affordability"] + values["quietness"] >= 155:
            label = "Affordable quiet city"
        else:
            label = "Balanced regional center"
        labels[int(cluster_id)] = label
    output["city_archetype"] = output["cluster_id"].map(labels)
    return output


def build_map(df: pd.DataFrame, ranked: pd.DataFrame | None = None) -> folium.Map:
    map_object = folium.Map(location=[41.2, 64.6], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    ranked_lookup = {}
    if ranked is not None:
        ranked_lookup = {
            city: (index + 1, float(score))
            for index, (city, score) in enumerate(zip(ranked["city"], ranked["match_score"]))
        }

    for _, row in df.iterrows():
        city = str(row["city"])
        rank_text = ""
        icon_color = "darkgreen"
        if city in ranked_lookup:
            rank, score = ranked_lookup[city]
            rank_text = f"<br><b>Recommendation rank:</b> #{rank}<br><b>Match:</b> {score:.1f}/100"
            icon_color = "green" if rank == 1 else "cadetblue"
        popup = folium.Popup(
            f"<b>{city}</b><br>{row['region']}<br>{row['description']}{rank_text}",
            max_width=330,
        )
        folium.Marker(
            location=[float(row["lat"]), float(row["lon"])],
            tooltip=city,
            popup=popup,
            icon=folium.Icon(color=icon_color, icon="home"),
        ).add_to(map_object)
    return map_object


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_weather(lat: float, lon: float) -> Dict[str, object]:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,sunset",
            "forecast_days": 3,
            "timezone": "auto",
        },
        timeout=7,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wikipedia_summary(city: str) -> Dict[str, object]:
    response = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(city)}",
        timeout=7,
        headers={"User-Agent": "AkbarxonLivingAdvisorPrototype/2.0"},
    )
    response.raise_for_status()
    return response.json()


def weather_code_label(code: int | float | None) -> str:
    if code is None:
        return "Unknown"
    code = int(code)
    if code == 0:
        return "Clear sky"
    if code in {1, 2, 3}:
        return "Partly cloudy"
    if code in {45, 48}:
        return "Fog"
    if 51 <= code <= 67:
        return "Rain or drizzle"
    if 71 <= code <= 77:
        return "Snow"
    if 80 <= code <= 82:
        return "Rain showers"
    if code >= 95:
        return "Thunderstorm"
    return "Mixed conditions"


def radar_figure(row: pd.Series, features: List[str]) -> go.Figure:
    values = [float(row[feature]) for feature in features]
    labels = [FEATURE_LABELS[feature] for feature in features]
    fig = go.Figure(
        data=[go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", name=str(row["city"]))]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=28, r=28, t=28, b=28),
        height=420,
    )
    return fig


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #fcfcfc;
          --fg: #333333;
          --muted: #7c7c7c;
          --primary: #8ea89b;
          --primary-dark: #536f61;
          --accent: #eef5f1;
          --border: #e9e9e9;
          --card: #ffffff;
        }
        html {scroll-behavior: smooth;}
        .stApp {background: var(--bg); color: var(--fg);}

        /*
         * Streamlit Community Cloud keeps its native toolbar fixed at the top.
         * The previous 1rem top padding pulled the custom brand/navigation under
         * that toolbar. Reserve a safe top zone for the main canvas instead.
         */
        .block-container {
          max-width: 1280px;
          padding-top: 4.75rem !important;
          padding-bottom: 4rem;
        }
        [data-testid="stMainBlockContainer"] {
          max-width: 1280px;
          padding-top: 4.75rem !important;
          padding-bottom: 4rem;
        }
        h1, h2, h3, h4 {letter-spacing: -0.02em;}
        [data-testid="stSidebar"] {background: #f9faf9; border-right: 1px solid var(--border);}
        [data-testid="stSidebar"] .block-container {padding-top: 1rem !important;}
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top: .35rem;}

        .top-brand {
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:1rem;
          min-height:42px;
          padding:.35rem .15rem 1rem .15rem;
          position:relative;
          z-index:1;
        }
        .brand-left {display:flex; align-items:center; gap:.7rem;}
        .brand-mark {width:34px; height:34px; border-radius:50%; display:grid; place-items:center; border:1px solid var(--border); background:white; font-size:18px;}
        .brand-name {font-size:1.02rem; font-weight:500; letter-spacing:.01em;}
        .team-mini {font-size:.78rem; color:var(--muted); text-align:right; line-height:1.45;}

        .hero-shell {position:relative; min-height:565px; border-radius:18px; overflow:hidden; margin: .4rem 0 4rem 0; box-shadow: 0 12px 38px rgba(0,0,0,.08);}
        .hero-shell img {width:100%; height:565px; object-fit:cover; display:block;}
        .hero-overlay {position:absolute; inset:0; background:linear-gradient(180deg, rgba(0,0,0,.05) 0%, rgba(0,0,0,.42) 100%);}
        .hero-copy {position:absolute; left:42px; bottom:54px; color:white; max-width:560px;}
        .hero-eyebrow {font-size:.74rem; text-transform:uppercase; letter-spacing:.18em; margin-bottom:1rem; opacity:.92;}
        .hero-title {font-size:clamp(2.45rem,5vw,4.7rem); font-weight:300; line-height:.96; margin:0 0 1.2rem 0;}
        .hero-sub {font-size:1rem; line-height:1.65; max-width:520px; opacity:.94; margin-bottom:1.4rem;}
        .hero-pill {display:inline-block; background:#fff; color:#303030; padding:.8rem 1.15rem; border-radius:999px; font-size:.82rem; text-decoration:none;}
        .hero-bars {position:absolute; bottom:22px; left:42px; right:42px; display:flex; gap:8px;}
        .hero-bars span {height:2px; flex:1; background:rgba(255,255,255,.35);}
        .hero-bars span:first-child {background:white;}

        .section-head {text-align:center; margin: 1.5rem 0 2.5rem 0;}
        .eyebrow {font-size:.7rem; text-transform:uppercase; letter-spacing:.16em; color:var(--muted); margin-bottom:.7rem;}
        .section-title {font-size:2rem; font-weight:300; margin:0 0 .7rem 0;}
        .section-sub {font-size:.93rem; color:var(--muted); max-width:620px; margin:0 auto; line-height:1.6;}

        .city-card-html {overflow:hidden; border:1px solid var(--border); background:var(--card); border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.055); min-height:100%;}
        .city-card-html img {width:100%; height:205px; object-fit:cover; display:block;}
        .city-card-body {padding:1.25rem 1.25rem 1.35rem 1.25rem;}
        .city-card-top {display:flex; justify-content:space-between; gap:1rem; align-items:flex-start;}
        .city-card-title {font-size:1.05rem; font-weight:500; margin:0;}
        .city-card-score {font-size:.76rem; padding:.3rem .55rem; border-radius:999px; background:var(--accent); color:var(--primary-dark); white-space:nowrap;}
        .city-region {font-size:.78rem; color:var(--muted); margin:.3rem 0 .85rem 0;}
        .city-desc {font-size:.86rem; color:#5e5e5e; line-height:1.55; min-height:64px;}
        .chips {display:flex; flex-wrap:wrap; gap:6px; margin-top:.8rem;}
        .chip {font-size:.64rem; text-transform:uppercase; letter-spacing:.08em; background:var(--accent); padding:.3rem .45rem; border-radius:4px; color:#4f655a;}

        .experience-row {display:flex; align-items:center; gap:1rem; padding:1.2rem 1.25rem; border:1px solid rgba(255,255,255,.8); background:rgba(255,255,255,.72); border-radius:10px; margin:.75rem 0; box-shadow:0 2px 12px rgba(0,0,0,.035); transition:.25s ease;}
        .experience-row:hover {transform:translateY(-3px); box-shadow:0 10px 25px rgba(0,0,0,.07);}
        .experience-icon {width:46px; height:46px; border-radius:50%; display:grid; place-items:center; background:var(--accent); font-size:21px; flex:0 0 auto;}
        .experience-title {font-size:.92rem; font-weight:500; margin-bottom:.25rem;}
        .experience-desc {font-size:.82rem; color:var(--muted); line-height:1.5;}

        .advisor-card {border:1px solid var(--border); border-radius:14px; background:white; padding:1.5rem; box-shadow:0 8px 26px rgba(0,0,0,.045);}
        .result-card {border:1px solid var(--border); border-radius:12px; background:white; padding:1.25rem; margin:.75rem 0;}
        .result-rank {font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted);}
        .result-title {font-size:1.3rem; font-weight:400; margin:.3rem 0;}
        .score-badge {display:inline-block; background:var(--accent); color:var(--primary-dark); border-radius:999px; padding:.35rem .65rem; font-size:.78rem; font-weight:600;}
        .result-note {font-size:.78rem; color:var(--muted); line-height:1.55;}

        .team-card {padding:1.05rem 1rem; border:1px solid var(--border); border-radius:12px; background:white; margin-bottom:1rem;}
        .team-title {font-size:.85rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:.65rem;}
        .team-name {font-size:.9rem; line-height:1.65;}
        .prototype-note {font-size:.8rem; color:#666; line-height:1.55; padding:.85rem; background:#f1f6f3; border-radius:10px; border:1px solid #e3eee8;}

        .ai-card {padding:1.3rem; border:1px solid var(--border); border-radius:12px; background:white; min-height:180px;}
        .ai-num {font-size:.7rem; letter-spacing:.14em; color:var(--muted); text-transform:uppercase;}
        .ai-title {font-size:1rem; font-weight:500; margin:.45rem 0;}
        .ai-desc {font-size:.84rem; line-height:1.58; color:#6b6b6b;}

        .footer-shell {background:#2f302f; color:#f6f6f6; border-radius:14px; padding:2rem 2.2rem; margin-top:4rem;}
        .footer-brand {font-size:1rem; font-weight:500; margin-bottom:.5rem;}
        .footer-copy {font-size:.8rem; color:rgba(255,255,255,.68); line-height:1.6;}
        .footer-rule {border-top:1px solid rgba(255,255,255,.16); margin:1.4rem 0 1rem 0;}

        div[data-testid="stRadio"] > div {gap:.15rem;}
        div[data-testid="stRadio"] label {border-radius:999px; padding:.18rem .55rem;}
        div[data-testid="stButton"] button {border-radius:999px; font-weight:500;}
        div[data-testid="stFormSubmitButton"] button {border-radius:999px;}
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {border-radius:10px;}

        @media (max-width: 800px) {
          .hero-shell, .hero-shell img {min-height:470px; height:470px;}
          .hero-copy {left:24px; right:24px; bottom:46px;}
          .hero-bars {left:24px; right:24px;}
          .team-mini {display:none;}
          .hero-title {font-size:2.7rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_brand() -> None:
    st.markdown(
        """
        <div class="top-brand">
          <div class="brand-left">
            <div class="brand-mark">🧭</div>
            <div class="brand-name">Akbarxon AI Living Advisor</div>
          </div>
          <div class="team-mini">Author: Akbarxon Nasirov<br>Mentor: Dr. Qingyang Xiao</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    hero = image_data_uri("hero-camping.jpg")
    st.markdown(
        f"""
        <div class="hero-shell">
          <img src="{hero}" alt="Lifestyle landscape design visual">
          <div class="hero-overlay"></div>
          <div class="hero-copy">
            <div class="hero-eyebrow">AI-powered city discovery · Uzbekistan</div>
            <div class="hero-title">Find a place<br>that fits your life</div>
            <div class="hero-sub">Describe the lifestyle you want. The advisor translates your words into priorities, ranks Uzbekistan cities, maps the results, and learns from feedback.</div>
            <a class="hero-pill" href="#advisor">Start Exploring &nbsp;→</a>
          </div>
          <div class="hero-bars"><span></span><span></span><span></span><span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_head(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
          <div class="eyebrow">{eyebrow}</div>
          <div class="section-title">{title}</div>
          <div class="section-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_city_card(row: pd.Series, image_file: str, rank: int | None = None) -> None:
    image = image_data_uri(image_file)
    tags = [item.strip() for item in str(row.get("tags", "")).split(",") if item.strip()][:3]
    chips = "".join(f'<span class="chip">{tag}</span>' for tag in tags)
    score = float(row.get("match_score", 0.0))
    score_html = f'<span class="city-card-score">{score:.1f} match</span>' if score else ""
    rank_text = f"#{rank} · " if rank else ""
    st.markdown(
        f"""
        <div class="city-card-html">
          <img src="{image}" alt="Lifestyle design visual">
          <div class="city-card-body">
            <div class="city-card-top">
              <div><div class="city-card-title">{rank_text}{row['city']}</div></div>
              {score_html}
            </div>
            <div class="city-region">📍 {row['region']}</div>
            <div class="city-desc">{row['description']}</div>
            <div class="chips">{chips}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_experience_rows() -> None:
    items = [
        ("🧠", "Natural-language understanding", "Turn everyday phrases such as 'cheap groceries' or 'great sunsets' into measurable lifestyle priorities."),
        ("🗺️", "Map-first exploration", "See candidate cities geographically and inspect how recommendations relate across Uzbekistan."),
        ("📊", "Transparent scoring", "Combine preference features, text similarity, neural-network estimates, and feedback into an explainable score."),
        ("👍", "Feedback learning", "Likes and dislikes update a lightweight bandit signal so the prototype can adapt over time."),
    ]
    for icon, title, desc in items:
        st.markdown(
            f"""
            <div class="experience-row">
              <div class="experience-icon">{icon}</div>
              <div><div class="experience-title">{title}</div><div class="experience-desc">{desc}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feedback_buttons(city: str, feedback_state: Dict[str, Dict[str, int]], key_prefix: str) -> None:
    left, right, stats = st.columns([1, 1, 2.4])
    if left.button("👍 Like", key=f"{key_prefix}_like_{city}", use_container_width=True):
        feedback_state[city]["likes"] += 1
        save_feedback_state(feedback_state)
        st.toast(f"Feedback saved for {city}")
        st.rerun()
    if right.button("👎 Not for me", key=f"{key_prefix}_dislike_{city}", use_container_width=True):
        feedback_state[city]["dislikes"] += 1
        save_feedback_state(feedback_state)
        st.toast(f"Feedback saved for {city}")
        st.rerun()
    values = feedback_state[city]
    stats.caption(
        f"Prototype feedback: {values['likes']} likes · {values['dislikes']} dislikes · "
        f"bandit confidence {bandit_posterior(city, feedback_state):.2f}"
    )


def ensure_default_results(
    df: pd.DataFrame,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> None:
    if "ranked_results" not in st.session_state:
        query = "I want an affordable city with cheap groceries, beautiful sunsets, and good entertainment."
        ranked, weights, matches = rank_cities(df, query, slider_weights, feedback_state, top_n)
        st.session_state["ranked_results"] = ranked
        st.session_state["active_weights"] = weights
        st.session_state["query_matches"] = matches
        st.session_state["active_query"] = query


def render_home(
    df: pd.DataFrame,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> None:
    ensure_default_results(df, slider_weights, feedback_state, top_n)
    render_hero()

    render_section_head(
        "Featured matches",
        "Lifestyle ideas, ranked for you",
        "A first look at cities selected by the same AI engine used in the full recommendation workspace.",
    )
    ranked = st.session_state["ranked_results"].head(3)
    cols = st.columns(3)
    for idx, (col, (_, row)) in enumerate(zip(cols, ranked.iterrows())):
        with col:
            render_city_card(row, CARD_IMAGES[idx % len(CARD_IMAGES)], idx + 1)

    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
    render_section_head(
        "The experience",
        "Minimal design, useful intelligence",
        "The uploaded UI's calm, image-led design language is preserved while the interaction is rebuilt around city discovery.",
    )
    render_experience_rows()

    st.markdown('<div id="advisor"></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    render_section_head(
        "AI advisor",
        "Describe your ideal place",
        "Use natural language now; detailed controls are available in the sidebar and the Recommend workspace.",
    )
    with st.form("home_quick_advisor"):
        quick_query = st.text_area(
            "Lifestyle request",
            value=st.session_state.get("active_query", "I want an affordable city with good food prices and beautiful scenery."),
            height=115,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Find my cities", type="primary", use_container_width=True)
    if submitted:
        ranked, weights, matches = rank_cities(df, quick_query, slider_weights, feedback_state, top_n)
        st.session_state["ranked_results"] = ranked
        st.session_state["active_weights"] = weights
        st.session_state["query_matches"] = matches
        st.session_state["active_query"] = quick_query
        st.success("Recommendations updated. Open the Recommend or Map & Compare workspace for full details.")
        st.dataframe(ranked[["city", "region", "match_score"]].head(5), hide_index=True, use_container_width=True)


def render_recommend(
    df: pd.DataFrame,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> None:
    render_section_head(
        "Personalized advisor",
        "Tell the AI what matters",
        "Text is the main signal. The ranking engine combines interpretable lifestyle features, NLP similarity, a neural model, and feedback.",
    )

    query = st.text_area(
        "What kind of place are you looking for?",
        value=st.session_state.get(
            "active_query",
            "I want an affordable city with cheap groceries, beautiful sunsets, and good entertainment.",
        ),
        height=120,
        help="Examples: family-friendly and safe; best for remote work; historic and walkable; quiet retirement city.",
    )
    if st.button("Find my best cities", type="primary", use_container_width=True):
        ranked, weights, matches = rank_cities(df, query, slider_weights, feedback_state, top_n)
        st.session_state["ranked_results"] = ranked
        st.session_state["active_weights"] = weights
        st.session_state["query_matches"] = matches
        st.session_state["active_query"] = query

    ensure_default_results(df, slider_weights, feedback_state, top_n)
    ranked = st.session_state["ranked_results"]
    weights = st.session_state["active_weights"]
    matches = st.session_state["query_matches"]

    with st.expander("How the AI interpreted your request", expanded=False):
        st.write("Detected signals:")
        for match in matches:
            st.write(f"- {match}")
        weight_table = pd.DataFrame(
            {
                "Preference": [FEATURE_LABELS[key] for key in FEATURE_COLUMNS],
                "Weight (%)": [round(weights[key] * 100, 1) for key in FEATURE_COLUMNS],
            }
        ).sort_values("Weight (%)", ascending=False)
        st.dataframe(weight_table, hide_index=True, use_container_width=True)

    st.markdown("### Top matches")
    for index, row in ranked.iterrows():
        reasons = top_reasons(row, weights)
        image = CARD_IMAGES[index % len(CARD_IMAGES)]
        left, right = st.columns([1, 2.2])
        with left:
            st.image(ASSET_DIR / image, use_container_width=True)
        with right:
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-rank">Recommendation #{index + 1}</div>
                  <div class="result-title">{row['city']} &nbsp; <span class="score-badge">{row['match_score']:.1f}/100 match</span></div>
                  <div class="city-region">{row['region']}</div>
                  <p>{row['description']}</p>
                  <p><b>Why it fits:</b> {' · '.join(reasons)}</p>
                  <div class="result-note">Hybrid components: preference {row['preference_score']:.1f}, text {row['text_similarity']:.1f}, neural model {row['dnn_like_probability']:.1f}, feedback {row['feedback_posterior']:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_feedback_buttons(str(row["city"]), feedback_state, f"rec_{index}")

    st.markdown("### Optional live context")
    selected_live_city = st.selectbox("City", ranked["city"].tolist(), key="live_city")
    if st.button("Fetch weather + city summary"):
        city_row = df.loc[df["city"] == selected_live_city].iloc[0]
        with st.spinner("Retrieving public context..."):
            try:
                wiki = fetch_wikipedia_summary(selected_live_city)
                weather = fetch_live_weather(float(city_row["lat"]), float(city_row["lon"]))
                current = weather.get("current", {})
                c1, c2, c3 = st.columns(3)
                c1.metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
                c2.metric("Feels like", f"{current.get('apparent_temperature', '—')} °C")
                c3.metric("Conditions", weather_code_label(current.get("weather_code")))
                st.write(wiki.get("extract", "No summary was returned."))
                source_url = wiki.get("content_urls", {}).get("desktop", {}).get("page")
                if source_url:
                    st.link_button("Open source article", source_url)
            except requests.RequestException as exc:
                st.warning(f"Live data is temporarily unavailable: {exc}")


def render_map_compare(df: pd.DataFrame, slider_weights, feedback_state, top_n) -> None:
    ensure_default_results(df, slider_weights, feedback_state, top_n)
    ranked = st.session_state["ranked_results"]
    render_section_head(
        "Map & compare",
        "See the recommendations geographically",
        "The map uses OpenStreetMap tiles and overlays the prototype Uzbekistan city profiles and current recommendation ranking.",
    )
    st_folium(build_map(df, ranked), width=None, height=590, returned_objects=[])

    st.markdown("### Recommendation score comparison")
    chart_df = ranked[["city", "match_score"]].sort_values("match_score")
    fig = px.bar(chart_df, x="match_score", y="city", orientation="h", range_x=[0, 100])
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Match score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    compare_names = st.multiselect(
        "Choose up to three cities for detailed comparison",
        df["city"].tolist(),
        default=ranked["city"].head(2).tolist(),
        max_selections=3,
    )
    comparison_features = [
        "affordability", "grocery_affordability", "entertainment", "sunset_scenery",
        "safety", "jobs", "internet", "healthcare", "heritage", "nature",
    ]
    columns = st.columns(max(1, len(compare_names)))
    for column, city in zip(columns, compare_names):
        row = df.loc[df["city"] == city].iloc[0]
        with column:
            st.plotly_chart(radar_figure(row, comparison_features), use_container_width=True)
            st.caption(row["description"])


def render_cities(df: pd.DataFrame, feedback_state: Dict[str, Dict[str, int]]) -> None:
    render_section_head(
        "City explorer",
        "Browse every prototype profile",
        "Cards mirror the uploaded location-card UI. The photos are lifestyle design assets from the uploaded UI, not documentary photos of the named cities.",
    )

    search = st.text_input("Filter cities", placeholder="Search by city, region, or tag")
    filtered = df.copy()
    if search.strip():
        needle = search.lower().strip()
        mask = filtered.apply(
            lambda row: needle in " ".join([str(row.get("city", "")), str(row.get("region", "")), str(row.get("tags", ""))]).lower(),
            axis=1,
        )
        filtered = filtered.loc[mask]

    for start in range(0, len(filtered), 3):
        cols = st.columns(3)
        batch = filtered.iloc[start : start + 3]
        for offset, (col, (_, row)) in enumerate(zip(cols, batch.iterrows())):
            with col:
                render_city_card(row, CARD_IMAGES[(start + offset) % len(CARD_IMAGES)])

    st.markdown("### Detailed city profile")
    city = st.selectbox("Select a city", df["city"].tolist(), key="city_explorer")
    row = df.loc[df["city"] == city].iloc[0]
    left, right = st.columns([1.15, 1])
    with left:
        st.image(ASSET_DIR / DETAIL_IMAGES[df.index[df["city"] == city][0] % len(DETAIL_IMAGES)], use_container_width=True)
        st.markdown(f"## {row['city']}")
        st.write(f"**Region:** {row['region']}")
        st.write(row["description"])
        st.write(f"**Tags:** {row['tags']}")
        score_table = pd.DataFrame(
            {
                "Dimension": [FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS],
                "Prototype score": [float(row[feature]) for feature in FEATURE_COLUMNS],
            }
        ).sort_values("Prototype score", ascending=False)
        st.dataframe(score_table, hide_index=True, use_container_width=True)
    with right:
        st.plotly_chart(
            radar_figure(row, ["affordability", "entertainment", "sunset_scenery", "safety", "jobs", "internet", "healthcare", "heritage", "nature"]),
            use_container_width=True,
        )
    render_feedback_buttons(city, feedback_state, "explorer")


def render_ai_lab(df: pd.DataFrame) -> None:
    render_section_head(
        "AI laboratory",
        "How the recommendation brain works",
        "The prototype intentionally combines explainable rules with machine learning, a neural-network demonstration, and feedback learning.",
    )
    c1, c2, c3, c4 = st.columns(4)
    items = [
        ("01", "Preference parser", "Phrase rules convert natural-language requests into transparent feature weights."),
        ("02", "Machine learning", "K-means groups city profiles into lifestyle archetypes from their feature vectors."),
        ("03", "Deep neural network", "A 64-32-16 MLP estimates user-city like probability using synthetic interactions for the demo."),
        ("04", "Feedback learning", "A Beta-Bernoulli bandit turns likes and dislikes into a small adaptive ranking signal."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4], items):
        with col:
            st.markdown(
                f'<div class="ai-card"><div class="ai-num">{num}</div><div class="ai-title">{title}</div><div class="ai-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Data-driven city archetypes")
    clustered = cluster_cities(df)
    st.dataframe(clustered[["city", "region", "city_archetype"] + FEATURE_COLUMNS], hide_index=True, use_container_width=True)

    st.markdown("### Project team")
    st.markdown(
        """
        <div class="advisor-card">
          <b>Author:</b> Akbarxon Nasirov<br>
          <b>Mentor:</b> Dr. Qingyang Xiao<br><br>
          This project is designed as an educational AI portfolio platform that demonstrates data engineering, machine learning, neural networks, feedback learning, geospatial visualization, and Streamlit deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "For production, replace illustrative scores and synthetic training data with licensed, dated, auditable sources. "
        "Use a real database for feedback, user accounts, privacy controls, source citations, and model monitoring."
    )


def render_about() -> None:
    render_section_head(
        "About the project",
        "Living decisions, made easier to explore",
        "A portfolio-ready prototype that turns broad lifestyle preferences into a structured, visual city-comparison workflow.",
    )
    image = image_data_uri("detail-forest-2.jpg")
    st.markdown(
        f"""
        <div class="hero-shell" style="min-height:430px;">
          <img src="{image}" alt="Lifestyle design visual" style="height:430px;">
          <div class="hero-overlay"></div>
          <div class="hero-copy" style="bottom:38px;">
            <div class="hero-eyebrow">Design + data + AI</div>
            <div class="hero-title" style="font-size:3rem;">Explore before<br>you relocate</div>
            <div class="hero-sub">The experience is inspired by the uploaded UI package and rebuilt in Streamlit so it can deploy directly from GitHub to Streamlit Community Cloud.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("### What is included")
        st.markdown(
            """
            - Natural-language city preference search
            - Uzbekistan map visualization
            - Multi-city scoring and radar comparison
            - ML clustering and neural-network demonstration
            - Like/dislike feedback learning
            - Optional public weather and city context
            - Responsive Streamlit UI styled after the uploaded React/Tailwind design
            """
        )
    with right:
        st.markdown("### UI source preservation")
        st.write(
            "The GitHub repository also contains the uploaded React/Tailwind UI project under `ui_source/`. "
            "Its `.env` file is intentionally excluded so secrets or environment-specific values are not published."
        )
        st.info(
            "The bundled photos are UI design assets. They are used as lifestyle visuals and should not be treated as verified photographs of specific Uzbekistan cities."
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-shell">
          <div class="footer-brand">🧭 Akbarxon AI Living Advisor</div>
          <div class="footer-copy">AI-assisted city discovery for Uzbekistan · Streamlit portfolio prototype<br>Author: Akbarxon Nasirov · Mentor: Dr. Qingyang Xiao</div>
          <div class="footer-rule"></div>
          <div class="footer-copy">Prototype city scores are illustrative, not official statistics. Verify housing, employment, safety, healthcare, visa, legal, and financial information before making relocation decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Akbarxon AI Living Advisor",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    df = load_city_data()
    feedback_state = load_feedback_state(df["city"])

    with st.sidebar:
        st.markdown(
            """
            <div class="team-card">
              <div class="team-title">Akbarxon AI Living Advisor</div>
              <div class="team-name"><b>Author:</b> Akbarxon Nasirov<br><b>Mentor:</b> Dr. Qingyang Xiao</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Your priorities")
        st.caption("Text is the main input. Sliders let you emphasize or correct specific priorities.")
        top_n = st.slider("Number of recommendations", 3, 8, 5)
        with st.expander("Advanced preference sliders", expanded=False):
            slider_weights = {
                feature: float(st.slider(FEATURE_LABELS[feature], 0, 5, 0, key=f"slider_{feature}"))
                for feature in FEATURE_COLUMNS
            }
        st.divider()
        st.markdown(
            """
            <div class="prototype-note"><b>Prototype note</b><br>City scores are illustrative, not official statistics. Verify housing, employment, safety, healthcare, visa, and legal information before relocating.</div>
            """,
            unsafe_allow_html=True,
        )

    render_top_brand()
    page = st.radio(
        "Navigation",
        ["Home", "Recommend", "Map & Compare", "Cities", "AI Lab", "About"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    if page == "Home":
        render_home(df, slider_weights, feedback_state, top_n)
    elif page == "Recommend":
        render_recommend(df, slider_weights, feedback_state, top_n)
    elif page == "Map & Compare":
        render_map_compare(df, slider_weights, feedback_state, top_n)
    elif page == "Cities":
        render_cities(df, feedback_state)
    elif page == "AI Lab":
        render_ai_lab(df)
    else:
        render_about()

    render_footer()


if __name__ == "__main__":
    main()
