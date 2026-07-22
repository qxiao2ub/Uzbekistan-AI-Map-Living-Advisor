from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import quote

import folium
import joblib
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
FEEDBACK_PATH = APP_DIR / "feedback_state.json"
MODEL_PATH = APP_DIR / "demo_preference_dnn.joblib"

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

CLUSTER_NAMES = {
    0: "Balanced regional center",
    1: "Culture and tourism hub",
    2: "Affordable quiet city",
    3: "Career and services hub",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1250px;}
        .hero {
            padding: 1.35rem 1.5rem; border-radius: 22px;
            background: linear-gradient(135deg, #0f766e 0%, #0369a1 55%, #4338ca 100%);
            color: white; margin-bottom: 1rem;
            box-shadow: 0 12px 32px rgba(3, 105, 161, 0.18);
        }
        .hero h1 {margin: 0 0 .25rem 0; font-size: 2.2rem;}
        .hero p {margin: 0; opacity: .94; font-size: 1.02rem;}
        .city-card {
            border: 1px solid rgba(100,116,139,.22); border-radius: 18px;
            padding: 1rem 1.05rem; margin: .35rem 0 .85rem 0;
            background: rgba(255,255,255,.02);
        }
        .score-pill {
            display: inline-block; padding: .22rem .62rem; border-radius: 999px;
            background: rgba(14,165,233,.12); font-weight: 700;
        }
        .small-note {font-size: .86rem; opacity: .78;}
        .sidebar-brand {
            padding: .85rem .8rem .78rem .8rem;
            border: 1px solid rgba(14, 116, 144, .22);
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(15,118,110,.10), rgba(67,56,202,.08));
            margin: .2rem 0 .65rem 0;
        }
        .sidebar-brand-title {
            font-size: 1.08rem; font-weight: 800; line-height: 1.25;
            margin-bottom: .55rem;
        }
        .sidebar-team {font-size: .92rem; line-height: 1.55;}
        .project-team-card {
            border: 1px solid rgba(67,56,202,.20);
            border-radius: 18px; padding: 1rem 1.15rem; margin: .2rem 0 1rem 0;
            background: linear-gradient(135deg, rgba(15,118,110,.08), rgba(3,105,161,.08), rgba(67,56,202,.08));
        }
        .project-team-card h3 {margin: 0 0 .55rem 0;}
        .project-team-card p {margin: 0; line-height: 1.65;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_city_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "cities_uzbekistan.csv is missing. Run the Colab notebook export cells first."
        )
    df = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(50).clip(0, 100)
    return df


def city_document(row: pd.Series) -> str:
    feature_words = " ".join(
        FEATURE_LABELS[column]
        for column in FEATURE_COLUMNS
        if float(row[column]) >= 72
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
                increment = 1.4 if " " in phrase else 1.0
                weights[feature] += increment
                matches.append(f"{FEATURE_LABELS[feature]} ← '{phrase}'")

    # Small intent rules for common compound requests.
    if "family" in text or "children" in text or "kids" in text:
        weights["safety"] += 1.1
        weights["education"] += 0.9
        weights["healthcare"] += 0.6
        matches.append("Family intent → safety, education, healthcare")
    if "retire" in text or "retirement" in text:
        weights["affordability"] += 0.8
        weights["quietness"] += 1.0
        weights["healthcare"] += 0.8
        matches.append("Retirement intent → affordability, quietness, healthcare")
    if "young professional" in text or "professional" in text:
        weights["jobs"] += 1.0
        weights["internet"] += 0.7
        weights["entertainment"] += 0.5
        matches.append("Professional intent → jobs, internet, entertainment")
    if "tourist" in text or "tourism" in text or "visit" in text:
        weights["heritage"] += 0.9
        weights["entertainment"] += 0.5
        weights["mobility"] += 0.4
        matches.append("Tourism intent → heritage, activities, transportation")

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
        matches.append("No strong keyword found → balanced default profile")
        total = sum(weights.values())

    return {key: value / total for key, value in weights.items()}, matches


def combine_weights(
    text_weights: Dict[str, float], slider_weights: Dict[str, float]
) -> Dict[str, float]:
    combined = {}
    for feature in FEATURE_COLUMNS:
        # Text remains the primary signal; sliders act as explicit corrections.
        combined[feature] = text_weights.get(feature, 0.0) + slider_weights.get(feature, 0.0) / 5.0
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
        # Some hosted deployments have read-only or ephemeral filesystems.
        pass


def bandit_posterior(city: str, state: Dict[str, Dict[str, int]]) -> float:
    values = state.get(city, {"likes": 0, "dislikes": 0})
    likes = max(0, int(values.get("likes", 0)))
    dislikes = max(0, int(values.get("dislikes", 0)))
    return (likes + 1.0) / (likes + dislikes + 2.0)


def make_dnn_training_data(df: pd.DataFrame, n_samples: int = 2400) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    features = df[FEATURE_COLUMNS].to_numpy(dtype=float) / 100.0
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for _ in range(n_samples):
        city_vector = features[rng.integers(0, len(features))]
        preference = rng.dirichlet(np.ones(len(FEATURE_COLUMNS)) * 0.75)
        interaction = preference * city_vector
        utility = float(np.dot(preference, city_vector))

        # Nonlinear lifestyle interactions create a useful neural-network demonstration.
        heritage_idx = FEATURE_COLUMNS.index("heritage")
        entertainment_idx = FEATURE_COLUMNS.index("entertainment")
        affordability_idx = FEATURE_COLUMNS.index("affordability")
        grocery_idx = FEATURE_COLUMNS.index("grocery_affordability")
        jobs_idx = FEATURE_COLUMNS.index("jobs")
        internet_idx = FEATURE_COLUMNS.index("internet")
        nature_idx = FEATURE_COLUMNS.index("nature")
        sunset_idx = FEATURE_COLUMNS.index("sunset_scenery")

        utility += 0.09 * min(
            preference[heritage_idx] * city_vector[heritage_idx],
            preference[entertainment_idx] * city_vector[entertainment_idx],
        )
        utility += 0.08 * min(
            preference[affordability_idx] * city_vector[affordability_idx],
            preference[grocery_idx] * city_vector[grocery_idx],
        )
        utility += 0.07 * min(
            preference[jobs_idx] * city_vector[jobs_idx],
            preference[internet_idx] * city_vector[internet_idx],
        )
        utility += 0.06 * min(
            preference[nature_idx] * city_vector[nature_idx],
            preference[sunset_idx] * city_vector[sunset_idx],
        )
        utility += rng.normal(0, 0.025)

        x_rows.append(np.concatenate([preference, city_vector, interaction]))
        like_probability = 1.0 / (1.0 + math.exp(-(utility - 0.73) / 0.055))
        y_rows.append(int(rng.random() < like_probability))

    return np.vstack(x_rows), np.asarray(y_rows, dtype=int)


@st.cache_resource
def train_or_load_demo_dnn(data_signature: str, _df: pd.DataFrame) -> Pipeline:
    del data_signature
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

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
                    max_iter=350,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    try:
        joblib.dump(model, MODEL_PATH)
    except OSError:
        pass
    return model


def dnn_like_probabilities(
    model: Pipeline, df: pd.DataFrame, weights: Dict[str, float]
) -> np.ndarray:
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
    dnn_model = train_or_load_demo_dnn(signature, df)
    dnn_score = dnn_like_probabilities(dnn_model, df, weights)

    posterior = np.array(
        [bandit_posterior(city, feedback_state) for city in df["city"]], dtype=float
    )

    # Transparent hybrid score: explicit preferences dominate.
    final_score = (
        0.68 * preference_score
        + 0.14 * text_similarity
        + 0.13 * dnn_score
        + 0.05 * posterior
    )

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
        contribution = weights.get(feature, 0.0) * float(row[feature])
        contributions.append((contribution, feature, float(row[feature])))
    contributions.sort(reverse=True)
    reasons = []
    for _, feature, value in contributions[:n]:
        reasons.append(f"{FEATURE_LABELS[feature]}: {value:.0f}/100")
    return reasons


def build_map(df: pd.DataFrame, ranked: pd.DataFrame | None = None) -> folium.Map:
    map_object = folium.Map(
        location=[41.2, 64.6],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    ranked_lookup = {}
    if ranked is not None:
        ranked_lookup = {
            city: (index + 1, float(score))
            for index, (city, score) in enumerate(zip(ranked["city"], ranked["match_score"]))
        }

    for _, row in df.iterrows():
        city = str(row["city"])
        rank_text = ""
        icon_color = "blue"
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
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,sunset",
        "forecast_days": 3,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=7)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wikipedia_summary(city: str) -> Dict[str, object]:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(city)}"
    response = requests.get(url, timeout=7, headers={"User-Agent": "LivingAdvisorPrototype/1.0"})
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
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    figure = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                name=str(row["city"]),
            )
        ]
    )
    figure.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=35, r=35, t=35, b=35),
        height=430,
    )
    return figure


def cluster_cities(df: pd.DataFrame) -> pd.DataFrame:
    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    output = df.copy()
    output["cluster_id"] = model.fit_predict(df[FEATURE_COLUMNS])

    # Assign human-readable labels by cluster characteristics rather than numeric ID.
    summaries = output.groupby("cluster_id")[FEATURE_COLUMNS].mean()
    cluster_labels: Dict[int, str] = {}
    for cluster_id, values in summaries.iterrows():
        if values["jobs"] + values["healthcare"] + values["education"] >= 230:
            label = "Career and services hub"
        elif values["heritage"] + values["entertainment"] >= 145:
            label = "Culture and tourism hub"
        elif values["affordability"] + values["quietness"] >= 155:
            label = "Affordable quiet city"
        else:
            label = "Balanced regional center"
        cluster_labels[int(cluster_id)] = label
    output["city_archetype"] = output["cluster_id"].map(cluster_labels)
    return output


def render_feedback_buttons(city: str, feedback_state: Dict[str, Dict[str, int]], key_prefix: str) -> None:
    left, right, stats = st.columns([1, 1, 2])
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

    st.markdown(
        """
        <div class="hero">
          <h1>🧭 Akbarxon AI Living Advisor</h1>
          <p>Describe the lifestyle you want, compare Uzbekistan cities, explore the map, and improve recommendations through feedback.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="sidebar-brand-title">🧭 Akbarxon AI Living Advisor</div>
              <div class="sidebar-team">
                <strong>Author:</strong> Akbarxon Nasirov<br>
                <strong>Mentor:</strong> Dr. Qingyang Xiao
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Your priorities")
        st.caption("Text is the main input. Sliders let you emphasize or correct specific priorities.")
        top_n = st.slider("Number of recommendations", 3, 8, 5)
        with st.expander("Advanced preference sliders", expanded=False):
            slider_weights = {
                feature: float(st.slider(FEATURE_LABELS[feature], 0, 5, 0, key=f"slider_{feature}"))
                for feature in FEATURE_COLUMNS
            }
        st.divider()
        st.info(
            "Prototype note: city scores are illustrative, not official statistics. "
            "Verify housing, employment, safety, healthcare, visa, and legal information before relocating."
        )

    recommend_tab, map_tab, explorer_tab, ai_tab = st.tabs(
        ["✨ Recommend", "🗺️ Map & compare", "🏙️ City explorer", "🧠 AI laboratory"]
    )

    with recommend_tab:
        query = st.text_area(
            "What kind of place are you looking for?",
            value="I want an affordable city with cheap groceries, beautiful sunsets, and good entertainment.",
            height=105,
            help="Examples: family-friendly and safe; best for remote work; historic and walkable; quiet retirement city.",
        )
        run = st.button("Find my best cities", type="primary", use_container_width=True)

        if run or "ranked_results" not in st.session_state:
            ranked, weights, matches = rank_cities(
                df, query, slider_weights, feedback_state, top_n
            )
            st.session_state["ranked_results"] = ranked
            st.session_state["active_weights"] = weights
            st.session_state["query_matches"] = matches
            st.session_state["active_query"] = query

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

        st.subheader("Top matches")
        for index, row in ranked.iterrows():
            reasons = top_reasons(row, weights)
            st.markdown(
                f"""
                <div class="city-card">
                  <h3>#{index + 1} {row['city']} <span class="score-pill">{row['match_score']:.1f}/100 match</span></h3>
                  <p><b>{row['region']}</b> · {row['description']}</p>
                  <p><b>Why it fits:</b> {' · '.join(reasons)}</p>
                  <p class="small-note">Hybrid components: preference {row['preference_score']:.1f}, text {row['text_similarity']:.1f}, neural model {row['dnn_like_probability']:.1f}, feedback {row['feedback_posterior']:.1f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_feedback_buttons(str(row["city"]), feedback_state, f"rec_{index}")

        selected_live_city = st.selectbox(
            "Optional live context",
            ranked["city"].tolist(),
            help="Uses public Wikipedia and Open-Meteo endpoints. Data may be unavailable or incomplete.",
        )
        if st.button("Fetch live context for selected city"):
            city_row = df.loc[df["city"] == selected_live_city].iloc[0]
            with st.spinner("Retrieving live context..."):
                try:
                    wiki = fetch_wikipedia_summary(selected_live_city)
                    weather = fetch_live_weather(float(city_row["lat"]), float(city_row["lon"]))
                    current = weather.get("current", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
                    col2.metric("Feels like", f"{current.get('apparent_temperature', '—')} °C")
                    col3.metric("Conditions", weather_code_label(current.get("weather_code")))
                    st.write(wiki.get("extract", "No city summary was returned."))
                    if wiki.get("content_urls", {}).get("desktop", {}).get("page"):
                        st.link_button(
                            "Open source article",
                            wiki["content_urls"]["desktop"]["page"],
                        )
                except requests.RequestException as exc:
                    st.warning(f"Live data is temporarily unavailable: {exc}")

    with map_tab:
        ranked = st.session_state.get("ranked_results")
        st.subheader("Uzbekistan city map")
        st_folium(build_map(df, ranked), width=None, height=560, returned_objects=[])

        if ranked is not None:
            st.subheader("Recommendation score comparison")
            chart_df = ranked[["city", "match_score"]].sort_values("match_score")
            figure = px.bar(
                chart_df,
                x="match_score",
                y="city",
                orientation="h",
                labels={"match_score": "Match score", "city": "City"},
                range_x=[0, 100],
            )
            figure.update_layout(height=390, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(figure, use_container_width=True)

            compare_names = st.multiselect(
                "Choose up to three cities for detailed comparison",
                df["city"].tolist(),
                default=ranked["city"].head(2).tolist(),
                max_selections=3,
            )
            comparison_features = [
                "affordability",
                "grocery_affordability",
                "entertainment",
                "sunset_scenery",
                "safety",
                "jobs",
                "internet",
                "healthcare",
                "heritage",
                "nature",
            ]
            columns = st.columns(max(1, len(compare_names)))
            for column, city in zip(columns, compare_names):
                row = df.loc[df["city"] == city].iloc[0]
                with column:
                    st.plotly_chart(radar_figure(row, comparison_features), use_container_width=True)
                    st.caption(row["description"])

    with explorer_tab:
        st.subheader("Explore every prototype city profile")
        city = st.selectbox("Select a city", df["city"].tolist(), key="city_explorer")
        row = df.loc[df["city"] == city].iloc[0]
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(f"### {row['city']}")
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
                radar_figure(
                    row,
                    [
                        "affordability",
                        "entertainment",
                        "sunset_scenery",
                        "safety",
                        "jobs",
                        "internet",
                        "healthcare",
                        "heritage",
                        "nature",
                    ],
                ),
                use_container_width=True,
            )
        render_feedback_buttons(city, feedback_state, "explorer")

    with ai_tab:
        st.markdown(
            """
            <div class="project-team-card">
              <h3>👥 Project team</h3>
              <p>
                <strong>Author:</strong> Akbarxon Nasirov<br>
                <strong>Mentor:</strong> Dr. Qingyang Xiao
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("What the prototype AI is doing")
        st.markdown(
            """
            **1. Natural-language preference parsing:** keyword and phrase rules convert a request into transparent lifestyle weights.

            **2. Text similarity:** TF–IDF and cosine similarity compare the user's request with each city profile.

            **3. Machine learning:** K-means groups cities into data-driven lifestyle archetypes.

            **4. Deep neural network demonstration:** a three-hidden-layer MLP predicts a like probability for each user–city pair. It is trained on synthetic prototype interactions until real, consented feedback is available.

            **5. Reinforcement-learning-style feedback:** a Beta-Bernoulli bandit updates each city's recommendation bonus after likes and dislikes.
            """
        )
        clustered = cluster_cities(df)
        st.dataframe(
            clustered[["city", "region", "city_archetype"] + FEATURE_COLUMNS],
            hide_index=True,
            use_container_width=True,
        )
        st.warning(
            "For production, replace illustrative scores and synthetic training data with licensed, dated, auditable sources. "
            "Use a real database for feedback, user accounts, privacy controls, source citations, and model monitoring."
        )

    st.divider()
    st.caption(
        "Author: Akbarxon Nasirov · Mentor: Dr. Qingyang Xiao · "
        "Built as an educational prototype for Akbarxon's GitHub portfolio. "
        "The app does not provide legal, immigration, housing, medical, or financial advice."
    )


if __name__ == "__main__":
    main()
