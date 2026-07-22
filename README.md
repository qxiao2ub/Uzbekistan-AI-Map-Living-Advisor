# Akbarxon AI Living Advisor for Uzbekistan

A portfolio-ready Streamlit prototype that recommends cities in Uzbekistan from a user's natural-language living preferences.

## Author / portfolio owner

Akbarxon

## Mentor

Qingyang Xiao

## Features

- Natural-language preference parsing
- TF-IDF and cosine-similarity matching
- Weighted, explainable city ranking
- K-means city archetypes
- Three-hidden-layer MLP neural-network demonstration
- Like/dislike feedback with a Beta-Bernoulli contextual-bandit-style adjustment
- Interactive Folium/OpenStreetMap city map
- Plotly comparisons and radar charts
- Optional live weather and Wikipedia city context
- GitHub and Streamlit Community Cloud deployment workflow

## Important data note

The included city scores are **illustrative prototype values**, not official measurements. Before public launch, replace them with licensed, dated, auditable data and show a source and update date for every metric.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `cities_uzbekistan.csv`, `requirements.txt`, and `README.md` to a public GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository.
3. Select `app.py` as the entry point.
4. Use Python 3.12 when the platform allows runtime selection.
5. Deploy and test the map, feedback controls, and optional live-data buttons.

## Production upgrades

- Replace local JSON feedback storage with PostgreSQL, Supabase, or Firebase.
- Add authenticated profiles and consent controls.
- Connect licensed housing, grocery, transit, healthcare, education, safety, air-quality, and employment data.
- Preserve source URLs, retrieval timestamps, and confidence values.
- Add Uzbek and Russian localization.
- Add fairness, privacy, security, and recommendation-quality monitoring.
- Never infer protected traits or use sensitive data without a lawful, consented purpose.

## Disclaimer

This is an educational prototype. It does not provide legal, immigration, housing, medical, employment, or financial advice.
