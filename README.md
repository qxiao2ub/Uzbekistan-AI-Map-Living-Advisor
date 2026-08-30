# Akbarxon AI Living Advisor: Uzbekistan

**Author:** Akbarxon Nasirov  
**Mentor:** Dr. Qingyang Xiao

An AI-based living-advisor platform that helps users explore which Uzbekistan cities may fit their lifestyle preferences. Users can describe what matters in natural language — affordability, grocery prices, scenery, entertainment, safety, careers, healthcare, internet, education, nature, transportation, and more — and receive ranked city recommendations with map-based visualization.

## New UI integration

This version ports the complete visual language of the supplied React/Tailwind UI package into a Streamlit-native interface:

- full-width image hero
- light minimalist design system
- muted green accents
- featured city cards
- image-based city detail layouts
- feature/experience rows
- pill-style buttons and navigation
- dark footer
- responsive layouts

The original UI project is also retained in `ui_source/` for reference. Its `.env` file is intentionally not included.

See [`UI_MIGRATION.md`](UI_MIGRATION.md) for the design mapping.

## AI / data features

1. **Natural-language preference parsing** turns everyday requests into transparent lifestyle weights.
2. **TF-IDF + cosine similarity** compares user language against city descriptions and tags.
3. **Machine learning** uses K-means to organize prototype cities into lifestyle archetypes.
4. **Deep neural-network demonstration** uses a 64 → 32 → 16 MLP to estimate a user-city like probability from synthetic demonstration interactions.
5. **Feedback learning** uses a Beta-Bernoulli bandit signal from likes/dislikes.
6. **Interactive mapping** uses Folium/OpenStreetMap.
7. **Optional live context** can retrieve current weather from Open-Meteo and city summaries from Wikipedia when network access is available.

## Repository structure

```text
.
├── app.py
├── cities_uzbekistan.csv
├── requirements.txt
├── README.md
├── UI_MIGRATION.md
├── .streamlit/
│   └── config.toml
├── assets/
│   └── uploaded UI image assets
├── ui_source/
│   └── original uploaded React/Tailwind source (without .env)
└── Akbarxon_AI_Living_Advisor_Uzbekistan_Updated_Colab.ipynb
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Extract this ZIP and upload all repository contents to GitHub.
2. In Streamlit Community Cloud, create a new app from that GitHub repository.
3. Choose `app.py` as the main file.
4. Deploy. Streamlit will install dependencies from `requirements.txt`.

No paid API key is required for the core prototype.

## Prototype / responsible-use note

The city scores in `cities_uzbekistan.csv` are illustrative prototype values, not official statistics. Before public or production use, replace them with licensed, dated, auditable sources and cite each source. Housing, employment, safety, healthcare, immigration, legal, tax, and financial decisions should be independently verified.

The bundled images are design assets from the uploaded UI and are used as generic lifestyle visuals; they are not presented as verified photographs of specific Uzbekistan cities.

## Live app

Streamlit domain: `https://uzbekistan-ai-map-living-advisor.streamlit.app/`

A QR code for the live app is included at `assets/app_qr_code.png`.
