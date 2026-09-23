# Akbarxon AI Living Advisor — Uzbekistan

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


## City-photo and card-rendering repair

- Every city card, featured match, recommendation result, and detailed city profile now resolves its photo by the exact city name.
- The 16 mapped images are stored under `assets/cities/` and are included in the ZIP.
- City-card HTML is rendered with `st.html()` and compact, escaped markup. This prevents Streamlit/Markdown from exposing closing tags such as `</div>` inside the visible city description area.
- The city-specific photo is resolved by exact city name at render time, with a fallback asset if a file is ever removed.

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
├── CITY_IMAGE_MAPPING.md
├── CHANGELOG.md
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── cities/                # city-specific photos supplied for all 16 profiles
│   └── original UI design assets
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

The city cards and detailed profiles use the 16 city-specific photos supplied with this revision. The remaining hero/detail artwork from the original UI package is retained only for general interface design. Confirm image ownership, attribution, and publication rights before a public production release.

## Live app

Streamlit domain: `https://uzbekistan-ai-map-living-advisor.streamlit.app/`

A QR code for the live app is included at `assets/app_qr_code.png`.

## Streamlit Cloud header-overlap fix

This repository includes a layout fix for Streamlit Community Cloud's fixed native toolbar. The main content container now reserves a safe top spacing before the custom Akbarxon AI Living Advisor brand/navigation area, while the sidebar keeps its compact spacing. This prevents the application title and navigation from being hidden underneath the Streamlit toolbar on wide desktop layouts.
