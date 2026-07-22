# Akbarxon AI Living Advisor for Uzbekistan

A Streamlit portfolio project that recommends cities in Uzbekistan from a user's natural-language living preferences.

## Project team

- **Author:** Akbarxon Nasirov
- **Mentor:** Dr. Qingyang Xiao

The author and mentor are displayed in two places in the web app:

1. At the top of the left sidebar, together with the app title.
2. In the **AI laboratory** tab under **Project team**.

## Main features

- Natural-language preference parsing
- Weighted and explainable city ranking
- TF-IDF and cosine-similarity matching
- K-means city archetypes
- Three-hidden-layer MLP neural-network demonstration
- Like/dislike feedback with a Beta-Bernoulli bandit-style adjustment
- Interactive Folium/OpenStreetMap visualization
- Plotly comparisons and radar charts
- Optional live weather and Wikipedia context

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Extract this ZIP file.
2. Create a new GitHub repository.
3. Upload all files and folders from the extracted repository into the repository root.
4. In Streamlit Community Cloud, select the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the app.

The following files must stay together in the repository root:

```text
akbarxon-ai-living-advisor-updated/
├── .streamlit/
│   └── config.toml
├── .gitignore
├── app.py
├── cities_uzbekistan.csv
├── requirements.txt
├── README.md
└── Akbarxon_AI_Living_Advisor_Uzbekistan_Updated_Colab.ipynb
```

## Data and model disclaimer

The included city scores are illustrative prototype values rather than official measurements. Before a public release, replace them with licensed, dated, auditable data and show the source, retrieval date, geographic scope, and confidence for each metric.

This educational prototype does not provide legal, immigration, housing, medical, employment, or financial advice.
