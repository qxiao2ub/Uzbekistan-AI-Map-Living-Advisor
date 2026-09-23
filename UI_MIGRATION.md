# UI Migration Notes

This repository ports the uploaded React/Tailwind design into a Streamlit-compatible experience.

## Design elements preserved

- Image-led full-width hero with dark overlay and bottom-left typography
- Minimal light theme with muted green accents
- Rounded pill-style calls to action
- Featured location/city cards with large imagery, score badges, metadata, and tags
- Clean experience/feature rows with hover-lift styling
- Spacious section headers with uppercase eyebrow labels
- Dark, minimal footer treatment
- Supporting city-detail imagery
- Responsive layout for desktop and mobile

## Streamlit adaptation

The original UI was a Vite/React/Tailwind application. Streamlit Community Cloud starts a Python entry point, so the visual language was recreated in `app.py` using Streamlit components plus CSS/HTML styling. The AI recommendation, map, comparison, feedback, and city-explorer functions remain Python-native.

The original uploaded UI source is preserved under `ui_source/` for design reference and future front-end development. The uploaded `.env` is intentionally excluded from the repository.

## City-specific image update

The original hero and decorative assets remain from the uploaded UI package. The city explorer, featured recommendations, recommendation results, and detailed profiles now use the supplied city-specific photos from `assets/cities/`, mapped by exact city name.

## Visible-HTML repair

City cards are emitted through `st.html()` as compact, escaped markup. This prevents a blank line in a raw Markdown HTML block from terminating the block and showing closing tags such as `</div>` to users.
