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

## Important image note

The bundled images came from the uploaded UI package and are used as generic lifestyle/design visuals. They are not asserted to be verified photographs of the Uzbekistan cities shown next to them.
