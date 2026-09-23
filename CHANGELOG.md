# Changelog

## City photo and city-card repair

- Mapped all 16 Uzbekistan city profiles to the matching supplied city photograph.
- Applied the mapping to featured cards, recommendation results, the full city grid, and detailed city profiles.
- Rebuilt city-card markup as compact, escaped HTML rendered with `st.html()` so raw closing tags no longer appear in descriptions.
- Added `CITY_IMAGE_MAPPING.md` for maintainers.
- Preserved the earlier Streamlit Community Cloud toolbar-overlap fix and the complete imported UI source.
