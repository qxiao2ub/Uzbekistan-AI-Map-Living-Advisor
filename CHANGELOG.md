# Changelog

## 2026-10-05
- Fixed a Streamlit startup `NameError: name 't' is not defined`.
- Removed all runtime `t()` calls from the module-level English translation dictionary.
- Added a dedicated startup/debug validation note and regression checks.
- Preserved the cumulative visitor counter, English/Russian/Uzbek presentation, city-specific photos, corrected HTML cards, and Streamlit toolbar-overlap fix.

## 2026-10-01
- Added persistent cumulative app visit counter using a public external counter service; no database is stored in the repository.
- Counter is incremented once per Streamlit browser session and displayed in the sidebar and main canvas on every navigation page.
- Added three UI languages: English, Russian, and Uzbek.
- Localized major navigation, controls, explanatory copy, feature labels, and all 16 city descriptions.
- Preserved the previous city-specific photo mapping and top-toolbar overlap fix.
