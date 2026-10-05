# Debug Fix Report

## Observed production error

Streamlit Cloud reported:

`NameError: name 't' is not defined`

The traceback pointed to `app.py` around the English `TRANSLATIONS` dictionary.

## Root cause

Python evaluates the right-hand side of dictionary entries immediately while importing the module. The English dictionary contained entries such as `"question": t("question")`, but the `t()` function was defined later in the file. Therefore the module failed before `main()` could run.

## Repair

The affected English entries now contain their literal English strings. The `t()` helper remains defined immediately after the localization dictionary and is used at runtime when rendering the interface.

This is intentionally preferable to calling `t()` while building `TRANSLATIONS`, because `t()` is a runtime lookup helper that reads the current Streamlit session language.

## Regression checks

The deliverable was checked for:

- `python -m py_compile app.py` success
- zero `t()` calls inside the module-level `TRANSLATIONS` dictionary
- all three language dictionaries containing the same translation keys
- 16 city profiles loading successfully
- recommendation ranking smoke test
- city-card HTML generation without visible Markdown/parser breakage
- required assets and repository files present

Streamlit itself was not installed in the offline build container, so a full browser-rendered Streamlit Cloud test could not be performed here.
