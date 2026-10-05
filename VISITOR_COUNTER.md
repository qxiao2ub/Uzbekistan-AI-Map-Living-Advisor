# Visitor counter design

## Behavior

1. A new Streamlit session calls `https://counterapi.com/api/uzbekistan-ai-map-living-advisor.streamlit.app/view/app-users`.
2. The returned cumulative value is stored in `st.session_state` so Streamlit reruns do not increment it repeatedly during the same session.
3. The value is rendered in the left sidebar and the main content area for every navigation page.
4. If the external counter is temporarily unreachable, the UI shows `—` rather than fabricating a zero.

## No database in this repository

The project itself does not create or require SQLite, PostgreSQL, Supabase, Firebase, MongoDB, or another database for visitor counting. Persistence is provided by the external counter service.

## Important interpretation

The metric should be described as cumulative visits/sessions rather than a guaranteed count of unique people.
