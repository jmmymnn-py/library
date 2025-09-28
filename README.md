# Library (Streamlit App)

A personal library viewer/editor. Edits in the app overwrite the CSV source:
- Locally: saves to a local CSV.
- On Streamlit Cloud / GitHub: commits changes back to the repo using the GitHub API.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
