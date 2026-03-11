# Training Data Curation Bot

A modular Python project for loading documents, generating training data, evaluating quality, exporting datasets, serving an API with FastAPI, and providing a UI with Streamlit.

## Planned Architecture

- `training_data_bot/` → core business logic
- `app/api/` → FastAPI backend
- `app/ui/` → Streamlit frontend

## Run locally

### FastAPI
uvicorn app.api.main:app --reload

### Streamlit
streamlit run app/ui/streamlit_app.py