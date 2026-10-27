# CareWise — Personalized Healthcare & Medicine Information Dashboard

A beginner-friendly Streamlit dashboard based on the uploaded `Cleaned_Dataset.csv`.
It trains a reproducible scikit-learn pipeline at runtime, so it does not depend on the uploaded pickle model.

## Project files
- `app.py` — interactive dashboard and model pipeline
- `Cleaned_Dataset.csv` — dataset supplied for this project
- `requirements.txt` — Python packages

## Run in VS Code (Windows)
1. Install Python 3.10–3.12 and VS Code.
2. Extract this folder and open it in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Activate it:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.
6. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
8. Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Run in Google Colab
1. Upload `app.py`, `Cleaned_Dataset.csv`, and `requirements.txt` to the same Colab session (left sidebar → Files → Upload).
2. Run:
   ```python
   !pip install -r requirements.txt
   ```
3. For a temporary public tunnel, run:
   ```python
   !streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
   !pip -q install pyngrok
   from pyngrok import ngrok
   print(ngrok.connect(8501))
   ```
   If ngrok asks for an authentication token, configure your own token in Colab; do not paste it into a public notebook. Alternatively use VS Code or Streamlit Community Cloud.
4. Keep the Colab runtime running while you use the dashboard. Colab tunnels are temporary.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `Cleaned_Dataset.csv`, and `requirements.txt`.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose the repository, branch, and `app.py` as the main file.
4. Deploy. The dataset must be committed in the same repository folder as `app.py`.

## What the dashboard includes
- Dashboard KPIs and plots
- Symptom/profile demo assessment with top model outputs
- Medicine safety information and clinician discussion checklist
- Dataset explorer and CSV download
- Test-split evaluation, confusion matrix, classification report, feature importance

## Important medical safety
This is an educational project, not a medical device. Dataset labels and model scores are not diagnoses or calibrated clinical probabilities. The app does not prescribe drugs or doses. Never start, stop, or change treatment based on this app. Seek qualified care for health concerns; severe or sudden breathing difficulty is an emergency.
