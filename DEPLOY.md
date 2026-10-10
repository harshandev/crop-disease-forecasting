# Free deployment (Streamlit Community Cloud) — done by the developer, once

Cost: ₹0. The person you hand the link to needs **no accounts and no installs**.

1. Create a **private** GitHub repo, e.g. `crop-disease-forecasting`.
2. Push this folder (the `.gitignore` already excludes `.venv/` and the big `data/raw/`):
   ```bash
   cd crop-disease-forecasting
   git init && git add . && git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/crop-disease-forecasting.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io → sign in with GitHub → **Create app** → pick the repo,
   branch `main`, main file `streamlit_app.py`.
4. Under **Advanced settings** choose **Python 3.12** (matches what was tested). Click **Deploy**.
   First build takes a few minutes.
5. Copy the app URL (`https://<name>.streamlit.app`). In app settings → **Sharing**, set it to
   **public / anyone with the link** so the viewer does not need to log in.
6. Open the link yourself 5 minutes before the demo (free apps sleep after ~12 h idle).

Only `requirements.txt` is installed on the server (no PyTorch). The LSTM runs from
`models/lstm_weights.npz` using NumPy.
