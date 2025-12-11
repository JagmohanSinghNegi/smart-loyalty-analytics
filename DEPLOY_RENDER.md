Render deployment notes

1. Two deployment options

Option A — Single Web Service (recommended)

- Root directory: repository root
- Build command: `pip install -r backend/requirements.txt`
- Start command (Procfile provided): `web: gunicorn backend.app:app --bind 0.0.0.0:$PORT`
- Pros: Flask serves both API and static frontend (no CORS, same origin).

Option B — Separate frontend (static) + backend

- Backend: create a Web Service with Root Directory `backend` and use the provided `backend/Procfile` (or start command `gunicorn app:app --bind 0.0.0.0:$PORT`).
- Frontend: create a Static Site, or host elsewhere. If frontend is hosted separately, set a global `BACKEND_URL` in your built HTML so `frontend/js/script.js` can call the correct backend URL.
  Example injection: add in your `index.html` (before `script.js`):

  <script>
    window.BACKEND_URL = "https://your-backend-service.onrender.com";
  </script>

2. Important checks before deploy

- Ensure model files exist in `models/` in the deployed repo: `loyalty_model.pkl`, `pipeline.pkl` (if used), and `rfm_features.csv`. If missing, `/predict-loyalty` will return errors.
- `backend/app.py` already sets permissive CORS headers via `@app.after_request`.

3. Local testing notes (Windows)

- Gunicorn is POSIX-only and not available on Windows. Locally test with Flask's dev server:

  ```powershell
  # from repository root
  python -m pip install -r backend/requirements.txt
  python -m flask --app backend.app run
  ```

- Or use `waitress` on Windows for a production-like server:

  ```powershell
  python -m pip install waitress
  cd backend
  python -m waitress --port=5000 backend.app:app
  ```

4. Troubleshooting

- If the UI shows "Network error...127.0.0.1:5000", it means the frontend is pointing to localhost. Either:
  - Deploy backend and serve frontend from same origin (recommended), or
  - Set `window.BACKEND_URL` in your frontend pages to the public backend URL.

5. Quick Render check list

- Add repo (GitHub/Git) to Render
- Create one Web Service using repo root (or backend as Root Directory)
- Set build and start commands as above
- Deploy and open the Render URL
- Test `/health` (e.g. `https://your-app.onrender.com/health`) — should return `{ "status": "ok" }`.
