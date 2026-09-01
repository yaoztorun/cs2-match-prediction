# CS2 Match Prediction — Web Frontend

Next.js PWA frontend for the CS2 match prediction system. It contains no model
logic — all predictions come from the FastAPI backend (see the repository root
README).

## Development

```bash
npm install
npm run dev
```

The dev server proxies `/api/*` to the backend at `http://127.0.0.1:8000`, so
start the API first (from the repository root):

```bash
python -m application.api.run_application_api
```

## Tests

```bash
npm test
```
