"""Convenience entrypoint for the Phase 9D application API. No behavior of
its own beyond invoking uvicorn - all logic lives in application_api.py."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("application_api:app", host="127.0.0.1", port=8000, reload=False)
