# Phase 9D.0 — API Dependency Bootstrap

Application-infrastructure-only dependency addition, executed under explicit user authorization,
prior to implementing the Phase 9D versioned application prediction API. This step installs the
minimal HTTP API stack (FastAPI, Pydantic, Uvicorn) and nothing else. It does not touch RF V2,
XGB V3, preprocessing, state snapshots, inference semantics, explanation semantics, historical
replay, or evaluation artifacts.

## 1. Environment snapshot

- Python: **3.14.3**
- pip: **25.3**
- Pre-existing installed package count: **149**
- Git status at start: working tree had accumulated Phase 9A/9B/9C artifacts (56 changed/added
  paths), no uncommitted change touched `requirements.txt` prior to this step.
- `requirements.txt` SHA-256 **before** this step: `325d1900b3b88be2546baac18d6529603cb4a68f3e0b5ef52bcbf6d40a9a665d`
- `requirements.txt` SHA-256 **after** this step: `68b1c26eb820c03e58cce2a99442f1191126820c32b1ca03a7483b247f0aaf3f`

## 2. Packages installed

Installed via `pip install fastapi pydantic uvicorn` (no version pins supplied — resolver chose
versions). No database, authentication, deployment/cloud SDK, Redis, Celery, gunicorn, or FastAPI
Cloud tooling packages were installed. `httpx` (already present at `0.28.1`) was reused as the
test HTTP client, not reinstalled.

| Package | Version | Reason |
|---|---|---|
| `fastapi` | 0.141.1 | requested (API framework) |
| `pydantic` | 2.13.4 | requested (schema validation) |
| `uvicorn` | 0.52.3 | requested (ASGI server) |
| `starlette` | 1.6.0 | transitive — FastAPI's ASGI toolkit dependency |
| `pydantic_core` | 2.46.4 | transitive — Pydantic's Rust core |
| `annotated-types` | 0.8.0 | transitive — Pydantic dependency |
| `typing-inspection` | 0.4.4 | transitive — Pydantic dependency |

**Total new packages: 7.** Post-install installed package count: **156** (149 + 7).

## 3. Existing-environment protection check

Full `pip list` was snapshotted before and after installation and diffed line-by-line.

`existing_pinned_dependency_changes = 0`

No existing package's version was upgraded, downgraded, or removed. The only diff between the
before/after snapshots is the addition of the 7 rows listed above. No conflict was encountered,
so no STOP condition was triggered.

## 4. `requirements.txt` update

Updated conservatively, as a pure append/insert operation:

- File format preserved exactly: UTF-16 LE encoding with BOM, `\r\n` line endings.
- All **145** pre-existing lines preserved byte-for-byte, in their original order.
- **7** new lines inserted at alphabetically-reasonable positions consistent with the file's
  existing (mostly-alphabetical) convention:
  - `annotated-types==0.8.0` — between `annotated-doc==0.0.4` and `anyio==4.12.1`
  - `fastapi==0.141.1` — between `faiss-cpu==1.13.2` and `fastjsonschema==2.22.1`
  - `pydantic==2.13.4` — between `pycparser==3.0` and `pydantic_core==2.46.4`
  - `pydantic_core==2.46.4` — between `pydantic==2.13.4` and `Pygments==2.19.2`
  - `starlette==1.6.0` — between `stack-data==0.6.3` and `svglib==1.6.0`
  - `typing-inspection==0.4.4` — between `typer==0.24.1` and `typing_extensions==4.15.0`
  - `uvicorn==0.52.3` — between `urllib3==2.6.3` and `wcwidth==0.8.2`
- File was **not** regenerated; no `pip freeze > requirements.txt` was run.
- Resulting file: **152** lines total.

## 5. Import contract verification

```
import fastapi, pydantic, starlette, uvicorn, httpx
```

Result: **PASS**, both immediately post-install and re-verified after the `requirements.txt`
edit. Resolved versions: `fastapi 0.141.1`, `pydantic 2.13.4`, `starlette 1.6.0`,
`uvicorn 0.52.3`, `httpx 0.28.1` (unchanged, pre-existing).

## 6. FastAPI/TestClient/OpenAPI smoke check

Executed as a scratch (uncommitted) check only — no application code was written for this step.

- `FastAPI()` app instantiation: **PASS**
- Pydantic `BaseModel` request/response schema definition: **PASS**
- `fastapi.testclient.TestClient` (backed by `starlette.testclient`, using the installed
  `httpx==0.28.1` as its transport): **PASS** — one informational
  `StarletteDeprecationWarning` was emitted (referencing a future optional `httpx2` package);
  this is non-blocking and does not indicate an incompatibility. No `httpx` downgrade was
  performed or needed.
- `/openapi.json` generation via `TestClient`: **PASS**
- `uvicorn` import (ASGI server, not run standalone in this smoke check): **PASS**

## 7. Full existing test-suite regression gate

- `pytest` (full repository suite): **674 passed**, 1 warning — identical to the pre-bootstrap
  baseline. **Zero regressions.**
- `scripts/validate_phase9b.py`: **32/32 checks passed.**
- `scripts/validate_phase9c.py`: **33/33 checks passed.**

## Summary

```
API DEPENDENCY BOOTSTRAP = COMPLETE
EXISTING PINNED DEPENDENCY CHANGES = 0
NEW PACKAGES INSTALLED = 7 (fastapi, pydantic, uvicorn, starlette, pydantic_core,
                             annotated-types, typing-inspection)
IMPORT CONTRACT = VERIFIED
FASTAPI/TESTCLIENT/OPENAPI SMOKE CHECK = PASSED
FULL TEST SUITE = 674 PASSED / 0 REGRESSIONS
PHASE 9B VALIDATOR = 32/32
PHASE 9C VALIDATOR = 33/33
RF V2 = UNCHANGED
XGB V3 = UNCHANGED
READY TO RESUME PHASE 9D IMPLEMENTATION
```
