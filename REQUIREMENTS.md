# Runtime dependencies

**Local** (tests, data audit, scoring, figures; no GPU): `requirements-dev.txt`, installed into a venv:

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

Local development used Python 3.9.1; `aiohttp==3.9.5` is pinned because newer aiohttp breaks `datasets` on
Python 3.9's `typing` module. Python 3.11+ is recommended.

**Kaggle** (GPU inference): `requirements-kaggle.txt`, pinned to the versions that actually resolved on the
real runs (transformers 5.16.1, datasets 4.8.5, torchcodec 0.11.0, jiwer 4.0.0). `torch` is Kaggle's
platform build (observed 2.11.0+cu128) and is not reinstalled. The bundled script pip-installs missing
packages at start-up and records the versions it actually used in each run's `experiment_config.json`.
