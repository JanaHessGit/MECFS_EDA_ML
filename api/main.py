from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import eda, predict

app = FastAPI(
    title="ME/CFS Data Explorer API",
    description=(
        "A learning-project API serving a small classifier trained on a "
        "public ME/CFS research dataset. Not a diagnostic tool — see "
        "PROJECT_PLAN.md for context on sample sizes and labeling caveats."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local practice; scope this down before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(eda.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "message": "ME/CFS Data Explorer API — see /docs for the interactive API reference",
        "disclaimer": "Research/learning project. Not a diagnostic tool.",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
