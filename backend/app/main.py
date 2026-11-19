from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import audit, health, inference, reviews, studies, upload
from app.core.config import settings
from app.core.security import resolve_safe_path
from app.database.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Educational and research AI-assisted medical imaging PACS API. "
        "FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Reviewer-Role"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled API exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error processing request."},
    )


app.include_router(health.router)
app.include_router(studies.router)
app.include_router(upload.router)
app.include_router(inference.router)
app.include_router(reviews.router)
app.include_router(audit.router)


@app.get("/api/artifacts/{filename}", tags=["artifacts"])
def serve_artifact(filename: str):
    """Safely serves generated PNG previews, Grad-CAM heatmaps, and segmentation masks from disk."""
    target = resolve_safe_path(settings.artifacts_dir, filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")
    return FileResponse(target, media_type="image/png")
