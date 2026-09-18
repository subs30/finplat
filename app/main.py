from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import ai, approvals, auth, detection, documents, investigations, rag, users

app = FastAPI(
    title="finplat — AI Financial Crime Intelligence & Investigation Platform",
    version="0.4.0",
)

# No frontend exists yet in V0.1; FRONTEND_ORIGIN lets a future one be
# allowed without a code change once it exists.
_allowed_origins: list[str] = []
if _frontend_origin := get_settings().FRONTEND_ORIGIN:
    _allowed_origins.append(_frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(ai.router)
app.include_router(documents.router)
app.include_router(rag.router)
app.include_router(detection.router)
app.include_router(investigations.router)
app.include_router(approvals.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
