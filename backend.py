"""
backend.py

FastAPI application: a thin HTTP layer over engine.Engine.

Every browser gets its own agent through a session cookie (see sessions.py),
so the app is safe to host publicly.

    GET  /                    dashboard (sets the session cookie)
    GET  /healthz             health check for hosting platforms
    GET  /api/config          defaults + available maps
    GET  /api/state           latest snapshot (JSON)
    GET  /api/stream          live snapshots over Server-Sent Events
    POST /api/train           start (or restart) training with hyperparameters
    POST /api/stop            stop the running training job
    POST /api/reset           discard the run -> fresh untrained agent
    POST /api/play/reset      start a new greedy episode
    POST /api/play/step       advance the episode one step (optional manual action)
    POST /api/evaluate        greedy evaluation of the current Q-table

Run:  uvicorn backend:app --reload          (use --workers 1: state lives in memory)
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import Engine, EpisodeFinished, TrainParams
from sessions import Busy, SessionStore

FRONTEND_DIR = Path(__file__).parent / "frontend"
COOKIE = "fl_sid"

app = FastAPI(title="FrozenLake Q-Learning Lab", version="1.1.0")
store = SessionStore()


class TrainRequest(BaseModel):
    map_name: str = Field("4x4", pattern="^(4x4|8x8)$")
    slippery: bool = True
    alpha: float = Field(0.1, gt=0.0, le=1.0, description="learning rate")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="discount factor")
    epsilon_start: float = Field(1.0, ge=0.0, le=1.0)
    epsilon_min: float = Field(0.01, ge=0.0, le=1.0)
    epsilon_decay: float = Field(0.9995, gt=0.0, le=1.0, description="per-episode multiplier")
    episodes: int = Field(15_000, ge=1, le=100_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    pace: int = Field(3_000, ge=0, le=1_000_000, description="episodes/sec cap, 0 = unthrottled")

    def to_params(self) -> TrainParams:
        return TrainParams(**self.model_dump())


class StepRequest(BaseModel):
    action: Optional[int] = Field(None, ge=0, le=3, description="0=Left 1=Down 2=Right 3=Up; omit for greedy")


class EvalRequest(BaseModel):
    episodes: int = Field(1_000, ge=1, le=10_000)


def engine_for(request: Request) -> Engine:
    sid = request.cookies.get(COOKIE)
    if not SessionStore.valid(sid):
        raise HTTPException(status_code=401, detail="No session found. Reload the page.")
    try:
        return store.get(sid)
    except Busy as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ------------------------------------------------------------------ routes
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True, "sessions": len(store), "training": store.training_count()}


@app.get("/api/config")
def get_config():
    return Engine.defaults()


@app.get("/api/state")
def get_state(request: Request):
    return Response(engine_for(request).current(), media_type="application/json")


@app.get("/api/stream")
def stream(request: Request):
    engine = engine_for(request)
    sid = request.cookies.get(COOKIE)

    def events():
        last = -1
        while True:
            version, payload = engine.wait(last, timeout=10)
            store.touch(sid)  # an open dashboard keeps its session alive
            if version != last:
                last = version
                yield f"data: {payload}\n\n"
            else:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/train", status_code=202)
def start_training(request: Request, req: TrainRequest):
    engine = engine_for(request)
    if not engine.is_training and store.training_count() >= store.max_training:
        raise HTTPException(
            status_code=503,
            detail="Other visitors are training right now. Try again in a minute.",
        )
    return {"run_id": engine.start_training(req.to_params()), "status": "training"}


@app.post("/api/stop")
def stop_training(request: Request):
    engine_for(request).stop_training()
    return {"status": "stopped"}


@app.post("/api/reset")
def reset(request: Request, req: TrainRequest):
    engine_for(request).reset(req.to_params())
    return {"status": "idle"}


@app.post("/api/play/reset")
def play_reset(request: Request):
    return engine_for(request).play_reset()


@app.post("/api/play/step")
def play_step(request: Request, req: StepRequest = StepRequest()):
    try:
        return engine_for(request).play_step(req.action)
    except EpisodeFinished as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.post("/api/evaluate")
def evaluate_policy(request: Request, req: EvalRequest = EvalRequest()):
    return engine_for(request).evaluate_policy(req.episodes)


# ---------------------------------------------------------------- frontend
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index(request: Request):
    page = FRONTEND_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    response = FileResponse(page)
    if not SessionStore.valid(request.cookies.get(COOKIE)):
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        response.set_cookie(
            COOKIE, SessionStore.new_id(), max_age=60 * 60 * 24 * 30,
            httponly=True, samesite="lax", secure=(proto == "https"),
        )
    return response
