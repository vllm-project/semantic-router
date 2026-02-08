"""Main entry point for the Computer Use Agent service."""

import os

import uvicorn

from cua_agent.app import app
from cua_agent.routes.routes import router
from cua_agent.routes.websocket import router as websocket_router

# Include routes
app.include_router(router, prefix="/api")
app.include_router(websocket_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "cua-agent"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"Starting Computer Use Agent Backend on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"WebSocket endpoint: ws://{host}:{port}/ws")

    uvicorn.run(
        "cua_agent.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
    )
