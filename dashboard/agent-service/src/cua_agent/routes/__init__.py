"""Routes package."""

from cua_agent.routes.routes import router
from cua_agent.routes.websocket import router as websocket_router

__all__ = ["router", "websocket_router"]
