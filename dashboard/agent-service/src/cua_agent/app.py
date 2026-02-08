"""FastAPI application for Computer Use Agent service."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cua_agent.services.agent_service import AgentService
from cua_agent.services.sandbox_service import SandboxService
from cua_agent.websocket.websocket_manager import WebSocketManager

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    print("Initializing Computer Use Agent services...")

    # Check for required API keys
    if not os.getenv("E2B_API_KEY"):
        raise ValueError("E2B_API_KEY is not set. Get one at https://e2b.dev")

    # Check for model provider (HuggingFace or Ollama)
    if not os.getenv("HF_TOKEN") and not os.getenv("OLLAMA_HOST"):
        print("Warning: Neither HF_TOKEN nor OLLAMA_HOST is set. Using default Ollama.")
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    max_sandboxes = int(os.getenv("MAX_SANDBOXES", "10"))

    websocket_manager = WebSocketManager()
    sandbox_service = SandboxService(max_sandboxes=max_sandboxes)
    agent_service = AgentService(websocket_manager, sandbox_service, max_sandboxes)

    # Start periodic cleanup of stuck sandboxes
    sandbox_service.start_periodic_cleanup()

    # Store services in app state for access in routes
    app.state.websocket_manager = websocket_manager
    app.state.sandbox_service = sandbox_service
    app.state.agent_service = agent_service

    print("Computer Use Agent services initialized successfully")

    yield

    print("Shutting down Computer Use Agent services...")
    sandbox_service.stop_periodic_cleanup()
    await agent_service.cleanup()
    await sandbox_service.cleanup_sandboxes()
    print("Computer Use Agent services shut down successfully")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Computer Use Agent Backend",
    description="Backend API for Computer Use Agent - E2B-powered desktop automation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
