"""WebSocket manager for handling connections and broadcasting."""

import asyncio
import json
from typing import Dict, Literal, Set

from fastapi import WebSocket

from cua_agent.models.models import (
    ActiveTask,
    AgentCompleteEvent,
    AgentErrorEvent,
    AgentProgressEvent,
    AgentStartEvent,
    AgentStep,
    AgentTrace,
    AgentTraceMetadata,
    VncUrlSetEvent,
    VncUrlUnsetEvent,
    WebSocketEvent,
)


class WebSocketException(Exception):
    """Exception for WebSocket errors."""

    pass


class WebSocketManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        if websocket in self.connection_tasks:
            self.connection_tasks[websocket].cancel()
            del self.connection_tasks[websocket]
        print(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_message(self, message: WebSocketEvent, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(
                json.dumps(
                    message.model_dump(
                        mode="json",
                        context={"actions_as_json": True, "image_as_path": False},
                    )
                )
            )
        except Exception as e:
            print(f"Error sending personal message: {e}")
            if websocket in self.active_connections:
                self.disconnect(websocket)
            raise WebSocketException()

    async def send_agent_start(
        self,
        active_task: ActiveTask,
        websocket: WebSocket,
        status: Literal["max_sandboxes_reached", "success"],
    ):
        """Send agent start event."""
        event = AgentStartEvent(
            agentTrace=AgentTrace(
                id=active_task.message_id,
                timestamp=active_task.timestamp,
                instruction=active_task.instruction,
                modelId=active_task.model_id,
                steps=active_task.steps,
                traceMetadata=active_task.traceMetadata,
                isRunning=True,
            ),
            status=status,
        )
        await self.send_message(event, websocket)

    async def send_agent_progress(
        self,
        step: AgentStep,
        metadata: AgentTraceMetadata,
        websocket: WebSocket,
    ):
        """Send agent progress event."""
        event = AgentProgressEvent(
            agentStep=step,
            traceMetadata=metadata,
        )
        await self.send_message(event, websocket)

    async def send_agent_complete(
        self,
        metadata: AgentTraceMetadata,
        websocket: WebSocket,
        final_state: Literal[
            "success", "stopped", "max_steps_reached", "error", "sandbox_timeout"
        ],
    ):
        """Send agent complete event."""
        event = AgentCompleteEvent(traceMetadata=metadata, final_state=final_state)
        await self.send_message(event, websocket)

    async def send_agent_error(self, error: str, websocket: WebSocket):
        """Send agent error event."""
        event = AgentErrorEvent(error=error)
        await self.send_message(event, websocket)

    async def send_vnc_url_set(self, vnc_url: str, websocket: WebSocket):
        """Send VNC URL set event."""
        event = VncUrlSetEvent(vncUrl=vnc_url)
        await self.send_message(event, websocket)

    async def send_vnc_url_unset(self, websocket: WebSocket):
        """Send VNC URL unset event."""
        event = VncUrlUnsetEvent()
        await self.send_message(event, websocket)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
