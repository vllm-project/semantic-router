"""WebSocket routes for the Computer Use Agent service."""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cua_agent.app import app
from cua_agent.models.models import AgentErrorEvent, AgentTrace, HeartbeatEvent
from cua_agent.services.agent_service import AgentService
from cua_agent.websocket.websocket_manager import WebSocketManager

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    websocket_manager: WebSocketManager = app.state.websocket_manager
    agent_service: AgentService = app.state.agent_service

    await websocket_manager.connect(websocket)

    try:
        # Create ID and register websocket
        uuid = await agent_service.create_id_and_sandbox(websocket)
        welcome_message = HeartbeatEvent(uuid=uuid)
        await websocket_manager.send_message(welcome_message, websocket)

        # Keep the connection alive and wait for messages
        while True:
            try:
                data = await websocket.receive_text()

                try:
                    message_data = json.loads(data)
                    print(f"Received message: {message_data}")

                    if message_data.get("type") == "user_task":
                        trace_data = message_data.get("trace")
                        if trace_data:
                            # Convert timestamp string to datetime if needed
                            if isinstance(trace_data.get("timestamp"), str):
                                from datetime import datetime

                                trace_data["timestamp"] = datetime.fromisoformat(
                                    trace_data["timestamp"].replace("Z", "+00:00")
                                )

                            trace = AgentTrace(**trace_data)
                            trace_id = await agent_service.process_user_task(
                                trace, websocket
                            )
                            print(f"Started processing trace: {trace_id}")
                        else:
                            print("No trace data in message")

                    elif message_data.get("type") == "stop_task":
                        trace_id = message_data.get("trace_id")
                        if trace_id:
                            await agent_service.stop_task(trace_id)
                            print(f"Stopped task: {trace_id}")
                        else:
                            print("No trace ID in message")

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    error_response = AgentErrorEvent(
                        type="agent_error", error="Invalid JSON format"
                    )
                    await websocket_manager.send_message(error_response, websocket)

                except Exception as e:
                    print(f"Error processing message: {e}")
                    import traceback

                    traceback.print_exc()
                    error_response = AgentErrorEvent(
                        type="agent_error", error=f"Error processing message: {str(e)}"
                    )
                    await websocket_manager.send_message(error_response, websocket)

            except Exception as e:
                print(f"Error receiving WebSocket message: {e}")
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected normally")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        try:
            await agent_service.cleanup_tasks_for_websocket(websocket)
        except Exception as e:
            print(f"Error cleaning up tasks for websocket: {e}")

        websocket_manager.disconnect(websocket)
