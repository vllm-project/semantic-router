"""Agent service for managing agent tasks and processing."""

import asyncio
import base64
import logging
import os
import time
from io import BytesIO
from typing import Callable
from uuid import uuid4

from e2b_desktop import Sandbox, TimeoutException
from fastapi import WebSocket
from PIL import Image
from smolagents import ActionStep, AgentMaxStepsError, TaskStep
from starlette.websockets import WebSocketState

from cua_agent.models.models import (
    ActiveTask,
    AgentAction,
    AgentStep,
    AgentTrace,
    AgentTraceMetadata,
    parse_function_call,
)
from cua_agent.services.agent_utils.desktop_agent import E2BVisionAgent
from cua_agent.services.agent_utils.get_model import get_model
from cua_agent.services.sandbox_service import SandboxService
from cua_agent.services.utils import compress_image_to_max_size
from cua_agent.websocket.websocket_manager import WebSocketException, WebSocketManager

logger = logging.getLogger(__name__)

# Timeout constants
AGENT_RUN_TIMEOUT = 1000  # seconds - maximum time for agent.run()
SANDBOX_KILL_TIMEOUT = 30  # seconds


class AgentStopException(Exception):
    """Exception for agent stop."""

    pass


class AgentService:
    """Service for handling agent tasks and processing."""

    def __init__(
        self,
        websocket_manager: WebSocketManager,
        sandbox_service: SandboxService,
        max_sandboxes: int,
    ):
        self.active_tasks: dict[str, ActiveTask] = {}
        self.websocket_manager: WebSocketManager = websocket_manager
        self.task_websockets: dict[str, WebSocket] = {}
        self.sandbox_service: SandboxService = sandbox_service
        self.last_screenshot: dict[str, tuple[Image.Image, str] | None] = {}
        self._lock = asyncio.Lock()
        self.max_sandboxes = max_sandboxes

    async def create_id_and_sandbox(self, websocket: WebSocket) -> str:
        """Create a new ID and register websocket."""
        async with self._lock:
            uuid = str(uuid4())
            while uuid in self.active_tasks:
                uuid = str(uuid4())
            self.task_websockets[uuid] = websocket
        logger.info(f"Created UUID {uuid} and registered websocket")
        return uuid

    async def process_user_task(
        self, trace: AgentTrace, websocket: WebSocket
    ) -> str | None:
        """Process a user task and return the trace ID."""
        trace_id = trace.id
        trace.steps = []
        trace.traceMetadata = AgentTraceMetadata(traceId=trace_id)

        async with self._lock:
            if trace_id not in self.task_websockets:
                self.task_websockets[trace_id] = websocket

            if self.task_websockets[trace_id] != websocket:
                if trace_id in self.task_websockets:
                    del self.task_websockets[trace_id]

            active_task = ActiveTask(
                message_id=trace_id,
                instruction=trace.instruction,
                model_id=trace.modelId,
                timestamp=trace.timestamp,
                steps=trace.steps,
                traceMetadata=trace.traceMetadata,
            )

            if len(self.active_tasks) >= self.max_sandboxes:
                await self.websocket_manager.send_agent_start(
                    active_task=active_task,
                    status="max_sandboxes_reached",
                    websocket=websocket,
                )
                return trace_id

            self.active_tasks[trace_id] = active_task
            self.last_screenshot[trace_id] = None

        asyncio.create_task(self._agent_processing(trace_id))
        return trace_id

    async def _agent_runner(
        self,
        message_id: str,
        step_callback: Callable[[ActionStep, E2BVisionAgent], None],
    ):
        """Run the task with the appropriate agent."""
        sandbox: Sandbox | None = None
        agent = None
        novnc_active = False
        websocket_exception = False
        final_state = "success"

        try:
            websocket = self.task_websockets.get(message_id)

            await self.websocket_manager.send_agent_start(
                active_task=self.active_tasks[message_id],
                websocket=websocket,
                status="success",
            )

            model = get_model(self.active_tasks[message_id].model_id)

            # Wait for sandbox to be ready
            max_attempts = 60  # 2 minutes timeout
            sandbox = None
            for attempt in range(max_attempts):
                response = await self.sandbox_service.acquire_sandbox(message_id)

                if response.error:
                    logger.error(f"Sandbox creation failed for {message_id}: {response.error}")
                    await asyncio.sleep(2)
                    continue

                if response.sandbox is not None and response.state == "ready":
                    sandbox = response.sandbox
                    break

                if response.state == "max_sandboxes_reached":
                    available, pending = await self.sandbox_service.get_sandbox_counts()
                    logger.warning(
                        f"Sandbox pool at capacity for {message_id}: "
                        f"{available} ready, {pending} pending, max: {self.max_sandboxes}"
                    )
                    await asyncio.sleep(2)
                    continue

                if attempt > 0 and attempt % 10 == 0:
                    logger.info(
                        f"Waiting for sandbox for {message_id}, attempt {attempt}/{max_attempts}"
                    )

                await asyncio.sleep(2)

            if sandbox is None:
                available, pending = await self.sandbox_service.get_sandbox_counts()
                # Get the last error for better debugging
                final_response = await self.sandbox_service.acquire_sandbox(message_id)
                error_info = ""
                if final_response.error:
                    error_info = f" Last creation error: {final_response.error}"
                    logger.error(f"Sandbox creation failed: {final_response.error}")
                raise Exception(
                    f"No sandbox available for {message_id} after {max_attempts} attempts: "
                    f"{available} ready, {pending} pending, max: {self.max_sandboxes}.{error_info}"
                )

            data_dir = self.active_tasks[message_id].trace_path
            user_content = self.active_tasks[message_id].instruction

            agent = E2BVisionAgent(
                model=model,
                data_dir=data_dir,
                desktop=sandbox,
                step_callbacks=[step_callback],
            )

            self.active_tasks[message_id].traceMetadata.maxSteps = agent.max_steps

            await self.websocket_manager.send_vnc_url_set(
                vnc_url=sandbox.stream.get_url(
                    auto_connect=True,
                    view_only=True,
                    resize="scale",
                    auth_key=sandbox.stream.get_auth_key(),
                )
                or "",
                websocket=websocket,
            )
            novnc_active = True

            step_filename = f"{message_id}-1"
            screenshot_bytes = agent.desktop.screenshot()
            image = Image.open(BytesIO(screenshot_bytes))
            self.last_screenshot[message_id] = (image, step_filename)

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(agent.run, user_content),
                    timeout=AGENT_RUN_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Agent run timed out after {AGENT_RUN_TIMEOUT} seconds for {message_id}"
                )
                raise Exception(f"Agent run timed out after {AGENT_RUN_TIMEOUT} seconds")

            self.active_tasks[message_id].traceMetadata.completed = True

        except AgentStopException as e:
            if str(e) == "Max steps reached":
                final_state = "max_steps_reached"
            elif str(e) == "Task not completed":
                final_state = "stopped"

        except WebSocketException:
            websocket_exception = True

        except TimeoutException:
            final_state = "sandbox_timeout"

        except Exception:
            import traceback
            logger.error(f"Error processing task: {traceback.format_exc()}")
            final_state = "error"
            if (
                not websocket_exception
                and websocket
                and websocket.client_state == WebSocketState.CONNECTED
            ):
                await self.websocket_manager.send_agent_error(
                    error="Error processing task", websocket=websocket
                )

        finally:
            if (
                not websocket_exception
                and websocket
                and websocket.client_state == WebSocketState.CONNECTED
            ):
                await self.websocket_manager.send_agent_complete(
                    metadata=self.active_tasks[message_id].traceMetadata,
                    websocket=websocket,
                    final_state=final_state,
                )

                if novnc_active:
                    await self.websocket_manager.send_vnc_url_unset(websocket=websocket)

            novnc_active = False

            await self.active_tasks[message_id].update_trace_metadata(
                final_state=final_state,
                completed=True,
            )

            if message_id in self.active_tasks:
                await self.active_tasks[message_id].save_to_file()

            async with self._lock:
                if message_id in self.active_tasks:
                    del self.active_tasks[message_id]

                if message_id in self.task_websockets:
                    del self.task_websockets[message_id]

                if message_id in self.last_screenshot:
                    del self.last_screenshot[message_id]

            try:
                await self.sandbox_service.release_sandbox(message_id)
            except Exception as e:
                logger.error(f"Error releasing sandbox for {message_id}: {e}")

    async def _agent_processing(self, message_id: str):
        """Process the user task with the appropriate agent."""
        try:
            active_task = self.active_tasks[message_id]
            os.makedirs(active_task.trace_path, exist_ok=True)
            loop = asyncio.get_running_loop()

            def step_callback(memory_step: ActionStep, agent: E2BVisionAgent):
                assert memory_step.step_number is not None

                if memory_step.step_number > agent.max_steps:
                    raise AgentStopException("Max steps reached")

                if self.active_tasks[message_id].traceMetadata.completed:
                    raise AgentStopException("Task not completed")

                model_output = (
                    memory_step.model_output_message.content
                    if memory_step.model_output_message
                    else None
                )
                if isinstance(memory_step.error, AgentMaxStepsError):
                    model_output = memory_step.action_output

                thought = (
                    model_output.split("```")[0].replace("\nAction:\n", "")
                    if model_output
                    and (
                        memory_step.error is None
                        or isinstance(memory_step.error, AgentMaxStepsError)
                    )
                    else None
                )

                if model_output is not None:
                    action_sequence = model_output.split("```")[1] if "```" in model_output else model_output
                else:
                    action_sequence = "The task failed due to an error"

                agent_actions = (
                    AgentAction.from_function_calls(parse_function_call(action_sequence))
                    if action_sequence
                    else None
                )

                time.sleep(3)

                image, step_filename = self.last_screenshot[message_id]  # type: ignore
                assert image is not None and step_filename is not None
                screenshot_path = os.path.join(agent.data_dir, f"{step_filename}.png")
                image.save(screenshot_path)

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                del buffered
                del image

                if memory_step.token_usage is not None:
                    step = AgentStep(
                        traceId=message_id,
                        stepId=str(memory_step.step_number),
                        image=image_base64,
                        thought=thought,
                        actions=agent_actions or [],
                        error=memory_step.error.message if memory_step.error else None,
                        duration=memory_step.timing.duration,
                        inputTokensUsed=memory_step.token_usage.input_tokens,
                        outputTokensUsed=memory_step.token_usage.output_tokens,
                        step_evaluation="neutral",
                    )

                    future1 = asyncio.run_coroutine_threadsafe(
                        self.active_tasks[message_id].update_trace_metadata(
                            step_input_tokens_used=memory_step.token_usage.input_tokens,
                            step_output_tokens_used=memory_step.token_usage.output_tokens,
                            step_duration=memory_step.timing.duration,
                            step_numberOfSteps=1,
                        ),
                        loop,
                    )
                    future2 = asyncio.run_coroutine_threadsafe(
                        self.active_tasks[message_id].update_step(step),
                        loop,
                    )
                    future1.result()
                    future2.result()

                    websocket = self.task_websockets.get(message_id)
                    if websocket and websocket.client_state == WebSocketState.CONNECTED:
                        future = asyncio.run_coroutine_threadsafe(
                            self.websocket_manager.send_agent_progress(
                                step=step,
                                metadata=self.active_tasks[message_id].traceMetadata,
                                websocket=websocket,
                            ),
                            loop,
                        )
                        future.result()

                if self.active_tasks[message_id].traceMetadata.completed:
                    raise AgentStopException("Task not completed")

                step_filename = f"{message_id}-{memory_step.step_number + 1}"
                screenshot_bytes = agent.desktop.screenshot()
                original_image = Image.open(BytesIO(screenshot_bytes))
                image = compress_image_to_max_size(original_image, max_size_kb=500)
                del original_image

                for previous_memory_step in agent.memory.steps:
                    if isinstance(previous_memory_step, ActionStep):
                        previous_memory_step.observations_images = None
                    elif isinstance(previous_memory_step, TaskStep):
                        previous_memory_step.task_images = None

                memory_step.observations_images = [image.copy()]

                del self.last_screenshot[message_id]
                self.last_screenshot[message_id] = (image, step_filename)

            await self._agent_runner(message_id, step_callback)

        except Exception as e:
            logger.error(f"Error in _agent_processing for {message_id}: {e}")
            try:
                await self.sandbox_service.release_sandbox(message_id)
            except Exception as release_error:
                logger.error(f"Error releasing sandbox: {release_error}")
            raise

    async def stop_task(self, trace_id: str):
        """Stop a task."""
        if trace_id in self.active_tasks:
            await self.active_tasks[trace_id].update_trace_metadata(completed=True)

    async def cleanup_tasks_for_websocket(self, websocket: WebSocket):
        """Clean up all tasks associated with a disconnected websocket."""
        tasks_to_cleanup = []

        async with self._lock:
            for message_id, ws in list(self.task_websockets.items()):
                if ws == websocket:
                    tasks_to_cleanup.append(message_id)
                    logger.info(f"Marking task {message_id} for cleanup")
                    del self.task_websockets[message_id]

        for message_id in tasks_to_cleanup:
            try:
                if message_id in self.active_tasks:
                    await self.active_tasks[message_id].update_trace_metadata(
                        completed=True
                    )
                    logger.info(f"Stopped task {message_id}")

                await self.sandbox_service.release_sandbox(message_id)
                logger.info(f"Released sandbox for task {message_id}")

            except Exception as e:
                logger.error(f"Error cleaning up task {message_id}: {e}")

    async def cleanup(self):
        """Cleanup method called during service shutdown."""
        logger.info("AgentService cleanup complete")
