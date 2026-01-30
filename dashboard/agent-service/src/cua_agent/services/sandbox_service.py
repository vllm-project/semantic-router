"""E2B Sandbox service for managing desktop sandboxes."""

import asyncio
import os
import time
from datetime import datetime
from typing import Literal

from e2b_desktop import Sandbox
from pydantic import BaseModel

SANDBOX_TIMEOUT = 600  # E2B sandbox lifetime
SANDBOX_READY_TIMEOUT = 480  # Seconds before a sandbox expires (8 minutes)
SANDBOX_CREATION_THREAD_TIMEOUT = 300  # Timeout for sandbox creation
SANDBOX_KILL_TIMEOUT = 30  # Timeout for sandbox.kill()
WIDTH = 1280
HEIGHT = 960


class SandboxResponse(BaseModel):
    """Response from sandbox operations."""

    model_config = {"arbitrary_types_allowed": True}

    sandbox: Sandbox | None
    state: Literal["creating", "ready", "max_sandboxes_reached"]
    error: str | None = None


class SandboxEntry:
    """Container for sandbox and its metadata."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """Check if sandbox has expired."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age >= SANDBOX_READY_TIMEOUT

    def update_access(self):
        """Update last access time."""
        self.last_accessed = datetime.now()


class SandboxService:
    """
    Sandbox service for managing E2B desktop sandboxes.

    Features:
    - Non-blocking sandbox creation (background tasks)
    - Expiration-based cleanup
    - Pooled sandbox management
    """

    def __init__(self, max_sandboxes: int = 10):
        if not os.getenv("E2B_API_KEY"):
            raise ValueError("E2B_API_KEY is not set")
        self.max_sandboxes = max_sandboxes
        self.sandboxes: dict[str, SandboxEntry] = {}  # Ready sandboxes
        self.pending: set[str] = set()  # Session hashes currently being created
        self.creation_errors: dict[str, str] = {}  # Track creation errors
        self.lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _create_and_setup_sandbox(self) -> Sandbox:
        """Create and setup a sandbox (synchronous operation)."""
        # Template ID for E2B desktop environment
        # This is the official E2B desktop template with browser support
        template_id = os.getenv("E2B_TEMPLATE_ID", "k0wmnzir0zuzye6dndlw")
        api_key = os.getenv("E2B_API_KEY")

        print(f"[Sandbox] Creating sandbox with template={template_id}, api_key={'*' * 8}...{api_key[-4:] if api_key else 'MISSING'}")
        print(f"[Sandbox] Resolution: {WIDTH}x{HEIGHT}, timeout: {SANDBOX_TIMEOUT}s")

        start_time = time.time()
        desktop = Sandbox.create(
            template=template_id,
            api_key=api_key,
            resolution=(WIDTH, HEIGHT),
            dpi=96,
            timeout=SANDBOX_TIMEOUT,
        )
        create_time = time.time() - start_time
        print(f"[Sandbox] Sandbox.create() completed in {create_time:.1f}s, ID: {desktop.sandbox_id}")

        print(f"[Sandbox] Starting stream...")
        desktop.stream.start(require_auth=True)
        print(f"[Sandbox] Stream started, setting up Firefox...")

        # Setup Firefox policies for clean browsing
        setup_cmd = """sudo mkdir -p /usr/lib/firefox-esr/distribution && echo '{"policies":{"OverrideFirstRunPage":"","OverridePostUpdatePage":"","DisableProfileImport":true,"DontCheckDefaultBrowser":true}}' | sudo tee /usr/lib/firefox-esr/distribution/policies.json > /dev/null"""
        desktop.commands.run(setup_cmd)
        time.sleep(3)

        total_time = time.time() - start_time
        print(f"[Sandbox] Setup complete in {total_time:.1f}s total")
        return desktop

    async def acquire_sandbox(self, session_hash: str) -> SandboxResponse:
        """
        Acquire a sandbox for a session.
        Returns immediately - either with ready sandbox, or "creating" if one is being created.
        """
        async with self.lock:
            # Check if we have a valid sandbox for this session
            if session_hash in self.sandboxes:
                entry = self.sandboxes[session_hash]
                if not entry.is_expired():
                    entry.update_access()
                    print(f"Reusing sandbox for session {session_hash}")
                    return SandboxResponse(sandbox=entry.sandbox, state="ready")
                else:
                    print(f"Removing expired sandbox for session {session_hash}")
                    old_entry = self.sandboxes.pop(session_hash)
                    asyncio.create_task(
                        self._kill_sandbox_safe(old_entry.sandbox, session_hash)
                    )

            # Check if already being created
            if session_hash in self.pending:
                print(f"Sandbox for session {session_hash} is already being created")
                if session_hash in self.creation_errors:
                    error_msg = self.creation_errors.pop(session_hash)
                    return SandboxResponse(
                        sandbox=None, state="creating", error=error_msg
                    )
                return SandboxResponse(sandbox=None, state="creating")

            # Check for previous creation error
            if session_hash in self.creation_errors:
                error_msg = self.creation_errors.pop(session_hash)
                return SandboxResponse(sandbox=None, state="creating", error=error_msg)

            # Check capacity
            total_count = len(self.sandboxes) + len(self.pending)
            if total_count >= self.max_sandboxes:
                print(
                    f"Sandbox pool at capacity: {len(self.sandboxes)} ready + {len(self.pending)} pending = {total_count}/{self.max_sandboxes}"
                )
                await self._cleanup_expired_internal()
                total_count = len(self.sandboxes) + len(self.pending)
                if total_count >= self.max_sandboxes:
                    return SandboxResponse(sandbox=None, state="max_sandboxes_reached")

            # Mark as pending and start creation in background
            self.pending.add(session_hash)
            print(f"Starting creation of sandbox for session {session_hash}")

        # Start creation in background (non-blocking)
        asyncio.create_task(self._create_sandbox_background(session_hash))
        return SandboxResponse(sandbox=None, state="creating")

    async def _create_sandbox_background(self, session_hash: str):
        """Background task to create a sandbox."""
        desktop = None
        try:
            desktop = await asyncio.wait_for(
                asyncio.to_thread(self._create_and_setup_sandbox),
                timeout=SANDBOX_CREATION_THREAD_TIMEOUT,
            )
            print(
                f"Sandbox created for session {session_hash}, ID: {desktop.sandbox_id}"
            )

            async with self.lock:
                was_released = session_hash not in self.pending
                self.pending.discard(session_hash)

                if was_released:
                    print(
                        f"Session {session_hash} was released during creation, killing sandbox"
                    )
                    asyncio.create_task(self._kill_sandbox_safe(desktop, session_hash))
                    return

                total_count = len(self.sandboxes) + len(self.pending)
                if total_count >= self.max_sandboxes:
                    print(
                        f"Pool at capacity ({total_count}/{self.max_sandboxes}), "
                        f"killing newly created sandbox for {session_hash}"
                    )
                    asyncio.create_task(self._kill_sandbox_safe(desktop, session_hash))
                    return

                self.sandboxes[session_hash] = SandboxEntry(desktop)
                print(f"Sandbox {session_hash} is now ready")

        except asyncio.TimeoutError:
            error_msg = f"Sandbox creation timed out after {SANDBOX_CREATION_THREAD_TIMEOUT} seconds"
            print(f"Error creating sandbox for session {session_hash}: {error_msg}")

            async with self.lock:
                self.pending.discard(session_hash)
                self.creation_errors[session_hash] = error_msg
            if desktop:
                asyncio.create_task(self._kill_sandbox_safe(desktop, session_hash))

        except Exception as e:
            error_msg = str(e)
            import traceback
            error_details = traceback.format_exc()
            print(f"Error creating sandbox for session {session_hash}: {error_msg}")
            print(f"Full traceback: {error_details}")

            async with self.lock:
                self.pending.discard(session_hash)
                self.creation_errors[session_hash] = error_msg
            if desktop:
                asyncio.create_task(self._kill_sandbox_safe(desktop, session_hash))

    async def release_sandbox(self, session_hash: str):
        """Release a sandbox for a session."""
        sandbox = None
        async with self.lock:
            if session_hash in self.sandboxes:
                entry = self.sandboxes.pop(session_hash)
                sandbox = entry.sandbox
            self.pending.discard(session_hash)
            self.creation_errors.pop(session_hash, None)

        if sandbox:
            await self._kill_sandbox_safe(sandbox, session_hash)
            print(f"Released sandbox for session {session_hash}")

    async def _kill_sandbox_safe(self, sandbox: Sandbox, session_hash: str):
        """Safely kill a sandbox with error handling."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(sandbox.kill),
                timeout=SANDBOX_KILL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            print(
                f"Sandbox kill timed out after {SANDBOX_KILL_TIMEOUT} seconds for session {session_hash}"
            )
        except Exception as e:
            print(f"Error killing sandbox for session {session_hash}: {str(e)}")

    async def _cleanup_expired_internal(self) -> int:
        """Internal cleanup of expired sandboxes (must be called with lock held)."""
        expired = []
        for session_hash, entry in list(self.sandboxes.items()):
            if entry.is_expired():
                expired.append((session_hash, entry.sandbox))
                del self.sandboxes[session_hash]

        for session_hash, sandbox in expired:
            await self._kill_sandbox_safe(sandbox, session_hash)
            print(f"Cleaned up expired sandbox for session {session_hash}")

        return len(expired)

    async def cleanup_expired_ready_sandboxes(self) -> int:
        """Clean up expired ready sandboxes."""
        async with self.lock:
            return await self._cleanup_expired_internal()

    async def get_sandbox_counts(self) -> tuple[int, int]:
        """Get the count of available (ready) and pending sandboxes."""
        async with self.lock:
            available = len(self.sandboxes)
            non_available = len(self.pending)
            return (available, non_available)

    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(60)
                async with self.lock:
                    cleaned = await self._cleanup_expired_internal()
                    if cleaned > 0:
                        print(f"Periodic cleanup: removed {cleaned} expired sandboxes")
                    ready_count = len(self.sandboxes)
                    pending_count = len(self.pending)
                    total = ready_count + pending_count
                    if total > 0:
                        print(
                            f"Sandbox pool: {ready_count} ready, {pending_count} pending, {total}/{self.max_sandboxes} total"
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic cleanup: {str(e)}")

    def start_periodic_cleanup(self):
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            except RuntimeError as e:
                print(f"Warning: Cannot start periodic cleanup (no event loop): {e}")

    def stop_periodic_cleanup(self):
        """Stop the periodic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def cleanup_sandboxes(self):
        """Clean up all sandboxes."""
        async with self.lock:
            sandboxes_to_kill = list(self.sandboxes.values())
            self.sandboxes.clear()

        for entry in sandboxes_to_kill:
            await self._kill_sandbox_safe(entry.sandbox, "cleanup")
