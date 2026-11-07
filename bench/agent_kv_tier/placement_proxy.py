"""OpenAI-compatible reverse proxy that spreads requests over vLLM replicas with one placement policy.

It measures what keeping an agent session on one replica is worth against the two policies the Router's
`lb_policy` offers for a model's backends today.

- round_robin: the Router's default.
- least_request: the replica with the fewest requests in flight.
- session: a hash of the X-Session-ID header picks the replica, so a session's turns reuse its prefix cache.
- sticky: a new session goes to the replica whose sessions, active in the last `sticky_idle` seconds, sent the
  fewest prompt bytes on their latest request, and then stays there.

Responses stream through unchanged.
Usage: placement_proxy.py POLICY PORT BACKEND_URL[,BACKEND_URL...]
"""

import asyncio
import hashlib
import itertools
import sys
import time

POLICIES = ("round_robin", "least_request", "session", "sticky")
STICKY_IDLE = 300.0


def session_hash(session_id, replicas):
    return int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % replicas


class Placement:
    def __init__(self, policy, replicas, sticky_idle=STICKY_IDLE, clock=time.monotonic):
        if policy not in POLICIES:
            raise ValueError(
                f"unknown policy {policy!r}, expected one of {', '.join(POLICIES)}"
            )
        self.policy = policy
        self.replicas = replicas
        self.sticky_idle = sticky_idle
        self.clock = clock
        self.inflight = [0] * replicas
        self._turn = itertools.cycle(range(replicas))
        self.sessions = {}  # session id -> [replica, last seen, latest prompt bytes]

    def pick(self, session_id, prompt_bytes):
        if self.policy == "sticky" and session_id:
            return self._sticky(session_id, prompt_bytes)
        if self.policy == "session" and session_id:
            return session_hash(session_id, self.replicas)
        if self.policy == "least_request":
            return min(range(self.replicas), key=lambda i: (self.inflight[i], i))
        return next(self._turn)

    def _sticky(self, session_id, prompt_bytes):
        now = self.clock()
        if session_id not in self.sessions:
            load = [0] * self.replicas
            for replica, seen, last in self.sessions.values():
                if now - seen < self.sticky_idle:
                    load[replica] += last
            self.sessions[session_id] = [
                min(range(self.replicas), key=lambda i: (load[i], i)),
                now,
                0,
            ]
        self.sessions[session_id][1:] = [now, prompt_bytes]
        return self.sessions[session_id][0]

    def touch(self, session_id):
        """A long prefill does not make its session look idle."""
        if session_id in self.sessions:
            self.sessions[session_id][1] = self.clock()


def serve(policy, port, backends):
    # Lazy: the placement rule and its tests need only the standard library.
    import aiohttp  # noqa: PLC0415
    from aiohttp import web  # noqa: PLC0415

    placement = Placement(policy, len(backends))

    async def handle(request):
        body = await request.read()
        session_id = request.headers.get("X-Session-ID")
        i = placement.pick(session_id, len(body))
        placement.inflight[i] += 1
        try:
            headers = {
                "Content-Type": request.headers.get("Content-Type", "application/json")
            }
            async with request.app["client"].request(
                request.method,
                backends[i] + request.path_qs,
                data=body,
                headers=headers,
            ) as upstream:
                resp = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        finally:
            placement.inflight[i] -= 1
            placement.touch(session_id)

    async def main():
        app = web.Application(client_max_size=1 << 30)
        app["client"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        app.router.add_route("*", "/{tail:.*}", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        print("proxy", policy, port, backends, flush=True)
        await asyncio.Event().wait()

    asyncio.run(main())


if __name__ == "__main__":
    serve(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","))
