"""Fixed sr-bench terminal agent using Harbor's task environment and verifier."""
import asyncio
import json

from harbor.agents.base import BaseAgent

from .harness_worker import call


class SRBenchTerminalAgent(BaseAgent):
    def __init__(self, *args, max_steps=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    @staticmethod
    def name():
        return "sr-bench-terminal"

    def version(self):
        return "1.0"

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        messages = [{"role": "system", "content": "Complete the task in the provided isolated terminal environment. Use the terminal tool to inspect and modify files. When finished, give a concise final response."}, {"role": "user", "content": instruction}]
        tools = [{"type": "function", "function": {"name": "terminal", "description": "Execute a shell command in the task sandbox.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}}}]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        transcript = self.logs_dir / "trajectory.json"
        for _ in range(self.max_steps):
            response = await asyncio.to_thread(call, messages, "subject", {"tools": tools, "parallel_tool_calls": False})
            item = {"role": "assistant", "content": response["final"] or None}
            if response.get("tool_calls"):
                item["tool_calls"] = response["tool_calls"]
            messages.append(item)
            transcript.write_text(json.dumps(messages, indent=2))
            context.metadata = {"turns": len(messages), "agent": "sr-bench-terminal-v1"}
            if not response.get("tool_calls"):
                return
            for tool in response["tool_calls"]:
                try:
                    args = json.loads(tool["function"]["arguments"])
                    if tool["function"]["name"] != "terminal" or not isinstance(args.get("command"), str):
                        raise ValueError("invalid tool invocation")
                    result = await environment.exec(command=args["command"], timeout_sec=30)
                    output = json.dumps({"stdout": (result.stdout or "")[-24000:], "stderr": (result.stderr or "")[-8000:], "return_code": result.return_code})
                except (ValueError, TimeoutError) as exc:
                    output = type(exc).__name__
                messages.append({"role": "tool", "tool_call_id": tool["id"], "content": output})
                transcript.write_text(json.dumps(messages, indent=2))
        # Task step exhaustion is a model outcome; the actual verifier decides
        # whether work completed before the step cap.
