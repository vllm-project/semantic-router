"""Harbor's Claude Code agent with its model aliases pointed at Router entrypoints.

With a custom ANTHROPIC_BASE_URL, Harbor's ClaudeCode sets every model alias and
CLAUDE_CODE_SUBAGENT_MODEL to the run's model. This subclass replaces them on
every command it runs in the task container, from LANE_<variable> entries that
``run.py`` puts in the Harbor process environment, so a tier or the subagent
default can go to another entrypoint. An entry set to the empty string removes
the variable, which leaves the choice to Claude Code (a subagent then inherits
its parent's model).
"""

import os

from harbor.agents.installed.claude_code import ClaudeCode

from .run import apply_lane_env


class RouterClaudeCode(ClaudeCode):
    async def exec_as_agent(self, environment, command, env=None, **kwargs):
        if env is not None:
            env = apply_lane_env(env, os.environ)
        return await super().exec_as_agent(environment, command, env=env, **kwargs)
