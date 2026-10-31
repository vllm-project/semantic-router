"""Harbor's Codex agent with Router entrypoints and the protocol-guide settings.

Harbor points ``codex exec --model`` at the segment after the last slash and
writes ``openai_base_url`` into ``$CODEX_HOME/config.toml``. This subclass keeps
the full Router entrypoint on that command and forces ``web_search = "disabled"``
and ``features.multi_agent = false`` on the config Harbor uploads.
"""

from __future__ import annotations

from harbor.agents.installed.codex import Codex

from .codex_run import apply_router_model, merge_router_settings


class RouterCodex(Codex):
    def _build_effective_config(self, openai_base_url: str | None = None) -> dict:
        config = super()._build_effective_config(openai_base_url)
        return merge_router_settings(config, self.model_name or "")

    async def exec_as_agent(self, environment, command, env=None, **kwargs):
        if isinstance(command, str) and self.model_name:
            command = apply_router_model(command, self.model_name)
        return await super().exec_as_agent(environment, command, env=env, **kwargs)
