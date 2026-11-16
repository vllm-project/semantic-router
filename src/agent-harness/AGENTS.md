# Agent harness plugins

- Each folder here is a plugin for one agent harness. `claude-code/` for
  Claude Code and `codex/` for Codex CLI bring the decision model in, as
  clients of `vllm-srun serve` and its public API. `vscode/` is a VS Code
  extension that puts a Router gateway's models in the chat model picker, as
  a client of the gateway's inference API. None imports router or runtime code.
- Each plugin folder is self-contained. The tools install one folder or
  one package and nothing beside it: Claude Code copies only the plugin folder
  and refuses paths that leave it, and a package for another tool carries only
  its own files. Do not import from a sibling folder or from `..`.
- What a tool must find at the repository root stays there and points into
  this folder: `.claude-plugin/marketplace.json` for Claude Code and
  `.agents/plugins/marketplace.json` for Codex.
- Each plugin has its own `OWNER`, `AGENTS.md` and `DEVELOPMENT.md`, and
  its checks are named in `tools/agent/domains.yaml`.
