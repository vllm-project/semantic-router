# Agent harness plugins

- Each folder here is a plugin that brings the decision model into one
  agent harness: `claude-code/` for Claude Code, `codex/` for Codex CLI. Each is a client of
  `vllm-srun serve` and its public API; none imports router or runtime code.
- Each plugin folder is self-contained. The tools install one folder or
  one package and nothing beside it: Claude Code copies only the plugin folder
  and refuses paths that leave it, and a package for another tool carries only
  its own files. Do not import from a sibling folder or from `..`.
- What a tool must find at the repository root stays there and points into
  this folder: `.claude-plugin/marketplace.json` for Claude Code and
  `.agents/plugins/marketplace.json` for Codex.
- Each plugin has its own `OWNER`, `AGENTS.md` and `DEVELOPMENT.md`, and
  its checks are named in `tools/agent/domains.yaml`.
