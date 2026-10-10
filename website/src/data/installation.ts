export const AGENT_SKILL_PATH = '/install/agent/vllm-sr/SKILL.md'
export const AGENT_SKILL_URL = `https://vllm-sr.ai${AGENT_SKILL_PATH}`
export const AGENT_INSTALL_DOC_PATH = '/docs/installation/agent'

// Latest docs describe main, including features not present in the stable CLI.
export const CURL_INSTALL_COMMAND = 'curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev'

export const PIP_INSTALL_COMMAND = `python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade --pre vllm-sr`

export const UV_INSTALL_COMMAND = 'uv tool install --upgrade --prerelease allow vllm-sr'

export const AGENT_INSTALL_PROMPT = `Install the development channel of vLLM Semantic Router, then configure and verify it on this machine by following the official skill: ${AGENT_SKILL_URL}`
