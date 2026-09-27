export const AGENT_SKILL_PATH = '/install/agent/vllm-sr/SKILL.md'
export const AGENT_SKILL_URL = `https://vllm-sr.ai${AGENT_SKILL_PATH}`
export const AGENT_INSTALL_DOC_PATH = '/docs/installation/agent'

export const CURL_INSTALL_COMMAND = 'curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel stable'

export const PIP_INSTALL_COMMAND = `python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade vllm-sr`

export const UV_INSTALL_COMMAND = 'uv tool install vllm-sr'

export const AGENT_INSTALL_PROMPT = `Install, configure, and verify vLLM Semantic Router on this machine by following the official skill: ${AGENT_SKILL_URL}`
