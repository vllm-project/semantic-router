export interface APIKeyQuickstartSnippets {
  python: string
  javascript: string
  curl: string
}

function sourceString(value: string) {
  return JSON.stringify(value)
}

function shellSingleQuoted(value: string) {
  return `'${value.replace(/'/g, `'"'"'`)}'`
}

export function buildAPIKeyQuickstartSnippets(
  baseURL: string,
  model: string,
  secret: string,
): APIKeyQuickstartSnippets {
  const apiKeyPython = secret ? sourceString(secret) : 'os.environ["VLLM_SR_API_KEY"]'
  const apiKeyJavaScript = secret ? sourceString(secret) : 'process.env.VLLM_SR_API_KEY'
  const payload = JSON.stringify({
    model,
    messages: [{ role: 'user', content: 'Hello' }],
  })
  const authorization = secret
    ? shellSingleQuoted(`Authorization: Bearer ${secret}`)
    : '"Authorization: Bearer $VLLM_SR_API_KEY"'

  return {
    python: `import os\nfrom openai import OpenAI\n\nclient = OpenAI(\n    base_url=${sourceString(baseURL)},\n    api_key=${apiKeyPython},\n)\n\nresponse = client.chat.completions.create(\n    model=${sourceString(model)},\n    messages=[{"role": "user", "content": "Hello"}],\n)\nprint(response.choices[0].message.content)`,
    javascript: `import OpenAI from "openai";\n\nconst client = new OpenAI({\n  baseURL: ${sourceString(baseURL)},\n  apiKey: ${apiKeyJavaScript},\n});\n\nconst response = await client.chat.completions.create({\n  model: ${sourceString(model)},\n  messages: [{ role: "user", content: "Hello" }],\n});\nconsole.log(response.choices[0].message.content);`,
    curl: `curl ${shellSingleQuoted(`${baseURL}/chat/completions`)} \\\n  -H ${authorization} \\\n  -H 'Content-Type: application/json' \\\n  -d ${shellSingleQuoted(payload)}`,
  }
}
