import { useId, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'

import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type { EntrypointConfig } from './configPageSupport'
import styles from './ConfigPageModelUsageDialog.module.css'

type ExampleKind = 'curl' | 'python' | 'javascript'

const copyText = async (value: string) => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }
  const input = document.createElement('textarea')
  input.value = value
  input.style.position = 'fixed'
  input.style.opacity = '0'
  document.body.appendChild(input)
  input.select()
  const copied = document.execCommand('copy')
  document.body.removeChild(input)
  if (!copied) throw new Error('Copy is not available in this browser.')
}

const buildExamples = (origin: string, model: string): Record<ExampleKind, string> => ({
  curl: `curl ${origin}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'`,
  python: `from openai import OpenAI

client = OpenAI(
    base_url="${origin}/v1",
    api_key="not-required",
)

stream = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")`,
  javascript: `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "${origin}/v1",
  apiKey: "not-required",
});

const stream = await client.chat.completions.create({
  model: "${model}",
  messages: [{ role: "user", content: "Hello" }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}`,
})

const EXAMPLE_LANGUAGES: Record<ExampleKind, string> = {
  curl: 'bash',
  python: 'python',
  javascript: 'javascript',
}

interface ConfigPageModelUsageDialogProps {
  entrypoint: EntrypointConfig
  onClose: () => void
}

export default function ConfigPageModelUsageDialog({
  entrypoint,
  onClose,
}: ConfigPageModelUsageDialogProps) {
  const titleId = useId()
  const [active, setActive] = useState<ExampleKind>('curl')
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState('')
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen: true, onClose })
  const model = entrypoint.model_names[0] ?? ''
  const origin = window.location.origin.replace(/\/$/, '')
  const examples = useMemo(() => buildExamples(origin, model), [model, origin])
  const highlightedExample = useMemo(
    () => `\`\`\`${EXAMPLE_LANGUAGES[active]}\n${examples[active]}\n\`\`\``,
    [active, examples],
  )

  const handleCopy = async () => {
    setCopyError('')
    try {
      await copyText(examples[active])
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch (cause) {
      setCopyError(cause instanceof Error ? cause.message : 'Copy failed.')
    }
  }

  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <header className={styles.header}>
          <div>
            <span>Use this model</span>
            <h2 id={titleId}>{model}</h2>
            <p>OpenAI-compatible. Streaming ready.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close">
            <ProductIcon name="close" />
          </button>
        </header>

        <div className={styles.endpoint}>
          <span>Base URL</span>
          <code>{origin}/v1</code>
          <small>No API key required</small>
        </div>

        <div className={styles.tabs} role="tablist" aria-label="Usage examples">
          {(['curl', 'python', 'javascript'] as const).map((kind) => (
            <button
              key={kind}
              type="button"
              role="tab"
              aria-selected={active === kind}
              onClick={() => {
                setActive(kind)
                setCopied(false)
                setCopyError('')
              }}
            >
              {kind === 'javascript' ? 'JavaScript' : kind === 'python' ? 'Python' : 'curl'}
            </button>
          ))}
        </div>

        <div className={styles.codeShell}>
          <button type="button" className={styles.copyButton} onClick={() => void handleCopy()}>
            <ProductIcon name={copied ? 'check' : 'copy'} />
            {copied ? 'Copied' : 'Copy'}
          </button>
          <div className={styles.codeViewport}>
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {highlightedExample}
            </ReactMarkdown>
          </div>
        </div>
        {copyError ? (
          <p className={styles.copyError} role="alert">
            {copyError}
          </p>
        ) : null}
      </div>
    </div>
  )
}
