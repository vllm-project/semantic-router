import React, { useEffect, useState } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import { FiAlertCircle, FiCheck, FiCopy } from 'react-icons/fi'
import duo from './duo.module.css'
import styles from './Terminal.module.css'

type Token = { text: string, kind: 'bin' | 'sub' | 'flag' | 'str' | 'val' | 'op' | 'space' }

// One shell line in Duo's syntax colours on ink: the binary, its subcommand, flags, model ids and URLs, values, pipes.
function tokenize(command: string): Token[] {
  const tokens: Token[] = []
  let expectBinary = true
  let lastBinary = ''
  for (const part of command.split(/(\s+)/)) {
    if (!part) continue
    if (/^\s+$/.test(part)) {
      tokens.push({ text: part, kind: 'space' })
      continue
    }
    if (part === '|' || part === '&&') {
      tokens.push({ text: part, kind: 'op' })
      expectBinary = true
      continue
    }
    if (expectBinary) {
      tokens.push({ text: part, kind: 'bin' })
      lastBinary = part
      expectBinary = false
      continue
    }
    if (part.startsWith('-')) tokens.push({ text: part, kind: 'flag' })
    else if (/^https?:\/\//.test(part) || part.includes('/')) tokens.push({ text: part, kind: 'str' })
    else if (lastBinary === 'vllm-sr' && !tokens.some(t => t.kind === 'sub')) tokens.push({ text: part, kind: 'sub' })
    else tokens.push({ text: part, kind: 'val' })
  }
  return tokens
}

type CopyState = 'idle' | 'copied' | 'error'

/* An ink terminal with a caption bar and a Copy button. A shell command is highlighted; plain text (an agent prompt)
 * is shown as written. The copied text never includes the prompt sign. */
export default function Terminal({ text, caption, shell = true, className }: {
  text: string
  caption: React.ReactNode
  shell?: boolean
  className?: string
}): JSX.Element {
  const [state, setState] = useState<CopyState>('idle')

  useEffect(() => {
    if (state === 'idle') return undefined
    const timer = window.setTimeout(() => setState('idle'), 1800)
    return () => window.clearTimeout(timer)
  }, [state])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setState('copied')
    }
    catch {
      setState('error')
    }
  }
  const label = state === 'copied'
    ? translate({ id: 'homepage.install.copy.copied', message: 'Copied' })
    : state === 'error'
      ? translate({ id: 'homepage.install.copy.error', message: 'Copy failed' })
      : translate({ id: 'homepage.install.copy.aria', message: 'Copy command to clipboard' })

  return (
    <div className={clsx(duo.codeblock, className)}>
      <div className={duo.codeCap}>
        <span>{caption}</span>
        <button
          type="button"
          className={clsx(duo.copy, state === 'copied' && duo.copyDone)}
          onClick={() => {
            void copy()
          }}
          title={label}
          aria-label={label}
        >
          {state === 'copied'
            ? <FiCheck aria-hidden="true" />
            : state === 'error' ? <FiAlertCircle aria-hidden="true" /> : <FiCopy aria-hidden="true" />}
          <span aria-hidden="true">
            {state === 'copied'
              ? <Translate id="homepage.install.copy.copied">Copied</Translate>
              : <Translate id="homepage.install.copy.label">Copy</Translate>}
          </span>
        </button>
      </div>
      <pre>
        <code>
          <span className={styles.prompt} aria-hidden="true">{shell ? '$ ' : '› '}</span>
          {shell
            ? tokenize(text).map((token, index) => (
                token.kind === 'space'
                  ? token.text
                  : <span key={index} className={styles[token.kind]}>{token.text}</span>
              ))
            : text}
        </code>
      </pre>
    </div>
  )
}
