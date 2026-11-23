import { useCallback, useState } from 'react'
import { Check, Copy } from 'lucide-react'

const CopyResponseButton = ({ copied, onCopy }: { copied: boolean; onCopy: () => void }) => {
  return (
    <button
      type="button"
      className="grid size-7 place-items-center rounded-md text-subtle-foreground outline-none transition-[color,background-color,transform] duration-150 hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 active:scale-95"
      onClick={onCopy}
      title={copied ? 'Copied!' : 'Copy'}
      aria-label={copied ? 'Copied!' : 'Copy'}
    >
      {copied ? (
        <Check className="size-4 text-success" strokeWidth={2.4} />
      ) : (
        <Copy className="size-4" />
      )}
    </button>
  )
}

export const MessageActionBar = ({ content }: { content: string }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!content) return
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(content)
      } else {
        const textArea = document.createElement('textarea')
        textArea.value = content
        textArea.style.position = 'fixed'
        textArea.style.left = '-9999px'
        document.body.appendChild(textArea)
        textArea.select()
        document.execCommand('copy')
        document.body.removeChild(textArea)
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }, [content])

  return (
    <div className="flex items-center gap-1">
      <CopyResponseButton copied={copied} onCopy={handleCopy} />
    </div>
  )
}
