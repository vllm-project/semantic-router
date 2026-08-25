import { useCallback, useMemo, useState } from 'react'

import ProductIcon from '@/components/ProductIcon'
import { copyText } from '@/utils/clipboard'

import styles from './BuilderPage.module.css'

interface BuilderOutputPanelProps {
  open: boolean
  width: number
  recipeDocument: string
  dslSource: string
  compileError: string | null
  onDragStart: (event: React.MouseEvent) => void
  onOpen: () => void
  onClose: () => void
}

export function BuilderOutputPanel({
  open,
  width,
  recipeDocument,
  dslSource,
  compileError,
  onDragStart,
  onOpen,
  onClose,
}: BuilderOutputPanelProps) {
  const [tab, setTab] = useState<'recipe' | 'dsl'>('recipe')
  const [copied, setCopied] = useState(false)
  const content = useMemo(
    () => (tab === 'recipe' ? recipeDocument : dslSource),
    [dslSource, recipeDocument, tab],
  )
  const copy = useCallback(async () => {
    if (!(await copyText(content))) return
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }, [content])

  if (!open) {
    return (
      <button
        type="button"
        className={styles.outputPanelToggle}
        onClick={onOpen}
        aria-label="Show Recipe output"
      >
        <ProductIcon name="chevron-left" />
      </button>
    )
  }

  return (
    <>
      <div
        className={styles.resizeHandle}
        onMouseDown={onDragStart}
        role="separator"
        aria-orientation="vertical"
      >
        <div className={styles.resizeHandleLine} />
      </div>
      <aside className={styles.outputPanel} style={{ width }} aria-label="Recipe output">
        <div className={styles.outputPanelTabs}>
          <button
            type="button"
            className={tab === 'recipe' ? styles.outputPanelTabActive : styles.outputPanelTab}
            onClick={() => setTab('recipe')}
          >
            Recipe
          </button>
          <button
            type="button"
            className={tab === 'dsl' ? styles.outputPanelTabActive : styles.outputPanelTab}
            onClick={() => setTab('dsl')}
          >
            DSL
          </button>
          <div className={styles.builderOutputActions}>
            {content ? (
              <button
                type="button"
                className={styles.outputPanelCopyBtn}
                onClick={copy}
                aria-live="polite"
              >
                <ProductIcon name={copied ? 'check' : 'copy'} /> {copied ? 'Copied' : 'Copy'}
              </button>
            ) : null}
            <button
              type="button"
              className={styles.outputPanelCloseBtn}
              onClick={onClose}
              aria-label="Close Recipe output"
            >
              <ProductIcon name="close" />
            </button>
          </div>
        </div>
        <div className={styles.outputPanelContent}>
          {compileError ? <div className={styles.outputPanelError}>{compileError}</div> : null}
          {content ? (
            <pre className={styles.outputPanelCode}>{content}</pre>
          ) : (
            <div className={styles.emptyState}>Compile to preview this Recipe.</div>
          )}
        </div>
      </aside>
    </>
  )
}
