import React, { useEffect, useRef, useCallback, useState } from 'react'
import Editor, { type OnMount, type BeforeMount, type Monaco } from '@monaco-editor/react'
import type * as monacoNs from 'monaco-editor'
import { useDSLStore } from '@/stores/dslStore'
import {
  registerDSLLanguage,
  defineTheme,
  DSL_LANGUAGE_ID,
  diagnosticsToMarkers,
} from '@/lib/dslLanguage'
import styles from './DslEditorPage.module.css'

type OutputTab = 'yaml' | 'crd'

const DEFAULT_DSL = `# Welcome to the Signal DSL Editor
# Press Ctrl+Enter to compile, Ctrl+Shift+F to format
#
# Example:
# SIGNAL keyword math_terms {
#   keywords: ["calculus", "algebra", "geometry"]
#   operator: "OR"
#   case_sensitive: false
# }
#
# ROUTE math_route (description = "Route math queries") {
#   PRIORITY 10
#   WHEN keyword("math_terms")
#   MODEL "gpt-4o"
# }
`

interface DslEditorPageProps {
  /** When embedded in BuilderPage, hide own toolbar and status bar */
  embedded?: boolean
  /** When true, hide the output pane (YAML/CRD) — the parent provides its own */
  hideOutput?: boolean
}

const DslEditorPage: React.FC<DslEditorPageProps> = ({ embedded = false, hideOutput = false }) => {
  const {
    dslSource,
    yamlOutput,
    crdOutput,
    diagnostics,
    symbols,
    wasmReady,
    wasmError,
    loading,
    compileError,
    dirty,
    initWasm,
    setDslSource,
    compile,
    validate,
    format,
    reset,
    importYaml,
  } = useDSLStore()

  const [outputTab, setOutputTab] = useState<OutputTab>('yaml')
  const [copied, setCopied] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [importText, setImportText] = useState('')
  const [importError, setImportError] = useState<string | null>(null)
  const [importUrl, setImportUrl] = useState('')
  const [importUrlLoading, setImportUrlLoading] = useState(false)
  const [diagHeight, setDiagHeight] = useState(160)
  const diagHeightRef = useRef(diagHeight)
  const diagDragging = useRef(false)
  const diagStartY = useRef(0)
  const diagStartH = useRef(0)
  const importTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const editorRef = useRef<monacoNs.editor.IStandaloneCodeEditor | null>(null)
  const monacoRef = useRef<Monaco | null>(null)

  // Keep ref in sync with state
  useEffect(() => { diagHeightRef.current = diagHeight }, [diagHeight])

  // Drag to resize diagnostics panel
  const onDiagDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    diagDragging.current = true
    diagStartY.current = e.clientY
    diagStartH.current = diagHeightRef.current

    const onMove = (ev: MouseEvent) => {
      if (!diagDragging.current) return
      const delta = diagStartY.current - ev.clientY
      setDiagHeight(Math.max(32, Math.min(500, diagStartH.current + delta)))
    }
    const onUp = () => {
      diagDragging.current = false
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [])

  // Initialize WASM on mount
  useEffect(() => {
    initWasm()
  }, [initWasm])

  // Set diagnostics as Monaco markers whenever they change
  useEffect(() => {
    const editor = editorRef.current
    const monaco = monacoRef.current
    if (!editor || !monaco) return

    const model = editor.getModel()
    if (!model) return

    const markers = diagnosticsToMarkers(monaco, diagnostics)
    monaco.editor.setModelMarkers(model, 'signal-dsl', markers)
  }, [diagnostics])

  // Keyboard shortcuts
  useEffect(() => {
    const editor = editorRef.current
    if (!editor) return

    // Ctrl+Enter → Compile
    editor.addAction({
      id: 'dsl-compile',
      label: 'Compile DSL',
      keybindings: [
        (monacoRef.current?.KeyMod.CtrlCmd ?? 0) | (monacoRef.current?.KeyCode.Enter ?? 0),
      ],
      run: () => compile(),
    })

    // Ctrl+Shift+F → Format
    editor.addAction({
      id: 'dsl-format',
      label: 'Format DSL',
      keybindings: [
        (monacoRef.current?.KeyMod.CtrlCmd ?? 0) |
          (monacoRef.current?.KeyMod.Shift ?? 0) |
          (monacoRef.current?.KeyCode.KeyF ?? 0),
      ],
      run: () => format(),
    })
  }, [compile, format, wasmReady])

  // Register theme before Monaco renders so the first paint is already dark
  const handleBeforeMount: BeforeMount = useCallback((monaco) => {
    defineTheme(monaco)
  }, [])

  const handleEditorMount: OnMount = useCallback(
    (editor, monaco) => {
      editorRef.current = editor
      monacoRef.current = monaco

      // Register DSL language with symbol table provider and diagnostics for Quick Fix
      registerDSLLanguage(
        monaco,
        () => useDSLStore.getState().symbols,
        () => useDSLStore.getState().diagnostics,
      )

      // Set model language
      const model = editor.getModel()
      if (model) {
        monaco.editor.setModelLanguage(model, DSL_LANGUAGE_ID)
      }

      // Load default content if empty
      if (!dslSource) {
        setDslSource(DEFAULT_DSL)
      }
    },
    [dslSource, setDslSource],
  )

  const handleEditorChange = useCallback(
    (value: string | undefined) => {
      setDslSource(value ?? '')
    },
    [setDslSource],
  )

  const handleCopyOutput = useCallback(async () => {
    const text = outputTab === 'yaml' ? yamlOutput : crdOutput
    if (!text) return
    try {
      // Try Clipboard API first
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        // Fallback for non-secure contexts (http://localhost etc.)
        const textarea = document.createElement('textarea')
        textarea.value = text
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand('copy')
        document.body.removeChild(textarea)
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Last-resort fallback
      const textarea = document.createElement('textarea')
      textarea.value = text
      textarea.style.position = 'fixed'
      textarea.style.opacity = '0'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }, [outputTab, yamlOutput, crdOutput])

  const handleGoToLine = useCallback((line: number, column: number) => {
    const editor = editorRef.current
    if (!editor) return
    editor.revealLineInCenter(line)
    editor.setPosition({ lineNumber: line, column })
    editor.focus()
  }, [])

  const handleApplyFix = useCallback((line: number, column: number, newText: string) => {
    const editor = editorRef.current
    if (!editor) return
    const model = editor.getModel()
    if (!model) return

    // Find the word/token at the diagnostic position to replace
    const lineContent = model.getLineContent(line)
    let startCol = column
    let endCol = column
    while (startCol > 1 && /[\w\-.]/.test(lineContent[startCol - 2])) startCol--
    while (endCol <= lineContent.length && /[\w\-.]/.test(lineContent[endCol - 1])) endCol++

    editor.executeEdits('quick-fix', [{
      range: {
        startLineNumber: line,
        startColumn: startCol,
        endLineNumber: line,
        endColumn: endCol,
      },
      text: newText,
    }])
    // Trigger re-validation
    const src = model.getValue()
    setDslSource(src)
  }, [setDslSource])

  const handleOpenImport = useCallback(() => {
    setImportText('')
    setImportError(null)
    setImportUrl('')
    setImportUrlLoading(false)
    setShowImportModal(true)
    setTimeout(() => importTextareaRef.current?.focus(), 50)
  }, [])

  const handleImportConfirm = useCallback(() => {
    const yaml = importText.trim()
    if (!yaml) {
      setImportError('Please paste YAML content')
      return
    }
    try {
      importYaml(yaml)
      setShowImportModal(false)
      setImportText('')
      setImportError(null)
    } catch {
      setImportError('Failed to decompile YAML. Make sure it is valid router config YAML.')
    }
  }, [importText, importYaml])

  const handleImportFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target?.result
      if (typeof text === 'string') {
        setImportText(text)
        setImportError(null)
      }
    }
    reader.readAsText(file)
    // Reset file input so same file can be re-selected
    e.target.value = ''
  }, [])

  const handleImportUrl = useCallback(async () => {
    const url = importUrl.trim()
    if (!url) { setImportError('Please enter a URL'); return }
    try {
      new URL(url)
    } catch {
      setImportError('Invalid URL format'); return
    }
    setImportUrlLoading(true)
    setImportError(null)
    try {
      const resp = await fetch('/api/tools/fetch-raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      })
      const data = await resp.json()
      if (data.error) {
        throw new Error(data.error)
      }
      if (!data.content?.trim()) {
        throw new Error('Remote returned empty content')
      }
      setImportText(data.content)
      setImportError(null)
    } catch (err) {
      setImportError(`Failed to fetch: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setImportUrlLoading(false)
    }
  }, [importUrl])

  // Diagnostic counts (3 severity levels per design doc)
  const errorCount = diagnostics.filter((d) => d.level === 'error').length
  const warnCount = diagnostics.filter((d) => d.level === 'warning').length
  const constraintCount = diagnostics.filter((d) => d.level === 'constraint').length

  // Stats for status bar
  const lineCount = dslSource.split('\n').length
  const signalCount = symbols?.signals?.length ?? 0
  const routeCount = symbols?.routes?.length ?? 0
  const isValid = errorCount === 0 && wasmReady

  return (
    <div className={styles.page}>
      {/* Toolbar — hidden when embedded in BuilderPage (parent provides toolbar) */}
      {!embedded && (
      <div className={styles.toolbar}>
        <div className={styles.toolbarTitle}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 3h12M2 8h8M2 13h10" strokeLinecap="round" />
          </svg>
          DSL Editor
          {dirty && <span style={{ color: 'var(--color-text-muted)', fontWeight: 400 }}>(unsaved)</span>}
        </div>

        {/* WASM status */}
        {wasmError ? (
          <span className={styles.statusError}>
            <span className={styles.dot} /> WASM Error
          </span>
        ) : wasmReady ? (
          <span className={styles.statusReady}>
            <span className={styles.dot} /> Ready
          </span>
        ) : (
          <span className={styles.statusLoading}>
            <span className={styles.dotPulse} /> Loading WASM…
          </span>
        )}

        <span className={styles.divider} />

        <button
          className={styles.toolbarBtn}
          onClick={handleOpenImport}
          disabled={!wasmReady}
          title="Import from YAML config"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M8 2v8M5 7l3 3 3-3" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M2 11v2a1 1 0 001 1h10a1 1 0 001-1v-2" strokeLinecap="round" />
          </svg>
          Import YAML
        </button>

        <button
          className={styles.toolbarBtn}
          onClick={format}
          disabled={!wasmReady || !dslSource.trim()}
          title="Format (Ctrl+Shift+F)"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 4h12M2 8h8M2 12h10" strokeLinecap="round" />
          </svg>
          Format
        </button>

        <button
          className={styles.toolbarBtn}
          onClick={validate}
          disabled={!wasmReady || !dslSource.trim()}
          title="Validate DSL"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Validate
        </button>

        <button
          className={styles.toolbarBtnPrimary}
          onClick={compile}
          disabled={!wasmReady || !dslSource.trim() || loading}
          title="Compile (Ctrl+Enter)"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M4 2l8 6-8 6V2z" fill="currentColor" />
          </svg>
          {loading ? 'Compiling…' : 'Compile'}
        </button>

        <span className={styles.divider} />

        <button
          className={styles.toolbarBtnDanger}
          onClick={reset}
          title="Reset editor"
        >
          Reset
        </button>
      </div>
      )}

      {/* Split panes */}
      <div className={styles.splitContainer}>
        {/* Left: Editor */}
        <div className={styles.editorPane} style={{ position: 'relative', ...(hideOutput ? { borderRight: 'none' } : {}) }}>
          <div className={styles.paneHeader}>
            <span className={styles.paneTitle}>Source</span>
          </div>

          {/* WASM loading overlay */}
          {!wasmReady && !wasmError && (
            <div className={styles.wasmOverlay}>
              <div className={styles.spinner} />
              Loading Signal Compiler…
            </div>
          )}

          <Editor
            height="100%"
            defaultLanguage={DSL_LANGUAGE_ID}
            defaultValue={dslSource || DEFAULT_DSL}
            value={dslSource}
            theme="signal-dsl-dark"
            onChange={handleEditorChange}
            beforeMount={handleBeforeMount}
            onMount={handleEditorMount}
            options={{
              fontSize: 13,
              fontFamily: 'var(--font-mono)',
              minimap: { enabled: false },
              lineNumbers: 'on',
              scrollBeyondLastLine: false,
              wordWrap: 'on',
              tabSize: 2,
              insertSpaces: true,
              automaticLayout: true,
              bracketPairColorization: { enabled: true },
              guides: { indentation: true, bracketPairs: true },
              suggest: { showKeywords: true, showSnippets: true },
              quickSuggestions: { strings: false, other: true, comments: false },
              padding: { top: 8, bottom: 8 },
              renderLineHighlight: 'line',
              smoothScrolling: true,
              cursorBlinking: 'smooth',
              cursorSmoothCaretAnimation: 'on',
            }}
          />

          {/* Diagnostics panel — always visible, resizable */}
          <div className={styles.diagnosticsPanel} style={{ height: diagHeight }}>
            <div className={styles.diagResizeHandle} onMouseDown={onDiagDragStart} />
            <div className={styles.diagnosticsHeader}>
              <span className={styles.diagnosticsTitle}>Problems</span>
              {errorCount > 0 ? (
                <span className={styles.diagCountError}>{errorCount} error{errorCount !== 1 ? 's' : ''}</span>
              ) : (
                <span className={styles.diagCountOk}>0 errors</span>
              )}
              {warnCount > 0 ? (
                <span className={styles.diagCountWarn}>{warnCount} warning{warnCount !== 1 ? 's' : ''}</span>
              ) : (
                <span className={styles.diagCountOk}>0 warnings</span>
              )}
              {constraintCount > 0 ? (
                <span className={styles.diagCountConstraint}>{constraintCount} constraint{constraintCount !== 1 ? 's' : ''}</span>
              ) : (
                <span className={styles.diagCountOk}>0 constraints</span>
              )}
            </div>
            {diagnostics.length > 0 && (
              <ul className={styles.diagnosticsList}>
                {diagnostics.map((d, i) => (
                  <li
                    key={i}
                    className={styles.diagnosticItem}
                    onClick={() => handleGoToLine(d.line, d.column)}
                  >
                    {d.level === 'error' ? (
                      <svg className={styles.diagIconError} viewBox="0 0 16 16" fill="currentColor">
                        <circle cx="8" cy="8" r="7" />
                        <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" />
                      </svg>
                    ) : d.level === 'warning' ? (
                      <svg className={styles.diagIconWarn} viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 1l7 13H1L8 1z" />
                        <path d="M8 6v3M8 11v1" stroke="#000" strokeWidth="1.5" strokeLinecap="round" />
                      </svg>
                    ) : (
                      <svg className={styles.diagIconConstraint} viewBox="0 0 16 16" fill="currentColor">
                        <circle cx="8" cy="8" r="7" />
                        <path d="M8 5v4M8 11v1" stroke="#000" strokeWidth="1.5" strokeLinecap="round" />
                      </svg>
                    )}
                    <span className={styles.diagMessage}>
                      {d.message}
                      {d.fixes?.map((fix, fi) => (
                        <button
                          key={fi}
                          className={styles.diagFixBtn}
                          onClick={(e) => { e.stopPropagation(); handleApplyFix(d.line, d.column, fix.newText) }}
                          title={fix.description}
                        >
                          Fix
                        </button>
                      ))}
                    </span>
                    <span className={styles.diagLocation}>Ln {d.line}, Col {d.column}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Right: Output (hidden when parent provides output panel) */}
        {!hideOutput && (
        <div className={styles.outputPane}>
          <div className={styles.outputTabs}>
            <button
              className={outputTab === 'yaml' ? styles.outputTabActive : styles.outputTab}
              onClick={() => setOutputTab('yaml')}
            >
              YAML Output
            </button>
            <button
              className={outputTab === 'crd' ? styles.outputTabActive : styles.outputTab}
              onClick={() => setOutputTab('crd')}
            >
              CRD Output
            </button>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', paddingRight: 'var(--spacing-sm)' }}>
              {(outputTab === 'yaml' ? yamlOutput : crdOutput) && (
                <button className={styles.toolbarBtn} onClick={handleCopyOutput} title="Copy to clipboard">
                  {copied ? (
                    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="var(--color-success)" strokeWidth="2">
                      <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <rect x="5" y="5" width="9" height="9" rx="1" />
                      <path d="M2 11V2h9" strokeLinecap="round" />
                    </svg>
                  )}
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              )}
            </div>
          </div>

          <div className={styles.outputContent}>
            {compileError && (
              <div style={{ padding: 'var(--spacing-md)', color: 'var(--color-danger)', fontSize: 'var(--text-xs)', fontFamily: 'var(--font-mono)' }}>
                {compileError}
              </div>
            )}

            {outputTab === 'yaml' && yamlOutput ? (
              <pre className={styles.outputCode}>{yamlOutput}</pre>
            ) : outputTab === 'crd' && crdOutput ? (
              <pre className={styles.outputCode}>{crdOutput}</pre>
            ) : (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>⚡</div>
                <div>Write DSL and press <strong>Compile</strong> to see output</div>
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
                  Ctrl+Enter to compile · Ctrl+Shift+F to format
                </div>
              </div>
            )}
          </div>
        </div>
        )}
      </div>

      {/* Status Bar — hidden when embedded in BuilderPage (parent provides status bar) */}
      {!embedded && (
      <div className={styles.statusBar}>
        <div className={`${styles.statusItem} ${isValid ? styles.statusValid : styles.statusInvalid}`}>
          {isValid ? (
            <svg className={styles.statusCheckmark} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg className={styles.statusCheckmark} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
          {isValid ? 'Config valid' : `${errorCount} error${errorCount !== 1 ? 's' : ''}`}
        </div>
        <div className={styles.statusItem}>Signals: {signalCount}</div>
        <div className={styles.statusItem}>Routes: {routeCount}</div>
        <div className={styles.statusItem}>Lines: {lineCount}</div>
      </div>
      )}

      {/* Hidden file input for YAML import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml,.json"
        style={{ display: 'none' }}
        onChange={handleImportFile}
      />

      {/* Import YAML Modal */}
      {showImportModal && (
        <div className={styles.modalOverlay} onClick={() => setShowImportModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3 className={styles.modalTitle}>Import YAML Config</h3>
              <button className={styles.modalClose} onClick={() => setShowImportModal(false)}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <div className={styles.modalBody}>
              <p className={styles.modalHint}>
                Paste a router config YAML below, load from a file, or fetch from a URL. It will be decompiled into DSL.
              </p>
              <div className={styles.importUrlRow}>
                <input
                  className={styles.importUrlInput}
                  type="url"
                  value={importUrl}
                  onChange={(e) => { setImportUrl(e.target.value); setImportError(null) }}
                  placeholder="https://example.com/config.yaml"
                  onKeyDown={(e) => { if (e.key === 'Enter') handleImportUrl() }}
                />
                <button
                  className={styles.toolbarBtn}
                  onClick={handleImportUrl}
                  disabled={importUrlLoading || !importUrl.trim()}
                >
                  {importUrlLoading ? (
                    <>
                      <span className={styles.dotPulse} />
                      Fetching…
                    </>
                  ) : (
                    <>
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M6 2a4 4 0 100 8 4 4 0 000-8z" />
                        <path d="M2 6h8M6 2v8" strokeLinecap="round" />
                        <path d="M14 14l-3.5-3.5" strokeLinecap="round" />
                      </svg>
                      Fetch
                    </>
                  )}
                </button>
              </div>
              <textarea
                ref={importTextareaRef}
                className={styles.importTextarea}
                value={importText}
                onChange={(e) => { setImportText(e.target.value); setImportError(null) }}
                placeholder="Paste YAML config here..."
                spellCheck={false}
              />
              {importError && <div className={styles.importError}>{importError}</div>}
            </div>
            <div className={styles.modalFooter}>
              <button
                className={styles.toolbarBtn}
                onClick={() => fileInputRef.current?.click()}
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 14h12M8 2v9M5 5l3-3 3 3" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Load File
              </button>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 'var(--spacing-sm)' }}>
                <button className={styles.toolbarBtn} onClick={() => setShowImportModal(false)}>
                  Cancel
                </button>
                <button
                  className={styles.toolbarBtnPrimary}
                  onClick={handleImportConfirm}
                  disabled={!importText.trim()}
                >
                  Import
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DslEditorPage
