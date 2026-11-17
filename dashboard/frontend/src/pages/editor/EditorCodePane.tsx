import * as React from 'react'
import Editor, { type BeforeMount, type Monaco, type OnMount } from '@monaco-editor/react'
import type * as monacoNs from 'monaco-editor'
import { DSL_LANGUAGE_ID, diagnosticsToMarkers, registerDSLLanguage } from '@/lib/dslLanguage'
import { useDSLStore } from '@/stores/dslStore'
import type { Diagnostic } from '@/types/dsl'
import { Alert, Skeleton, Textarea } from '../../ui'
import {
  defineEditorTheme,
  editorBaseOptions,
  useEditorMonacoTheme,
  useMonacoLoader,
} from './editorMonaco'

export interface CodePaneHandle {
  /** Scrolls a line range into view, selects its first line and optionally focuses the editor. */
  reveal: (line: number, endLine?: number, column?: number, focus?: boolean) => void
}

interface EditorCodePaneProps {
  value: string
  diagnostics: Diagnostic[]
  readOnly?: boolean
  onChange: (value: string) => void
  onCursor: (position: { line: number; column: number }) => void
  onCompile: () => void
  onFormat: () => void
  onReady?: () => void
}

const revealCss = `.vsr-dsl-editor .vsr-editor-reveal { background: color-mix(in oklab, var(--primary) 9%, transparent); transition: background 300ms; }`

function CodeSkeleton() {
  return (
    <div className="flex h-full flex-col gap-2.5 px-12 py-4" aria-hidden>
      {[38, 62, 54, 70, 30, 58, 46, 66, 40].map((width, index) => (
        <Skeleton key={index} className="h-3" style={{ width: `${width}%` }} />
      ))}
    </div>
  )
}

/** The Signal DSL source in Monaco, with compiler markers and editor shortcuts. */
export const EditorCodePane = React.forwardRef<CodePaneHandle, EditorCodePaneProps>(
  function EditorCodePane(
    { value, diagnostics, readOnly = false, onChange, onCursor, onCompile, onFormat, onReady },
    ref,
  ) {
    const [monaco, setMonaco] = React.useState<Monaco | null>(null)
    const editorRef = React.useRef<monacoNs.editor.IStandaloneCodeEditor | null>(null)
    const highlight = React.useRef<monacoNs.editor.IEditorDecorationsCollection | null>(null)
    const actions = React.useRef({ onCompile, onFormat, onCursor })
    actions.current = { onCompile, onFormat, onCursor }
    const theme = useEditorMonacoTheme(monaco)
    const loadState = useMonacoLoader()
    const fallback = React.useRef<HTMLTextAreaElement>(null)

    React.useImperativeHandle(
      ref,
      () => ({
        reveal(line, endLine = line, column = 1, focus = true) {
          const editor = editorRef.current
          const api = monaco
          if (!editor || !api) {
            const area = fallback.current
            if (!area) return
            const offset = area.value
              .split('\n')
              .slice(0, line - 1)
              .join('\n').length
            area.setSelectionRange(offset ? offset + 1 : 0, offset ? offset + 1 : 0)
            if (focus) area.focus()
            return
          }
          editor.revealLinesInCenterIfOutsideViewport(line, Math.max(line, endLine))
          editor.setPosition({ lineNumber: line, column })
          highlight.current?.clear()
          highlight.current = editor.createDecorationsCollection([
            {
              range: new api.Range(line, 1, Math.max(line, endLine), 1),
              options: { isWholeLine: true, className: 'vsr-editor-reveal' },
            },
          ])
          window.setTimeout(() => highlight.current?.clear(), 1400)
          if (focus) editor.focus()
        },
      }),
      [monaco],
    )

    React.useEffect(() => {
      const editor = editorRef.current
      const model = editor?.getModel()
      if (!monaco || !model) return
      monaco.editor.setModelMarkers(model, 'signal-dsl', diagnosticsToMarkers(monaco, diagnostics))
    }, [diagnostics, monaco])

    const beforeMount: BeforeMount = React.useCallback((api) => {
      registerDSLLanguage(
        api,
        () => useDSLStore.getState().symbols,
        () => useDSLStore.getState().diagnostics,
      )
      // The first paint already uses the token theme; the hook refreshes it on mode changes.
      defineEditorTheme(api, document.documentElement.classList.contains('dark') ? 'dark' : 'light')
    }, [])

    const onMount: OnMount = React.useCallback(
      (editor, api) => {
        editorRef.current = editor
        setMonaco(api)
        editor.addAction({
          id: 'vsr-dsl-compile',
          label: 'Compile DSL',
          keybindings: [api.KeyMod.CtrlCmd | api.KeyCode.Enter],
          run: () => actions.current.onCompile(),
        })
        editor.addAction({
          id: 'vsr-dsl-format',
          label: 'Format DSL',
          keybindings: [api.KeyMod.CtrlCmd | api.KeyMod.Shift | api.KeyCode.KeyF],
          run: () => actions.current.onFormat(),
        })
        editor.onDidChangeCursorPosition((event) =>
          actions.current.onCursor({
            line: event.position.lineNumber,
            column: event.position.column,
          }),
        )
        onReady?.()
      },
      [onReady],
    )

    React.useEffect(() => {
      if (loadState === 'failed') onReady?.()
    }, [loadState, onReady])

    if (loadState === 'failed') {
      return (
        <div className="flex h-full min-h-0 flex-col gap-2 p-3">
          <Alert tone="warning" title="The code editor could not load">
            It is served from a CDN this browser cannot reach. You can still edit the DSL as plain
            text; validation and the problems panel keep working.
          </Alert>
          <Textarea
            ref={fallback}
            aria-label="Editor content"
            value={value}
            readOnly={readOnly}
            spellCheck={false}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                event.preventDefault()
                onCompile()
              }
            }}
            className="min-h-0 flex-1 resize-none font-mono text-[12.5px] leading-5"
          />
        </div>
      )
    }

    return (
      <div className="vsr-dsl-editor relative h-full min-h-0">
        <style>{revealCss}</style>
        {loadState === 'loading' ? <CodeSkeleton /> : null}
        {loadState === 'ready' ? (
          <Editor
            height="100%"
            language={DSL_LANGUAGE_ID}
            value={value}
            theme={theme}
            loading={<CodeSkeleton />}
            beforeMount={beforeMount}
            onMount={onMount}
            onChange={(next) => onChange(next ?? '')}
            options={{
              ...editorBaseOptions(),
              readOnly,
              suggest: { showKeywords: true, showSnippets: true },
              quickSuggestions: { strings: false, other: true, comments: false },
            }}
          />
        ) : null}
      </div>
    )
  },
)
