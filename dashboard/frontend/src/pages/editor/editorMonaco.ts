import * as React from 'react'
import type * as monacoNs from 'monaco-editor'
import { useTheme } from '../../shell/theme'
import { createMonacoLoad } from './editorMonacoLoad'

const monacoLoad = createMonacoLoad(async (fail) => {
  const { installMonaco } = await import('./monacoSetup')
  await installMonaco(fail)
})

/**
 * Loads the bundled Monaco once a view that shows it is `enabled`. If its code
 * or worker fails, the editors switch to their plain-text fallbacks for the
 * rest of the page.
 */
export function useMonacoLoader(enabled = true): 'loading' | 'ready' | 'failed' {
  const state = React.useSyncExternalStore(monacoLoad.subscribe, monacoLoad.getState)
  React.useEffect(() => {
    if (enabled) void monacoLoad.start()
  }, [enabled])
  return state === 'idle' ? 'loading' : state
}

/** Turns anything Monaco throws while mounting into the plain-text fallback, not a route crash. */
export class MonacoBoundary extends React.Component<
  { children: React.ReactNode },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: unknown) {
    // This boundary already renders nothing, so even a cancellation-shaped throw must fall back.
    monacoLoad.fail(error instanceof Error ? error : new Error(String(error)))
  }

  render() {
    return this.state.failed ? null : this.props.children
  }
}

type MonacoApi = typeof monacoNs
type Mode = 'light' | 'dark'

// Monaco only accepts hex colors, so its themes are computed from the design
// tokens at runtime: each token is resolved on a probe element and painted to
// a 1px canvas to read it back as sRGB.
const TOKENS = [
  '--surface',
  '--surface-subtle',
  '--surface-sunken',
  '--popover',
  '--foreground',
  '--muted-foreground',
  '--subtle-foreground',
  '--border',
  '--border-strong',
  '--primary',
  '--brand-400',
  '--destructive',
  '--warning',
  '--info',
  '--success',
  '--chart-1',
  '--chart-2',
  '--chart-3',
  '--chart-4',
  '--chart-5',
  '--chart-6',
  '--chart-7',
] as const

type Token = (typeof TOKENS)[number]
type Rgba = [number, number, number, number]

let canvasContext: CanvasRenderingContext2D | null = null

function toRgba(color: string): Rgba {
  canvasContext ??= Object.assign(document.createElement('canvas'), {
    width: 1,
    height: 1,
  }).getContext('2d', { willReadFrequently: true })
  if (!canvasContext || !color) return [0, 0, 0, 255]
  canvasContext.clearRect(0, 0, 1, 1)
  canvasContext.fillStyle = color
  canvasContext.fillRect(0, 0, 1, 1)
  const [r, g, b, a] = canvasContext.getImageData(0, 0, 1, 1).data
  return [r, g, b, a]
}

function hex([r, g, b, a]: Rgba, alpha?: number): string {
  const channels = [r, g, b, Math.round((alpha ?? a / 255) * 255)]
  return `#${channels.map((value) => value.toString(16).padStart(2, '0')).join('')}`
}

const rgb = (color: Rgba) => hex(color).slice(1, 7)

function mix(a: Rgba, b: Rgba, amount: number): Rgba {
  return [0, 1, 2, 3].map((index) => Math.round(a[index] + (b[index] - a[index]) * amount)) as Rgba
}

// The `.dark` rule applies to any element, so a probe resolves dark tokens in
// either mode; light tokens are only exact while the document is light, which
// is why the active theme is refreshed whenever the mode changes.
function readTokens(mode: Mode): Record<Token, Rgba> {
  const probe = document.createElement('div')
  probe.style.display = 'none'
  if (mode === 'dark') probe.className = 'dark'
  document.body.appendChild(probe)
  const style = getComputedStyle(probe)
  const tokens = Object.fromEntries(
    TOKENS.map((token) => [token, toRgba(style.getPropertyValue(token).trim())]),
  ) as Record<Token, Rgba>
  probe.remove()
  return tokens
}

function themeFor(mode: Mode): monacoNs.editor.IStandaloneThemeData {
  const t = readTokens(mode)
  // Chart tones are tuned for fills; on a light surface they need more ink to read as text.
  const ink = (token: Token) =>
    rgb(mode === 'light' ? mix(t[token], t['--foreground'], 0.24) : t[token])
  return {
    base: mode === 'dark' ? 'vs-dark' : 'vs',
    inherit: true,
    rules: [
      { token: '', foreground: rgb(t['--foreground']) },
      { token: 'comment', foreground: rgb(t['--subtle-foreground']), fontStyle: 'italic' },
      { token: 'keyword', foreground: ink('--chart-3'), fontStyle: 'bold' },
      { token: 'keyword.operator', foreground: ink('--chart-3') },
      { token: 'keyword.yaml', foreground: ink('--chart-7'), fontStyle: '' },
      { token: 'type', foreground: ink('--chart-7') },
      { token: 'type.yaml', foreground: ink('--chart-1') },
      { token: 'type.plugin', foreground: ink('--chart-5') },
      { token: 'type.algorithm', foreground: ink('--chart-4') },
      { token: 'variable.field', foreground: ink('--chart-1') },
      { token: 'string', foreground: ink('--chart-6') },
      { token: 'string.quote', foreground: ink('--chart-6') },
      { token: 'string.escape', foreground: ink('--chart-2') },
      { token: 'number', foreground: ink('--chart-7') },
      { token: 'number.float', foreground: ink('--chart-7') },
      { token: 'constant.language', foreground: ink('--chart-7') },
      { token: 'identifier', foreground: rgb(t['--foreground']) },
      { token: 'operators', foreground: rgb(t['--subtle-foreground']) },
      { token: 'delimiter', foreground: rgb(t['--subtle-foreground']) },
    ],
    colors: {
      'editor.background': hex(t['--surface']),
      'editor.foreground': hex(t['--foreground']),
      'editorGutter.background': hex(t['--surface']),
      'editor.lineHighlightBackground': hex(t['--surface-subtle']),
      'editor.lineHighlightBorder': hex(t['--surface-subtle']),
      'editor.selectionBackground': hex(t['--brand-400'], mode === 'dark' ? 0.3 : 0.24),
      'editor.inactiveSelectionBackground': hex(t['--brand-400'], 0.14),
      'editor.selectionHighlightBackground': hex(t['--brand-400'], 0.12),
      'editor.wordHighlightBackground': hex(t['--brand-400'], 0.12),
      'editor.findMatchBackground': hex(t['--chart-2'], 0.45),
      'editor.findMatchHighlightBackground': hex(t['--chart-2'], 0.22),
      'editorCursor.foreground': hex(t['--primary']),
      'editorLineNumber.foreground': hex(t['--subtle-foreground'], 0.7),
      'editorLineNumber.activeForeground': hex(t['--foreground']),
      'editorIndentGuide.background1': hex(t['--border']),
      'editorIndentGuide.activeBackground1': hex(t['--border-strong']),
      'editorWhitespace.foreground': hex(t['--border-strong']),
      'editorBracketMatch.background': hex(t['--brand-400'], 0.14),
      'editorBracketMatch.border': hex(t['--brand-400'], 0.5),
      'editorBracketHighlight.foreground1': hex(t['--muted-foreground']),
      'editorBracketHighlight.foreground2': hex(t['--muted-foreground']),
      'editorBracketHighlight.foreground3': hex(t['--muted-foreground']),
      'editorBracketHighlight.unexpectedBracket.foreground': hex(t['--destructive']),
      'editorError.foreground': hex(t['--destructive']),
      'editorWarning.foreground': hex(t['--warning']),
      'editorInfo.foreground': hex(t['--info']),
      'editorOverviewRuler.border': hex(t['--border'], 0),
      'editorOverviewRuler.errorForeground': hex(t['--destructive']),
      'editorOverviewRuler.warningForeground': hex(t['--warning']),
      'editorOverviewRuler.infoForeground': hex(t['--info']),
      'editorGutter.modifiedBackground': hex(t['--primary']),
      'editorGutter.addedBackground': hex(t['--success']),
      'editorWidget.background': hex(t['--popover']),
      'editorWidget.foreground': hex(t['--foreground']),
      'editorWidget.border': hex(t['--border-strong']),
      'editorHoverWidget.background': hex(t['--popover']),
      'editorHoverWidget.border': hex(t['--border-strong']),
      'editorSuggestWidget.background': hex(t['--popover']),
      'editorSuggestWidget.border': hex(t['--border-strong']),
      'editorSuggestWidget.selectedBackground': hex(t['--surface-sunken']),
      'editorStickyScroll.background': hex(t['--surface']),
      'editorStickyScrollHover.background': hex(t['--surface-subtle']),
      'editorLightBulb.foreground': hex(t['--warning']),
      'diffEditor.insertedTextBackground': hex(t['--success'], 0.16),
      'diffEditor.removedTextBackground': hex(t['--destructive'], 0.16),
      'diffEditor.insertedLineBackground': hex(t['--success'], mode === 'dark' ? 0.12 : 0.09),
      'diffEditor.removedLineBackground': hex(t['--destructive'], mode === 'dark' ? 0.12 : 0.08),
      'diffEditorGutter.insertedLineBackground': hex(t['--success'], 0.18),
      'diffEditorGutter.removedLineBackground': hex(t['--destructive'], 0.18),
      'diffEditor.unchangedRegionBackground': hex(t['--surface-subtle']),
      'diffEditor.unchangedRegionForeground': hex(t['--muted-foreground']),
      'diffEditor.unchangedCodeBackground': hex(t['--surface-subtle'], 0.6),
      'diffEditor.border': hex(t['--border']),
      focusBorder: hex(t['--primary'], 0),
      'scrollbar.shadow': hex(t['--border'], 0),
      'scrollbarSlider.background': hex(t['--border-strong'], 0.55),
      'scrollbarSlider.hoverBackground': hex(t['--border-strong'], 0.85),
      'scrollbarSlider.activeBackground': hex(t['--subtle-foreground'], 0.6),
      'input.background': hex(t['--surface']),
      'input.border': hex(t['--border-strong']),
      'list.hoverBackground': hex(t['--surface-subtle']),
      'list.activeSelectionBackground': hex(t['--surface-sunken']),
      'list.activeSelectionForeground': hex(t['--foreground']),
      'list.highlightForeground': hex(t['--primary']),
      'minimap.background': hex(t['--surface']),
    },
  }
}

export const editorThemeName = (mode: Mode) => `vsr-editor-${mode}`

/** Defines (or refreshes) the editor theme of a mode from the tokens currently in effect. */
export function defineEditorTheme(monaco: MonacoApi, mode: Mode): string {
  const name = editorThemeName(mode)
  monaco.editor.defineTheme(name, themeFor(mode))
  return name
}

/**
 * The Monaco theme that follows the shell's light or dark mode and accent.
 * Pass the Monaco instance once it has loaded; the theme is redefined from the
 * tokens and applied whenever the mode or accent changes.
 */
export function useEditorMonacoTheme(monaco: MonacoApi | null): string {
  const { resolved, accent } = useTheme()
  const name = editorThemeName(resolved)
  React.useEffect(() => {
    if (!monaco) return
    // Tokens switch with the `.dark` class in the same frame as this effect.
    const frame = window.requestAnimationFrame(() => {
      defineEditorTheme(monaco, resolved)
      monaco.editor.setTheme(editorThemeName(resolved))
    })
    return () => window.cancelAnimationFrame(frame)
  }, [monaco, resolved, accent])
  return name
}

// Monaco measures glyphs on a canvas, which cannot resolve a CSS variable.
function monoFontFamily(): string {
  if (typeof document === 'undefined') return 'ui-monospace, monospace'
  const family = getComputedStyle(document.documentElement).getPropertyValue('--font-mono').trim()
  return family || 'ui-monospace, SFMono-Regular, Menlo, monospace'
}

/** Shared Monaco options so the DSL editor and the deploy diff read alike. */
export const editorBaseOptions = (): monacoNs.editor.IStandaloneEditorConstructionOptions => ({
  fontSize: 12.5,
  lineHeight: 20,
  fontFamily: monoFontFamily(),
  fontLigatures: false,
  minimap: { enabled: false },
  lineNumbers: 'on',
  scrollBeyondLastLine: false,
  wordWrap: 'on',
  tabSize: 2,
  insertSpaces: true,
  automaticLayout: true,
  bracketPairColorization: { enabled: false },
  guides: { indentation: true, bracketPairs: false },
  padding: { top: 10, bottom: 10 },
  renderLineHighlight: 'line',
  smoothScrolling: true,
  cursorBlinking: 'smooth',
  cursorSmoothCaretAnimation: 'on',
  overviewRulerBorder: false,
  hideCursorInOverviewRuler: true,
  scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10, useShadows: false },
  stickyScroll: { enabled: true },
  fixedOverflowWidgets: true,
})
