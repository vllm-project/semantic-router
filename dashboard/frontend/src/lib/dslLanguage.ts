/**
 * Monaco Editor language definition for the Signal DSL.
 *
 * Provides:
 * - Syntax highlighting (tokenizer / Monarch grammar)
 * - Language configuration (brackets, comments, auto-closing)
 * - Completion provider (keywords, signal types, plugin types, etc.)
 * - Context-aware completion using symbol table from WASM validation
 * - Diagnostics adapter (maps WASM Diagnostic[] → Monaco markers)
 */

import type * as monacoNs from 'monaco-editor'
import type { Diagnostic, SymbolTable } from '@/types/dsl'

export const DSL_LANGUAGE_ID = 'signal-dsl'

// ---------- Language Configuration ----------

export const languageConfiguration: monacoNs.languages.LanguageConfiguration = {
  comments: {
    lineComment: '#',
  },
  brackets: [
    ['{', '}'],
    ['[', ']'],
    ['(', ')'],
  ],
  autoClosingPairs: [
    { open: '{', close: '}' },
    { open: '[', close: ']' },
    { open: '(', close: ')' },
    { open: '"', close: '"', notIn: ['string'] },
  ],
  surroundingPairs: [
    { open: '{', close: '}' },
    { open: '[', close: ']' },
    { open: '(', close: ')' },
    { open: '"', close: '"' },
  ],
  folding: {
    offSide: false,
    markers: {
      start: /\{\s*$/,
      end: /^\s*\}/,
    },
  },
  indentationRules: {
    increaseIndentPattern: /\{\s*$/,
    decreaseIndentPattern: /^\s*\}/,
  },
}

// ---------- Monarch Tokenizer ----------

export const monarchTokens: monacoNs.languages.IMonarchLanguage = {
  defaultToken: '',
  ignoreCase: false,

  keywords: [
    'SIGNAL', 'ROUTE', 'PLUGIN', 'BACKEND', 'GLOBAL',
    'PRIORITY', 'WHEN', 'MODEL', 'ALGORITHM',
  ],

  operators: ['AND', 'OR', 'NOT'],

  signalTypes: [
    'keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
    'preference', 'language', 'context', 'complexity', 'modality', 'authz',
  ],

  pluginTypes: [
    'jailbreak', 'pii', 'semantic_cache', 'memory', 'system_prompt',
    'header_mutation', 'hallucination', 'router_replay', 'rag', 'image_gen',
  ],

  algoTypes: [
    'confidence', 'ratings', 'remom', 'static', 'elo', 'router_dc',
    'automix', 'hybrid', 'rl_driven', 'gmtrouter', 'latency_aware',
    'knn', 'kmeans', 'svm',
  ],

  backendTypes: [
    'vllm_endpoint', 'provider_profile', 'embedding_model',
    'semantic_cache', 'memory', 'response_api',
  ],

  booleans: ['true', 'false'],

  tokenizer: {
    root: [
      // Comments
      [/#.*$/, 'comment'],

      // Strings
      [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],

      // Numbers
      [/\d+\.\d+/, 'number.float'],
      [/\d+/, 'number'],

      // Booleans
      [/\b(true|false)\b/, 'constant.language'],

      // Top-level keywords (SIGNAL, ROUTE, etc.)
      [/\b(SIGNAL|ROUTE|PLUGIN|BACKEND|GLOBAL)\b/, 'keyword'],

      // Route sub-keywords
      [/\b(PRIORITY|WHEN|MODEL|ALGORITHM)\b/, 'keyword'],

      // Boolean operators
      [/\b(AND|OR|NOT)\b/, 'keyword.operator'],

      // Signal types (after SIGNAL keyword)
      [
        /\b(keyword|embedding|domain|fact_check|user_feedback|preference|language|context|complexity|modality|authz)\b/,
        'type',
      ],

      // Plugin types
      [
        /\b(jailbreak|pii|semantic_cache|memory|system_prompt|header_mutation|hallucination|router_replay|rag|image_gen)\b/,
        'type.plugin',
      ],

      // Algorithm types
      [
        /\b(confidence|ratings|remom|static|elo|router_dc|automix|hybrid|rl_driven|gmtrouter|latency_aware|knn|kmeans|svm)\b/,
        'type.algorithm',
      ],

      // Backend types
      [
        /\b(vllm_endpoint|provider_profile|embedding_model|response_api)\b/,
        'type.backend',
      ],

      // Field names (identifier followed by colon)
      [/[a-zA-Z_][\w\-.]*(?=\s*:)/, 'variable.field'],

      // Identifiers
      [/[a-zA-Z_][\w\-.]*/, 'identifier'],

      // Punctuation
      [/[{}()[\]]/, '@brackets'],
      [/[:,=]/, 'delimiter'],

      // Whitespace
      [/\s+/, 'white'],
    ],

    string: [
      [/[^\\"]+/, 'string'],
      [/\\./, 'string.escape'],
      [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }],
    ],
  },
}

// ---------- Theme ----------

export function defineTheme(monaco: typeof monacoNs): void {
  monaco.editor.defineTheme('signal-dsl-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '6A9955', fontStyle: 'italic' },
      { token: 'keyword', foreground: 'C586C0', fontStyle: 'bold' },
      { token: 'keyword.operator', foreground: 'D4D4D4', fontStyle: 'bold' },
      { token: 'type', foreground: '4EC9B0' },
      { token: 'type.plugin', foreground: 'DCDCAA' },
      { token: 'type.algorithm', foreground: '9CDCFE' },
      { token: 'type.backend', foreground: '4FC1FF' },
      { token: 'string', foreground: 'CE9178' },
      { token: 'string.escape', foreground: 'D7BA7D' },
      { token: 'number', foreground: 'B5CEA8' },
      { token: 'number.float', foreground: 'B5CEA8' },
      { token: 'constant.language', foreground: '569CD6' },
      { token: 'variable.field', foreground: '9CDCFE' },
      { token: 'identifier', foreground: 'D4D4D4' },
      { token: 'delimiter', foreground: 'D4D4D4' },
    ],
    colors: {
      'editor.background': '#1a1a1a',
      'editor.foreground': '#e8e8e8',
      'editor.lineHighlightBackground': '#252525',
      'editor.selectionBackground': '#76b90040',
      'editorCursor.foreground': '#76b900',
      'editorLineNumber.foreground': '#666666',
      'editorLineNumber.activeForeground': '#76b900',
      'editorIndentGuide.background': '#333333',
      'editorIndentGuide.activeBackground': '#444444',
    },
  })
}

// ---------- Completion Provider ----------

const KEYWORD_SUGGESTIONS = [
  { label: 'SIGNAL', insertText: 'SIGNAL ${1:keyword} ${2:name} {\n\t$0\n}', detail: 'Signal declaration' },
  { label: 'ROUTE', insertText: 'ROUTE ${1:name} (description = "${2:desc}") {\n\tPRIORITY ${3:10}\n\tWHEN ${4:condition}\n\tMODEL "${5:model}"\n\t$0\n}', detail: 'Route declaration' },
  { label: 'PLUGIN', insertText: 'PLUGIN ${1:name} ${2:type} {\n\t$0\n}', detail: 'Plugin template' },
  { label: 'BACKEND', insertText: 'BACKEND ${1:vllm_endpoint} ${2:name} {\n\taddress: "${3:localhost}"\n\tport: ${4:8000}\n\t$0\n}', detail: 'Backend declaration' },
  { label: 'GLOBAL', insertText: 'GLOBAL {\n\t$0\n}', detail: 'Global settings' },
  { label: 'PRIORITY', insertText: 'PRIORITY ${1:10}', detail: 'Route priority (1-100)' },
  { label: 'WHEN', insertText: 'WHEN ${1:condition}', detail: 'Route condition' },
  { label: 'MODEL', insertText: 'MODEL "${1:model-name}"', detail: 'Model reference' },
  { label: 'ALGORITHM', insertText: 'ALGORITHM ${1:confidence} {\n\t$0\n}', detail: 'Algorithm block' },
  { label: 'AND', insertText: 'AND', detail: 'Boolean AND' },
  { label: 'OR', insertText: 'OR', detail: 'Boolean OR' },
  { label: 'NOT', insertText: 'NOT', detail: 'Boolean NOT' },
]

const SIGNAL_TYPE_SUGGESTIONS = [
  { label: 'keyword', detail: 'Keyword matching signal' },
  { label: 'embedding', detail: 'Embedding similarity signal' },
  { label: 'domain', detail: 'Domain classification signal' },
  { label: 'fact_check', detail: 'Fact-checking signal' },
  { label: 'user_feedback', detail: 'User feedback signal' },
  { label: 'preference', detail: 'User preference signal' },
  { label: 'language', detail: 'Language detection signal' },
  { label: 'context', detail: 'Context length signal' },
  { label: 'complexity', detail: 'Query complexity signal' },
  { label: 'modality', detail: 'Input modality signal' },
  { label: 'authz', detail: 'Authorization signal' },
]

const PLUGIN_TYPE_SUGGESTIONS = [
  { label: 'jailbreak', detail: 'Jailbreak detection plugin' },
  { label: 'pii', detail: 'PII detection/masking plugin' },
  { label: 'semantic_cache', detail: 'Semantic caching plugin' },
  { label: 'memory', detail: 'Conversation memory plugin' },
  { label: 'system_prompt', detail: 'System prompt injection plugin' },
  { label: 'header_mutation', detail: 'HTTP header mutation plugin' },
  { label: 'hallucination', detail: 'Hallucination detection plugin' },
  { label: 'router_replay', detail: 'Request replay plugin' },
  { label: 'rag', detail: 'RAG (Retrieval Augmented Generation) plugin' },
  { label: 'image_gen', detail: 'Image generation plugin' },
]

const ALGO_TYPE_SUGGESTIONS = [
  { label: 'confidence', detail: 'Confidence-based routing' },
  { label: 'ratings', detail: 'Ratings-based routing' },
  { label: 'remom', detail: 'ReMoM algorithm' },
  { label: 'static', detail: 'Static model assignment' },
  { label: 'elo', detail: 'Elo rating algorithm' },
  { label: 'router_dc', detail: 'Router DC algorithm' },
  { label: 'automix', detail: 'AutoMix algorithm' },
  { label: 'hybrid', detail: 'Hybrid routing' },
  { label: 'latency_aware', detail: 'Latency-aware routing' },
]

/**
 * Determine the completion context by scanning backwards from the cursor.
 * Returns the context keyword if the cursor is in a recognizable position.
 */
function getCompletionContext(model: monacoNs.editor.ITextModel, position: monacoNs.Position): string {
  const lineContent = model.getLineContent(position.lineNumber)
  const textBefore = lineContent.substring(0, position.column - 1).trim()

  // After SIGNAL keyword → suggest signal types
  if (/^SIGNAL\s*$/.test(textBefore)) return 'signal-type'

  // After PLUGIN keyword (top-level or route-level) → suggest plugin types/templates
  if (/^PLUGIN\s*$/.test(textBefore) || /PLUGIN\s+\w*$/.test(textBefore)) return 'plugin-ref'

  // After ALGORITHM keyword → suggest algorithm types
  if (/^ALGORITHM\s*$/.test(textBefore) || /ALGORITHM\s+\w*$/.test(textBefore)) return 'algo-type'

  // After MODEL keyword → suggest model names
  if (/^MODEL\s*"?[^"]*$/.test(textBefore) || /MODEL\s*$/.test(textBefore)) return 'model-ref'

  // After WHEN keyword or boolean operators → suggest signal references
  if (/\b(WHEN|AND|OR|NOT)\s+\w*$/.test(textBefore)) return 'when-expr'

  // Inside a WHEN expression: detect by scanning upward for WHEN on the same or previous lines
  // within a ROUTE block (simple heuristic: check current and a few previous lines)
  for (let ln = position.lineNumber; ln >= Math.max(1, position.lineNumber - 5); ln--) {
    const line = model.getLineContent(ln).trim()
    if (/^WHEN\b/.test(line)) return 'when-expr'
    if (/^(PRIORITY|MODEL|ALGORITHM|PLUGIN|\}|ROUTE)\b/.test(line)) break
  }

  return 'default'
}

export function createCompletionProvider(
  monaco: typeof monacoNs,
  getSymbols: () => SymbolTable | null,
): monacoNs.languages.CompletionItemProvider {
  return {
    triggerCharacters: [' ', '\n', '"'],
    provideCompletionItems(model, position) {
      const word = model.getWordUntilPosition(position)
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      }

      const context = getCompletionContext(model, position)
      const symbols = getSymbols()
      const suggestions: monacoNs.languages.CompletionItem[] = []

      switch (context) {
        case 'signal-type':
          for (const s of SIGNAL_TYPE_SUGGESTIONS) {
            suggestions.push({
              label: s.label,
              kind: monaco.languages.CompletionItemKind.TypeParameter,
              insertText: s.label,
              detail: s.detail,
              range,
            })
          }
          return { suggestions }

        case 'plugin-ref':
          // Suggest defined plugin templates first
          if (symbols?.plugins.length) {
            for (const name of symbols.plugins) {
              suggestions.push({
                label: name,
                kind: monaco.languages.CompletionItemKind.Reference,
                insertText: name,
                detail: 'Plugin template',
                sortText: '0_' + name,
                range,
              })
            }
          }
          // Then suggest inline plugin types
          for (const s of PLUGIN_TYPE_SUGGESTIONS) {
            suggestions.push({
              label: s.label,
              kind: monaco.languages.CompletionItemKind.TypeParameter,
              insertText: s.label,
              detail: s.detail,
              sortText: '1_' + s.label,
              range,
            })
          }
          return { suggestions }

        case 'algo-type':
          for (const s of ALGO_TYPE_SUGGESTIONS) {
            suggestions.push({
              label: s.label,
              kind: monaco.languages.CompletionItemKind.TypeParameter,
              insertText: s.label,
              detail: s.detail,
              range,
            })
          }
          return { suggestions }

        case 'model-ref':
          if (symbols?.models.length) {
            for (const name of symbols.models) {
              suggestions.push({
                label: name,
                kind: monaco.languages.CompletionItemKind.Value,
                insertText: `"${name}"`,
                detail: 'Declared model',
                range,
              })
            }
          }
          return { suggestions }

        case 'when-expr':
          // Suggest declared signal references: type("name") format
          if (symbols?.signals.length) {
            for (const sig of symbols.signals) {
              const label = `${sig.type}("${sig.name}")`
              suggestions.push({
                label,
                kind: monaco.languages.CompletionItemKind.Variable,
                insertText: label,
                detail: `${sig.type} signal`,
                sortText: '0_' + label,
                range,
              })
            }
          }
          // Also suggest boolean operators
          for (const op of ['AND', 'OR', 'NOT']) {
            suggestions.push({
              label: op,
              kind: monaco.languages.CompletionItemKind.Keyword,
              insertText: op,
              detail: `Boolean ${op}`,
              sortText: '1_' + op,
              range,
            })
          }
          return { suggestions }

        default:
          // Default: suggest all keywords
          for (const kw of KEYWORD_SUGGESTIONS) {
            suggestions.push({
              label: kw.label,
              kind: monaco.languages.CompletionItemKind.Keyword,
              insertText: kw.insertText,
              insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
              detail: kw.detail,
              range,
            })
          }
          return { suggestions }
      }
    },
  }
}

// ---------- Diagnostics → Monaco Markers ----------

export function diagnosticsToMarkers(
  monaco: typeof monacoNs,
  diagnostics: Diagnostic[],
): monacoNs.editor.IMarkerData[] {
  return diagnostics.map((d) => ({
    severity: diagLevelToSeverity(monaco, d.level),
    message: d.message,
    startLineNumber: Math.max(1, d.line),
    startColumn: Math.max(1, d.column),
    endLineNumber: Math.max(1, d.line),
    endColumn: Math.max(1, d.column + 1),
    source: 'signal-dsl',
    // Encode fix info as JSON in relatedInformation tag for CodeAction provider
    ...(d.fixes?.length ? { tags: [], relatedInformation: d.fixes.map(f => ({
      resource: { path: '' } as unknown as monacoNs.Uri,
      message: JSON.stringify({ description: f.description, newText: f.newText }),
      startLineNumber: Math.max(1, d.line),
      startColumn: Math.max(1, d.column),
      endLineNumber: Math.max(1, d.line),
      endColumn: Math.max(1, d.column + 1),
    })) } : {}),
  }))
}

function diagLevelToSeverity(
  monaco: typeof monacoNs,
  level: string,
): monacoNs.MarkerSeverity {
  switch (level) {
    case 'error':
      return monaco.MarkerSeverity.Error
    case 'warning':
      return monaco.MarkerSeverity.Warning
    case 'constraint':
      return monaco.MarkerSeverity.Info
    default:
      return monaco.MarkerSeverity.Info
  }
}

// ---------- CodeAction Provider (Quick Fix) ----------

export function createCodeActionProvider(
  _monaco: typeof monacoNs,
  getDiagnostics: () => Diagnostic[],
): monacoNs.languages.CodeActionProvider {
  return {
    provideCodeActions(model, _range, context) {
      const actions: monacoNs.languages.CodeAction[] = []
      const diagnostics = getDiagnostics()

      for (const marker of context.markers) {
        // Find matching diagnostic with fixes
        const diag = diagnostics.find(d =>
          d.line === marker.startLineNumber &&
          d.column === marker.startColumn &&
          d.message === marker.message
        )
        if (!diag?.fixes?.length) continue

        for (const fix of diag.fixes) {
          // Compute the range of text to replace: find the word at the diagnostic position
          const lineContent = model.getLineContent(diag.line)
          let startCol = diag.column
          let endCol = diag.column

          // Expand to cover the current word/token at position
          while (startCol > 1 && /[\w\-.]/.test(lineContent[startCol - 2])) startCol--
          while (endCol <= lineContent.length && /[\w\-.]/.test(lineContent[endCol - 1])) endCol++

          actions.push({
            title: fix.description,
            kind: 'quickfix',
            diagnostics: [marker],
            isPreferred: true,
            edit: {
              edits: [{
                resource: model.uri,
                textEdit: {
                  range: {
                    startLineNumber: diag.line,
                    startColumn: startCol,
                    endLineNumber: diag.line,
                    endColumn: endCol,
                  },
                  text: fix.newText,
                },
                versionId: model.getVersionId(),
              }],
            },
          })
        }
      }

      return { actions, dispose() {} }
    },
  }
}

// ---------- Registration ----------

export function registerDSLLanguage(
  monaco: typeof monacoNs,
  getSymbols?: () => SymbolTable | null,
  getDiagnostics?: () => Diagnostic[],
): void {
  // Only register once
  if (monaco.languages.getLanguages().some((l) => l.id === DSL_LANGUAGE_ID)) {
    return
  }

  monaco.languages.register({ id: DSL_LANGUAGE_ID })
  monaco.languages.setLanguageConfiguration(DSL_LANGUAGE_ID, languageConfiguration)
  monaco.languages.setMonarchTokensProvider(DSL_LANGUAGE_ID, monarchTokens)
  monaco.languages.registerCompletionItemProvider(
    DSL_LANGUAGE_ID,
    createCompletionProvider(monaco, getSymbols ?? (() => null)),
  )
  // Register CodeAction provider for Quick Fix support
  if (getDiagnostics) {
    monaco.languages.registerCodeActionProvider(DSL_LANGUAGE_ID, createCodeActionProvider(monaco, getDiagnostics))
  }
  defineTheme(monaco)
}
