import React, { useState, useCallback } from 'react'
import {
  SIGNAL_TYPES,
  PLUGIN_TYPES,
  BACKEND_TYPES,
  ALGORITHM_TYPES,
  PLUGIN_DESCRIPTIONS,
  ALGORITHM_DESCRIPTIONS,
  getSignalFieldSchema,
  getPluginFieldSchema,
  getAlgorithmFieldSchema,
} from '@/lib/dslMutations'
import styles from './DslGuide.module.css'

interface DslGuideProps {
  onInsertSnippet?: (snippet: string) => void
}

// Collapsible section
const Section: React.FC<{
  title: string
  icon: string
  defaultOpen?: boolean
  children: React.ReactNode
}> = ({ title, icon, defaultOpen = false, children }) => {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={styles.section}>
      <button className={styles.sectionHeader} onClick={() => setOpen(!open)}>
        <span className={styles.sectionChevron}>{open ? '▾' : '▸'}</span>
        <span className={styles.sectionIcon}>{icon}</span>
        <span className={styles.sectionTitle}>{title}</span>
      </button>
      {open && <div className={styles.sectionBody}>{children}</div>}
    </div>
  )
}

// Code block with optional insert button
const CodeBlock: React.FC<{
  code: string
  label?: string
  onInsert?: (code: string) => void
}> = ({ code, label, onInsert }) => (
  <div className={styles.codeBlock}>
    {label && <div className={styles.codeLabel}>{label}</div>}
    <pre className={styles.codeContent}>{code}</pre>
    {onInsert && (
      <button className={styles.insertBtn} onClick={() => onInsert(code)} title="Insert into editor">
        + Insert
      </button>
    )}
  </div>
)

// Field schema table
const FieldTable: React.FC<{
  fields: { key: string; label: string; type: string; required?: boolean; description?: string; options?: string[] }[]
}> = ({ fields }) => {
  if (fields.length === 0) return <div className={styles.muted}>No configurable fields</div>
  return (
    <table className={styles.fieldTable}>
      <thead>
        <tr><th>Field</th><th>Type</th><th>Info</th></tr>
      </thead>
      <tbody>
        {fields.map((f) => (
          <tr key={f.key}>
            <td>
              <code>{f.key}</code>
              {f.required && <span className={styles.required}>*</span>}
            </td>
            <td className={styles.fieldType}>
              {f.type}
              {f.options && f.options.length > 0 && (
                <span className={styles.fieldOptions}> ({f.options.join(' | ')})</span>
              )}
            </td>
            <td className={styles.fieldDesc}>{f.description || ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// Sub-section for a type (signal type, plugin type, etc.)
const TypeDetail: React.FC<{
  name: string
  description?: string
  fields: { key: string; label: string; type: string; required?: boolean; description?: string; options?: string[] }[]
}> = ({ name, description, fields }) => {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className={styles.typeItem}>
      <button className={styles.typeHeader} onClick={() => setExpanded(!expanded)}>
        <span className={styles.typeChevron}>{expanded ? '▾' : '▸'}</span>
        <code className={styles.typeName}>{name}</code>
        {description && <span className={styles.typeDesc}>{description}</span>}
      </button>
      {expanded && (
        <div className={styles.typeBody}>
          <FieldTable fields={fields} />
        </div>
      )}
    </div>
  )
}

// ---------- Snippet templates ----------

const QUICK_START_SNIPPET = `# ============================================
# SIGNALS
# ============================================

SIGNAL keyword urgent_request {
  operator: "any"
  keywords: ["urgent", "asap", "emergency"]
  method: "regex"
  case_sensitive: false
}

SIGNAL embedding ai_topics {
  threshold: 0.75
  candidates: ["machine learning", "neural network", "deep learning"]
  aggregation_method: "max"
}

SIGNAL domain math {
  description: "Mathematics and quantitative reasoning"
  mmlu_categories: ["math"]
}

# ============================================
# PLUGINS
# ============================================

PLUGIN safe_pii pii {
  enabled: true
  pii_types_allowed: []
}

# ============================================
# ROUTES
# ============================================

ROUTE ai_route (description = "AI-related requests") {
  PRIORITY 100

  WHEN keyword("urgent_request") AND embedding("ai_topics")

  MODEL "qwen2.5:3b" (reasoning = false)

  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
    on_error: "skip"
  }

  PLUGIN safe_pii
}

# ============================================
# BACKENDS
# ============================================

BACKEND vllm_endpoint ollama {
  address: "127.0.0.1"
  port: 11434
  weight: 1
  type: "ollama"
}

# ============================================
# GLOBAL
# ============================================

GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
  default_reasoning_effort: "low"
}`

const SIGNAL_SNIPPET = `SIGNAL keyword my_signal {
  operator: "any"
  keywords: ["hello", "world"]
  method: "regex"
  case_sensitive: false
}`

const ROUTE_SNIPPET = `ROUTE my_route (description = "Description") {
  PRIORITY 100

  WHEN keyword("my_signal")

  MODEL "qwen2.5:3b" (reasoning = false)

  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
  }
}`

const PLUGIN_SNIPPET = `PLUGIN my_plugin pii {
  enabled: true
  pii_types_allowed: []
}`

const BACKEND_SNIPPET = `BACKEND vllm_endpoint my_backend {
  address: "127.0.0.1"
  port: 11434
  weight: 1
  type: "ollama"
}`

const GLOBAL_SNIPPET = `GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
  default_reasoning_effort: "low"
}`

// Signal type descriptions
const SIGNAL_DESCRIPTIONS: Record<string, string> = {
  keyword: 'Match queries by keyword lists (regex, BM25, n-gram)',
  embedding: 'Semantic similarity matching via embedding vectors',
  domain: 'MMLU-based academic domain detection',
  fact_check: 'Flag queries requiring factual verification',
  user_feedback: 'Route based on user feedback signals',
  preference: 'User preference-based routing',
  language: 'Detect query language',
  context: 'Context window length requirements',
  complexity: 'Estimate query difficulty via embedding similarity',
  modality: 'Detect multi-modal input (text, image, audio)',
  authz: 'Authorization-based routing (RBAC)',
}

// Backend type descriptions
const BACKEND_DESCRIPTIONS: Record<string, string> = {
  vllm_endpoint: 'vLLM or Ollama inference endpoint',
  provider_profile: 'Cloud provider API profile (OpenAI, Azure, etc.)',
  embedding_model: 'Embedding model endpoint for signal detection',
  semantic_cache: 'Semantic cache backend storage',
  memory: 'Persistent memory/vector storage',
  response_api: 'OpenAI Responses API endpoint',
  vector_store: 'Vector store for RAG retrieval',
  image_gen_backend: 'Image generation backend (DALL-E, SD, etc.)',
}

const DslGuide: React.FC<DslGuideProps> = ({ onInsertSnippet }) => {
  const [searchQuery, setSearchQuery] = useState('')

  const handleInsert = useCallback((snippet: string) => {
    onInsertSnippet?.(snippet)
  }, [onInsertSnippet])

  const q = searchQuery.toLowerCase().trim()
  const matchesSearch = (text: string) => !q || text.toLowerCase().includes(q)

  return (
    <div className={styles.guide}>
      {/* Search */}
      <div className={styles.searchBox}>
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="6.5" cy="6.5" r="5" />
          <path d="M10 10l4.5 4.5" strokeLinecap="round" />
        </svg>
        <input
          className={styles.searchInput}
          type="text"
          placeholder="Search keywords, types, fields..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <button className={styles.searchClear} onClick={() => setSearchQuery('')}>
            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        )}
      </div>

      {/* Quick Start */}
      {matchesSearch('quick start template example') && (
        <Section title="Quick Start" icon="⚡" defaultOpen>
          <p className={styles.hint}>
            A DSL file defines <strong>signals</strong> (what to detect),{' '}
            <strong>routes</strong> (how to decide),{' '}
            <strong>plugins</strong> (pre/post processing),{' '}
            <strong>backends</strong> (infrastructure), and{' '}
            <strong>global</strong> settings.
          </p>
          <CodeBlock code={QUICK_START_SNIPPET} label="Full Example" onInsert={handleInsert} />
          <div className={styles.snippetGrid}>
            <CodeBlock code={SIGNAL_SNIPPET} label="Signal" onInsert={handleInsert} />
            <CodeBlock code={ROUTE_SNIPPET} label="Route" onInsert={handleInsert} />
            <CodeBlock code={PLUGIN_SNIPPET} label="Plugin" onInsert={handleInsert} />
            <CodeBlock code={BACKEND_SNIPPET} label="Backend" onInsert={handleInsert} />
            <CodeBlock code={GLOBAL_SNIPPET} label="Global" onInsert={handleInsert} />
          </div>
        </Section>
      )}

      {/* Signals */}
      {matchesSearch('signal keyword embedding domain') && (
        <Section title={`Signals (${SIGNAL_TYPES.length} types)`} icon="📡">
          <p className={styles.hint}>
            Signals detect patterns in user queries. Syntax:{' '}
            <code>SIGNAL &lt;type&gt; &lt;name&gt; {'{ fields }'}</code>
          </p>
          {SIGNAL_TYPES.filter((t) => matchesSearch(`signal ${t} ${SIGNAL_DESCRIPTIONS[t] || ''}`)).map((t) => (
            <TypeDetail
              key={t}
              name={t}
              description={SIGNAL_DESCRIPTIONS[t]}
              fields={getSignalFieldSchema(t)}
            />
          ))}
        </Section>
      )}

      {/* Routes & WHEN Expressions */}
      {matchesSearch('route when expression boolean model algorithm priority') && (
        <Section title="Routes & WHEN Expressions" icon="🔀">
          <p className={styles.hint}>
            Routes define decision logic. Syntax:{' '}
            <code>ROUTE &lt;name&gt; (description = &quot;...&quot;) {'{ ... }'}</code>
          </p>

          <div className={styles.subsection}>
            <h4>Route Structure</h4>
            <table className={styles.fieldTable}>
              <thead><tr><th>Clause</th><th>Required</th><th>Description</th></tr></thead>
              <tbody>
                <tr><td><code>PRIORITY</code></td><td>Yes</td><td>Integer priority (higher = matched first)</td></tr>
                <tr><td><code>WHEN</code></td><td>No</td><td>Boolean expression of signal references</td></tr>
                <tr><td><code>MODEL</code></td><td>Yes</td><td>One or more model references with attributes</td></tr>
                <tr><td><code>ALGORITHM</code></td><td>No</td><td>Model selection algorithm with config</td></tr>
                <tr><td><code>PLUGIN</code></td><td>No</td><td>Plugin references (with optional inline overrides)</td></tr>
              </tbody>
            </table>
          </div>

          <div className={styles.subsection}>
            <h4>WHEN Boolean Expressions</h4>
            <p className={styles.hint}>Combine signal references with <code>AND</code>, <code>OR</code>, <code>NOT</code>, and parentheses.</p>
            <table className={styles.fieldTable}>
              <thead><tr><th>Operator</th><th>Precedence</th><th>Example</th></tr></thead>
              <tbody>
                <tr><td><code>NOT</code></td><td>Highest</td><td><code>NOT domain(&quot;other&quot;)</code></td></tr>
                <tr><td><code>AND</code></td><td>Medium</td><td><code>keyword(&quot;a&quot;) AND embedding(&quot;b&quot;)</code></td></tr>
                <tr><td><code>OR</code></td><td>Lowest</td><td><code>domain(&quot;math&quot;) OR domain(&quot;code&quot;)</code></td></tr>
              </tbody>
            </table>
            <CodeBlock
              code={`WHEN keyword("urgent") AND (domain("math") OR domain("code")) AND NOT embedding("general")`}
              label="Complex WHEN example"
            />
          </div>

          <div className={styles.subsection}>
            <h4>MODEL Attributes</h4>
            <table className={styles.fieldTable}>
              <thead><tr><th>Attribute</th><th>Type</th><th>Description</th></tr></thead>
              <tbody>
                <tr><td><code>reasoning</code></td><td>boolean</td><td>Enable reasoning mode</td></tr>
                <tr><td><code>effort</code></td><td>string</td><td><code>low</code> | <code>medium</code> | <code>high</code></td></tr>
                <tr><td><code>param_size</code></td><td>string</td><td>e.g. <code>3b</code>, <code>70b</code></td></tr>
                <tr><td><code>weight</code></td><td>number</td><td>Routing weight (0-1)</td></tr>
                <tr><td><code>lora</code></td><td>string</td><td>LoRA adapter name</td></tr>
                <tr><td><code>reasoning_family</code></td><td>string</td><td>e.g. <code>qwen3</code>, <code>deepseek</code></td></tr>
              </tbody>
            </table>
            <CodeBlock
              code={`MODEL "qwen3:70b" (reasoning = true, effort = "high", param_size = "70b"),
      "qwen2.5:3b" (reasoning = false, param_size = "3b")`}
              label="Multi-model example"
            />
          </div>
        </Section>
      )}

      {/* Algorithms */}
      {matchesSearch('algorithm confidence ratings remom static elo') && (
        <Section title={`Algorithms (${ALGORITHM_TYPES.length} types)`} icon="🧮">
          <p className={styles.hint}>
            Algorithms determine how to select among multiple models. Syntax:{' '}
            <code>ALGORITHM &lt;type&gt; {'{ fields }'}</code> (inside a ROUTE)
          </p>
          {ALGORITHM_TYPES.filter((t) => matchesSearch(`algorithm ${t} ${ALGORITHM_DESCRIPTIONS[t] || ''}`)).map((t) => (
            <TypeDetail
              key={t}
              name={t}
              description={ALGORITHM_DESCRIPTIONS[t]}
              fields={getAlgorithmFieldSchema(t)}
            />
          ))}
        </Section>
      )}

      {/* Plugins */}
      {matchesSearch('plugin jailbreak pii cache memory rag') && (
        <Section title={`Plugins (${PLUGIN_TYPES.length} types)`} icon="🔌">
          <p className={styles.hint}>
            Plugins add pre/post processing. Declare with{' '}
            <code>PLUGIN &lt;name&gt; &lt;type&gt; {'{ fields }'}</code>, reference in routes with{' '}
            <code>PLUGIN &lt;name&gt;</code>.
          </p>
          {PLUGIN_TYPES.filter((t) => matchesSearch(`plugin ${t} ${PLUGIN_DESCRIPTIONS[t] || ''}`)).map((t) => (
            <TypeDetail
              key={t}
              name={t}
              description={PLUGIN_DESCRIPTIONS[t]}
              fields={getPluginFieldSchema(t)}
            />
          ))}
        </Section>
      )}

      {/* Backends */}
      {matchesSearch('backend vllm endpoint provider embedding') && (
        <Section title={`Backends (${BACKEND_TYPES.length} types)`} icon="🖥️">
          <p className={styles.hint}>
            Backends define infrastructure. Syntax:{' '}
            <code>BACKEND &lt;type&gt; &lt;name&gt; {'{ fields }'}</code>
          </p>
          {BACKEND_TYPES.filter((t) => matchesSearch(`backend ${t} ${BACKEND_DESCRIPTIONS[t] || ''}`)).map((t) => (
            <div key={t} className={styles.typeItem}>
              <div className={styles.typeHeader} style={{ cursor: 'default' }}>
                <code className={styles.typeName}>{t}</code>
                <span className={styles.typeDesc}>{BACKEND_DESCRIPTIONS[t] || ''}</span>
              </div>
            </div>
          ))}
          <CodeBlock
            code={`BACKEND vllm_endpoint ollama {
  address: "127.0.0.1"
  port: 11434
  weight: 1
  type: "ollama"
}

BACKEND provider_profile openai {
  provider: "openai"
  api_key_env: "OPENAI_API_KEY"
}

BACKEND embedding_model my_embed {
  model: "all-MiniLM-L6-v2"
  backend: "ollama"
}`}
            label="Backend examples"
            onInsert={handleInsert}
          />
        </Section>
      )}

      {/* Global */}
      {matchesSearch('global settings default strategy reasoning') && (
        <Section title="Global Settings" icon="⚙️">
          <p className={styles.hint}>
            One <code>GLOBAL {'{ ... }'}</code> block per file. Sets defaults for the entire router.
          </p>
          <table className={styles.fieldTable}>
            <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td><code>default_model</code></td><td>string</td><td>Fallback model when no route matches</td></tr>
              <tr><td><code>strategy</code></td><td>string</td><td><code>priority</code> | <code>first_match</code> | <code>weighted</code></td></tr>
              <tr><td><code>default_reasoning_effort</code></td><td>string</td><td><code>low</code> | <code>medium</code> | <code>high</code></td></tr>
              <tr><td><code>reasoning_families</code></td><td>object</td><td>Per-family reasoning config</td></tr>
              <tr><td><code>max_routing_time_ms</code></td><td>number</td><td>Timeout for signal evaluation</td></tr>
              <tr><td><code>enable_parallel_signals</code></td><td>boolean</td><td>Evaluate signals in parallel</td></tr>
            </tbody>
          </table>
          <CodeBlock
            code={GLOBAL_SNIPPET}
            label="Global example"
            onInsert={handleInsert}
          />
        </Section>
      )}

      {/* Cheat Sheet */}
      {matchesSearch('cheat sheet syntax grammar reference') && (
        <Section title="Cheat Sheet" icon="📋">
          <div className={styles.cheatSheet}>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>SIGNAL &lt;type&gt; &lt;name&gt; {'{ ... }'}</code>
              <span>Declare a signal</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>ROUTE &lt;name&gt; (description = &quot;...&quot;) {'{ ... }'}</code>
              <span>Define a route</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>PLUGIN &lt;name&gt; &lt;type&gt; {'{ ... }'}</code>
              <span>Declare a plugin template</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>BACKEND &lt;type&gt; &lt;name&gt; {'{ ... }'}</code>
              <span>Declare a backend</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>GLOBAL {'{ ... }'}</code>
              <span>Global settings (one per file)</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>PRIORITY &lt;number&gt;</code>
              <span>Route priority (inside ROUTE)</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>WHEN &lt;bool_expr&gt;</code>
              <span>Condition (inside ROUTE)</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>MODEL &quot;name&quot; (attrs...)</code>
              <span>Model ref (inside ROUTE)</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>ALGORITHM &lt;type&gt; {'{ ... }'}</code>
              <span>Algorithm (inside ROUTE)</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}># comment</code>
              <span>Line comment</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>&quot;string&quot; | 123 | true | [a, b]</code>
              <span>Value types</span>
            </div>
            <div className={styles.cheatItem}>
              <code className={styles.cheatSyntax}>{'{ key: value, ... }'}</code>
              <span>Nested object</span>
            </div>
          </div>
        </Section>
      )}
    </div>
  )
}

export default DslGuide
