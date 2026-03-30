package dsl

// SchemaReference is the complete DSL grammar reference injected into the
// system prompt when generating DSL from natural language. It serves as
// "schema-as-supervision" -- the LLM uses it as both documentation and
// implicit few-shot signal for the exact surface syntax.
const SchemaReference = `# Semantic Router DSL Reference

You generate programs in the Semantic Router DSL. Output ONLY valid DSL code.
Do NOT wrap the output in markdown code fences. Do NOT add explanatory prose.

## Top-Level Blocks

A DSL program consists of one or more top-level blocks:

  SIGNAL <type> <name> { <fields> }
  ROUTE <name> { <body> }
  DECISION_TREE <name> { IF ... ELSE IF ... ELSE ... }
  MODEL <name> { <fields> }
  PLUGIN <name> <type> { <fields> }
  TEST <name> { <entries> }

IMPORTANT: ROUTE and DECISION_TREE cannot coexist in the same program.
Use ROUTE for priority-based routing or DECISION_TREE for if/else logic.

## Signal Types

Each SIGNAL block declares a detection signal. The type determines what fields are available.

### keyword
Matches queries containing specific keywords.
  Fields: keywords (string[]), operator ("any"|"all"), case_sensitive (bool), method (string),
          fuzzy_match (bool), fuzzy_threshold (int), bm25_threshold (float), ngram_threshold (float)

### embedding
Matches queries semantically similar to candidate phrases.
  Fields: threshold (float 0-1), candidates (string[]), aggregation_method (string)

### domain
Classifies queries into academic/topical domains.
  Fields: description (string), mmlu_categories (string[])

### jailbreak
Detects jailbreak/prompt-injection attempts.
  Fields: method (string), threshold (float 0-1), include_history (bool), description (string),
          jailbreak_patterns (string[]), benign_patterns (string[])

### pii
Detects personally identifiable information.
  Fields: threshold (float 0-1), pii_types_allowed (string[]), include_history (bool), description (string)

### complexity
Classifies query difficulty.
  Fields: threshold (float 0-1), description (string),
          hard: { candidates (string[]) }, easy: { candidates (string[]) }

### language
Detects the language of the query.
  Fields: description (string)

### context
Matches based on conversation context length.
  Fields: min_tokens (string), max_tokens (string), description (string)

### modality
Detects input modality (text, image, audio, etc.).
  Fields: description (string)

### preference
Matches user preference patterns.
  Fields: description (string), examples (string[]), threshold (float 0-1)

### fact_check
Flags queries needing factual verification.
  Fields: description (string)

### user_feedback
Reacts to user feedback signals.
  Fields: description (string)

### authz
Role-based access control signal.
  Fields: role (string), description (string),
          subjects: [{ kind (string), name (string) }]

### kb
Knowledge base retrieval signal.
  Fields: kb (string), target: { kind (string), value (string) }, match (string)

### structure
Detects structural patterns in queries (code, JSON, URLs, etc.).
  Fields: description (string),
          feature: { type (string), source: { type (string), pattern (string), keywords (string[]) } },
          predicate: { gt (float), gte (float), lt (float), lte (float) }

## Route Body

Inside a ROUTE block:

  PRIORITY <int>           -- Higher priority routes are evaluated first
  TIER <int>               -- Logical grouping tier
  DESCRIPTION "text"       -- Human-readable description
  WHEN <bool_expr>         -- Condition for this route to match
  MODEL "<name>" (options) -- Target model(s), comma-separated for multiple
  ALGORITHM <type> { }     -- Model selection algorithm
  PLUGIN <type> { }        -- Plugin configuration

### Model Options
  MODEL "<name>" (reasoning = true, effort = "high", lora = "<name>", weight = 1.0, param_size = "3b")

### Algorithm Types
  confidence, ratings, remom, elo, router_dc, automix, hybrid, rl_driven,
  gmtrouter, latency_aware, static, knn, kmeans, svm

### Plugin Types (inline)
  system_prompt, semantic_cache, hallucination, memory, rag, tools,
  image_gen, fast_response, request_params, router_replay, header_mutation

## Boolean Expressions (WHEN clauses)

  signal_type("signal_name")         -- Reference a declared signal
  expr AND expr                      -- Both must match
  expr OR expr                       -- Either matches
  NOT expr                           -- Negation
  (expr)                             -- Grouping

Operator precedence: NOT > AND > OR. Use parentheses to override.

## DECISION_TREE

For if/else conditional routing (mutually exclusive with ROUTE):

  DECISION_TREE <name> {
    IF <bool_expr> {
      NAME "branch_name"
      MODEL "<model>"
      PLUGIN <type> { ... }
    }
    ELSE IF <bool_expr> {
      NAME "branch_name"
      MODEL "<model>"
    }
    ELSE {
      NAME "default_branch"
      MODEL "<model>"
    }
  }

Each branch MUST have at least one MODEL. The ELSE branch is required.

## PROJECTION

Projections define derived scoring and partitioning over signals.

### partition
  PROJECTION partition <name> {
    semantics: "softmax_exclusive"
    temperature: 1.0
    members: ["signal_a", "signal_b"]
    default: "signal_a"
  }

### score
  PROJECTION score <name> {
    method: "weighted_sum"
    inputs: [{ type: "domain", name: "math", weight: 1.0 }]
  }

### mapping
  PROJECTION mapping <name> {
    source: "score_name"
    method: "threshold_bands"
    outputs: [{ name: "high", gte: 0.8 }, { name: "low", lt: 0.8 }]
  }

## Field Syntax

Fields use key: value pairs. Commas between fields are optional.
  key: "string value"
  key: 42
  key: 3.14
  key: true
  key: ["item1", "item2"]
  key: { nested_key: "value" }

## TEST Blocks

  TEST <name> {
    "query text" -> expected_route_name
  }

## Comments

Lines starting with # are comments.
`

// FewShotExamples contains canonical DSL programs that demonstrate common
// patterns. These are injected into the system prompt alongside the schema
// reference to provide concrete examples of correct syntax.
const FewShotExamples = `# Example 1: Simple domain-based routing

SIGNAL domain math {
  description: "Mathematical and quantitative queries"
}

SIGNAL domain coding {
  description: "Programming and software development queries"
}

ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5-math:7b" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    system_prompt: "You are a math expert. Show your work step by step."
  }
}

ROUTE coding_route {
  PRIORITY 100
  WHEN domain("coding")
  MODEL "deepseek-coder:6.7b"
  PLUGIN system_prompt {
    system_prompt: "You are an expert programmer."
  }
}

ROUTE default_route {
  PRIORITY 1
  MODEL "qwen2.5:7b"
}

# Example 2: Keyword + boolean logic routing

SIGNAL keyword urgent {
  keywords: ["urgent", "emergency", "asap", "critical"]
  operator: "any"
}

SIGNAL keyword billing {
  keywords: ["invoice", "payment", "billing", "charge", "refund"]
  operator: "any"
}

SIGNAL jailbreak detector {
  method: "classifier"
  threshold: 0.8
}

ROUTE jailbreak_block {
  PRIORITY 1000
  WHEN jailbreak("detector")
  MODEL "fast-reject:1b"
  PLUGIN fast_response {
    message: "I cannot process that request."
  }
}

ROUTE urgent_billing {
  PRIORITY 200
  WHEN keyword("urgent") AND keyword("billing")
  MODEL "gpt-4o"
  PLUGIN system_prompt {
    system_prompt: "Handle this urgent billing query with care."
  }
}

ROUTE billing_general {
  PRIORITY 100
  WHEN keyword("billing")
  MODEL "qwen2.5:7b"
}

ROUTE fallback {
  PRIORITY 1
  MODEL "qwen2.5:3b"
}

# Example 3: Decision tree with safety guardrails

SIGNAL jailbreak detector {
  method: "classifier"
  threshold: 0.85
}

SIGNAL domain math {
  description: "Math queries"
}

SIGNAL domain coding {
  description: "Code queries"
}

DECISION_TREE routing_policy {
  IF jailbreak("detector") {
    NAME "jailbreak_block"
    TIER 1
    MODEL "fast-reject:1b"
    PLUGIN fast_response {
      message: "Request blocked for safety."
    }
  }
  ELSE IF domain("math") {
    NAME "math_route"
    TIER 2
    MODEL "qwen2.5-math:7b" (reasoning = true)
  }
  ELSE IF domain("coding") {
    NAME "coding_route"
    TIER 2
    MODEL "deepseek-coder:6.7b"
  }
  ELSE {
    NAME "default_route"
    TIER 3
    MODEL "qwen2.5:7b"
  }
}

# Example 4: Multi-model with algorithm selection

SIGNAL domain math {
  description: "Mathematical reasoning"
}

SIGNAL complexity hard_problem {
  threshold: 0.7
  description: "Complex problems requiring deep reasoning"
}

ROUTE hard_math {
  PRIORITY 200
  WHEN domain("math") AND complexity("hard_problem")
  MODEL "qwen2.5-math:72b" (reasoning = true, effort = "high"),
        "deepseek-r1:70b" (reasoning = true)
  ALGORITHM confidence {
    confidence_method: "logprob"
    threshold: 0.85
    on_error: "fallback"
  }
}

ROUTE easy_math {
  PRIORITY 100
  WHEN domain("math") AND NOT complexity("hard_problem")
  MODEL "qwen2.5-math:7b"
}

ROUTE default {
  PRIORITY 1
  MODEL "qwen2.5:7b"
}

# Example 5: Embedding signals with OR logic

SIGNAL embedding ai_topics {
  threshold: 0.75
  candidates: ["machine learning", "neural networks", "deep learning", "AI safety"]
}

SIGNAL embedding web_dev {
  threshold: 0.75
  candidates: ["React", "CSS layout", "REST API design", "web performance"]
}

ROUTE ai_or_webdev {
  PRIORITY 100
  WHEN embedding("ai_topics") OR embedding("web_dev")
  MODEL "gpt-4o"
}

ROUTE general {
  PRIORITY 1
  MODEL "qwen2.5:3b"
}

# Example 6: Knowledge base routing with projections

SIGNAL kb privacy_policy {
  kb: "privacy_kb"
  target: { kind: "group", value: "privacy_policy" }
  match: "best"
}

PROJECTION score privacy_score {
  method: "weighted_sum"
  inputs: [{ type: "kb", kb: "privacy_kb", metric: "relevance", weight: 1.0, value_source: "score" }]
}

ROUTE privacy_route {
  PRIORITY 200
  WHEN kb("privacy_policy")
  MODEL "qwen2.5:14b"
  PLUGIN rag {
    enabled: true
    backend: "privacy_kb"
    top_k: 5
  }
  PLUGIN tools {
    enabled: true
    mode: "passthrough"
    semantic_selection: true
  }
}

ROUTE default {
  PRIORITY 1
  MODEL "qwen2.5:7b"
}

# Example 7: PII and authz with plugin configuration

SIGNAL pii detector {
  threshold: 0.9
  pii_types_allowed: ["name", "email"]
}

SIGNAL authz admin_only {
  role: "admin"
  subjects: [{ kind: "group", name: "platform-admins" }]
}

ROUTE admin_pii_safe {
  PRIORITY 300
  WHEN authz("admin_only") AND NOT pii("detector")
  MODEL "gpt-4o"
  PLUGIN system_prompt {
    system_prompt: "You have admin access. Answer fully."
  }
}

ROUTE pii_blocked {
  PRIORITY 200
  WHEN pii("detector")
  MODEL "qwen2.5:3b"
  PLUGIN fast_response {
    message: "PII detected. Please remove personal information and try again."
  }
}

ROUTE default {
  PRIORITY 1
  MODEL "qwen2.5:7b"
}
`

// SystemPrompt is the system message used when generating DSL from natural
// language. It pins the role and output format, following DocFlow's pattern
// of keeping the LLM strictly in the target formal language.
const SystemPrompt = `You are a Semantic Router DSL generator. You produce valid Semantic Router DSL programs from natural language descriptions.

Rules:
- Output ONLY valid DSL code. No markdown fences. No explanatory text.
- Every signal referenced in a WHEN clause must be declared as a top-level SIGNAL.
- Every ROUTE or DECISION_TREE branch must have at least one MODEL.
- Include a default/fallback route with PRIORITY 1 when using ROUTE blocks.
- Use quoted strings for model names: MODEL "model-name"
- Use quoted strings in signal references: domain("math"), keyword("urgent")
- Do NOT mix ROUTE and DECISION_TREE in the same program.
- When the user describes if/else or decision tree logic, use DECISION_TREE (not ROUTE with AND NOT chains).
- When the user describes priority-based routing or independent conditions, use ROUTE blocks.
`

// BuildNLPrompt constructs the full user-message prompt for NL-to-DSL generation.
// It concatenates the schema reference, few-shot examples, and the user's
// natural language instruction into one user message.
func BuildNLPrompt(instruction string) string {
	return SchemaReference + "\n" + FewShotExamples + "\n# Task\n\nGenerate a Semantic Router DSL program for the following requirement:\n\n" + instruction + "\n\nGenerate the DSL program:\n"
}

// BuildRepairPrompt constructs a follow-up prompt for the LLM when its
// previous output failed to parse. It includes the bad code, the error,
// and the full schema reference so the model can self-correct.
func BuildRepairPrompt(badCode string, parseErr string) string {
	return SchemaReference + "\n\nThe following DSL program has errors:\n\n" + badCode + "\n\nError: " + parseErr + "\n\nFix the errors and output ONLY the corrected DSL program. Do not wrap in code fences.\n"
}
