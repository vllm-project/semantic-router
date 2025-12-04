# Signal Fusion Engine

A policy-driven routing engine that combines multiple signal sources using complex boolean expressions.

## Features

- ✅ **Boolean Expression Parser** - Full support for AND (`&&`), OR (`||`), NOT (`!`), comparisons
- ✅ **Short-Circuit Evaluation** - Stops evaluating as soon as result is determined
- ✅ **Priority-Based** - Policies evaluated highest-to-lowest priority
- ✅ **Multi-Signal Fusion** - Combine keyword, regex, similarity, and BERT signals
- ✅ **Type-Safe** - Strong typing throughout
- ✅ **Well-Tested** - 53 comprehensive test cases

## Quick Start

### 1. Create Policies

```go
import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/fusion"

policies := []fusion.FusionPolicy{
    {
        Name:      "block-ssn",
        Priority:  200,
        Condition: "regex.ssn.matched",
        Action:    fusion.ActionBlock,
        Message:   "SSN detected",
    },
    {
        Name:      "route-k8s",
        Priority:  150,
        Condition: "keyword.k8s.matched && bert.category == 'computer_science'",
        Action:    fusion.ActionRoute,
        Models:    []string{"k8s-expert-model"},
    },
}
```

### 2. Create Engine

```go
engine, err := fusion.NewFusionEngine(policies)
if err != nil {
    log.Fatalf("Failed to create fusion engine: %v", err)
}
```

### 3. Prepare Signal Context

```go
ctx := fusion.NewSignalContext()
ctx.KeywordMatches["k8s"] = true
ctx.RegexMatches["ssn"] = false
ctx.BERTCategory = "computer_science"
ctx.BERTConfidence = 0.95
```

### 4. Evaluate

```go
result, err := engine.Evaluate(ctx)
if err != nil {
    log.Fatalf("No policy matched: %v", err)
}

switch result.Action {
case fusion.ActionBlock:
    fmt.Printf("Blocked: %s\n", result.Message)
case fusion.ActionRoute:
    fmt.Printf("Route to: %v\n", result.Models)
case fusion.ActionBoost:
    fmt.Printf("Boost %s by %.1fx\n", result.Category, result.BoostWeight)
case fusion.ActionFallthrough:
    fmt.Println("Use existing decision engine")
}
```

## Expression Syntax

### Signal References

```javascript
// Keyword matches (boolean)
keyword.{rule_name}.matched

// Regex matches (boolean)
regex.{pattern_name}.matched

// Similarity scores (float)
similarity.{concept_name}.score

// BERT classification (string or float)
bert.category
bert.confidence
```

### Operators

```javascript
// Boolean
&&  // AND
||  // OR
!   // NOT

// Comparison
==  // Equals
!=  // Not equals
>   // Greater than
<   // Less than
>=  // Greater than or equal
<=  // Less than or equal

// Grouping
()  // Parentheses
```

### Examples

```javascript
// Simple boolean
keyword.k8s.matched

// Comparison
similarity.reasoning.score > 0.75

// Complex expression
(keyword.k8s.matched || keyword.kubernetes.matched) &&
bert.category == "computer_science" &&
!regex.pii.matched
```

## Action Types

### Block

Blocks the request with a custom message.

```go
{
    Action:  fusion.ActionBlock,
    Message: "Request blocked due to PII detection",
}
```

### Route

Routes to specific model candidates.

```go
{
    Action: fusion.ActionRoute,
    Models: []string{"k8s-expert-model", "devops-model"},
}
```

### Boost Category

Boosts the weight of a specific category.

```go
{
    Action:      fusion.ActionBoost,
    Category:    "reasoning",
    BoostWeight: 1.5, // 50% boost
}
```

### Fallthrough

Falls through to existing decision engine.

```go
{
    Action: fusion.ActionFallthrough,
}
```

## Priority Levels

Recommended priority ranges:

- **200:** Safety blocks (SSN, credit cards, PII)
- **150:** High-confidence routing (keyword + BERT matches)
- **100:** Category boosting (embedding similarity)
- **50:** Consensus requirements (multiple signals agree)
- **0:** Default fallthrough

## Testing

Run the comprehensive test suite:

```bash
cd src/semantic-router
go test -v ./pkg/fusion/...

# Output:
# Running Suite: Fusion Suite
# Will run 53 of 53 specs
# SUCCESS! -- 53 Passed | 0 Failed
```

## Configuration

See `config/fusion-example.yaml` for a complete configuration example.

```yaml
fusion:
  enabled: true
  policies:
    - name: "route-k8s-expert"
      priority: 150
      condition: "keyword.k8s.matched && bert.category == 'computer_science'"
      action: "route"
      models:
        - "k8s-expert-model"
```

## Performance

- **AST Parsing:** O(n) - cached at initialization
- **Policy Evaluation:** O(m) - short-circuit optimized
- **Latency:** < 1ms typical expression evaluation

## Documentation

- **Architecture:** `docs/signal-fusion-architecture.md`
- **Implementation Summary:** `docs/signal-fusion-implementation-summary.md`
- **GitHub Issue:** [#367](https://github.com/vllm-project/semantic-router/issues/367)

## License

Apache 2.0 - See LICENSE file for details
