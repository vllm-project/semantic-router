# Signal Fusion Engine

The Signal Fusion Engine is a policy-driven decision-making system that combines multiple signals into actionable routing decisions. It provides configurable boolean expression parsing, priority-based rule evaluation, and short-circuit evaluation.

## Features

### 1. Boolean Expression Parser
- Supports complex boolean logic: `&&` (AND), `||` (OR), `!` (NOT)
- Handles comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Parentheses for grouping expressions
- Signal references in format: `provider.name.field`

### 2. Priority-Based Rule Evaluation
- Rules are evaluated in priority order (highest first)
- Priority levels (recommended):
  - **200**: Safety blocks (SSN, credit cards, PII)
  - **150**: High-confidence routing overrides (keyword + regex matches)
  - **100**: Category boosting (embedding similarity signals)
  - **50**: Consensus requirements (multiple signals must agree)
  - **0**: Default fallthrough

### 3. Short-Circuit Evaluation
- First matching rule wins
- No further evaluation after a match
- Efficient for common cases

### 4. Action Types
- **block**: Immediately reject requests
- **route**: Route to specific model candidates
- **boost_category**: Apply weight multipliers to categories
- **fallthrough**: Use default behavior (BERT classification)

## Usage Example

```go
package main

import (
    "fmt"
    "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/fusion"
)

func main() {
    // Create a signal context
    context := fusion.NewSignalContext()
    
    // Add signals from various providers
    context.AddSignal(fusion.Signal{
        Provider: "keyword",
        Name:     "kubernetes",
        Matched:  true,
    })
    
    context.AddSignal(fusion.Signal{
        Provider: "similarity",
        Name:     "infrastructure",
        Score:    0.85,
        Matched:  true,
    })
    
    // Define policy rules
    policy := &fusion.Policy{
        Rules: []fusion.Rule{
            {
                Name:      "safety-block",
                Condition: "regex.ssn.matched || regex.credit-card.matched",
                Action:    fusion.ActionBlock,
                Priority:  200,
                Message:   "PII detected - request blocked",
            },
            {
                Name:      "k8s-routing",
                Condition: "keyword.kubernetes.matched && similarity.infrastructure.score > 0.75",
                Action:    fusion.ActionRoute,
                Priority:  150,
                Models:    []string{"k8s-expert", "devops-model"},
            },
            {
                Name:      "boost-reasoning",
                Condition: "similarity.reasoning.score > 0.75",
                Action:    fusion.ActionBoostCategory,
                Priority:  100,
                Category:  "reasoning",
                BoostWeight: 1.5,
            },
        },
    }
    
    // Create engine and evaluate
    engine := fusion.NewEngine(policy)
    result, err := engine.Evaluate(context)
    
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    // Handle result
    if result.Matched {
        fmt.Printf("Rule matched: %s\n", result.MatchedRule)
        fmt.Printf("Action: %s\n", result.Action)
        
        switch result.Action {
        case fusion.ActionBlock:
            fmt.Printf("Blocked: %s\n", result.Message)
        case fusion.ActionRoute:
            fmt.Printf("Route to models: %v\n", result.Models)
        case fusion.ActionBoostCategory:
            fmt.Printf("Boost %s by %.2fx\n", result.Category, result.BoostWeight)
        }
    } else {
        fmt.Println("No rules matched - fallthrough to default behavior")
    }
}
```

## Expression Syntax

### Signal References
Signals are referenced using the format: `provider.name.field`

**Fields:**
- `matched`: Boolean indicating if the signal matched
- `score`: Numeric value (e.g., similarity score, confidence)
- `value`: String value (e.g., category name)

**Examples:**
```
keyword.kubernetes.matched
similarity.reasoning.score
bert.category.value
```

### Boolean Operators
- `&&` (AND): Both conditions must be true
- `||` (OR): At least one condition must be true
- `!` (NOT): Negates the condition

**Examples:**
```
keyword.kubernetes.matched && keyword.security.matched
keyword.docker.matched || keyword.kubernetes.matched
!regex.pii.matched
```

### Comparison Operators
- `==`: Equal to
- `!=`: Not equal to
- `>`: Greater than
- `<`: Less than
- `>=`: Greater than or equal to
- `<=`: Less than or equal to

**Examples:**
```
similarity.reasoning.score > 0.75
similarity.infrastructure.score >= 0.8
bert.category.value == 'computer science'
bert.confidence != 0.5
```

### Complex Expressions
Combine operators and use parentheses for grouping:

```
keyword.kubernetes.matched && (similarity.infrastructure.score > 0.8 || regex.k8s-pattern.matched)
(keyword.security.matched && bert.category.value == 'security') || similarity.security.score > 0.9
!regex.pii.matched && (keyword.safe.matched || similarity.safe.score > 0.7)
```

## Policy Configuration

A policy consists of multiple rules that are evaluated in priority order:

```go
policy := &fusion.Policy{
    Rules: []fusion.Rule{
        {
            Name:      "unique-rule-identifier",
            Condition: "boolean expression",
            Action:    fusion.ActionType,
            Priority:  100,
            // Action-specific fields:
            Models:      []string{"model1", "model2"},  // For ActionRoute
            Category:    "category-name",                // For ActionBoostCategory
            BoostWeight: 1.5,                            // For ActionBoostCategory
            Message:     "Block message",                // For ActionBlock
        },
    },
}
```

## Testing

The package includes comprehensive unit tests using Ginkgo/Gomega:

```bash
cd src/semantic-router
go test -v ./pkg/utils/fusion/
```

Test coverage includes:
- Simple signal references
- Boolean operators (AND, OR, NOT)
- Comparison operators (==, !=, >, <, >=, <=)
- Complex nested expressions
- Priority-based evaluation
- Short-circuit behavior
- All action types
- Edge cases and error handling

## Architecture

### Components

1. **Types** (`types.go`)
   - `Signal`: Represents a single signal from a provider
   - `SignalContext`: Container for all available signals
   - `Rule`: Defines a policy rule with condition and action
   - `Policy`: Collection of rules
   - `EvaluationResult`: Result of policy evaluation

2. **Expression Evaluator** (`expression.go`)
   - Tokenizes and parses boolean expressions
   - Evaluates expressions against signal context
   - Supports complex nested logic

3. **Policy Engine** (`engine.go`)
   - Manages policy rules
   - Sorts rules by priority
   - Implements short-circuit evaluation
   - Returns first matching rule result

### Design Principles

- **Configurable**: All aspects are configurable through data structures
- **Efficient**: Short-circuit evaluation and priority ordering minimize work
- **Flexible**: Expression language supports complex routing logic
- **Type-Safe**: Strong typing for actions and signals
- **Tested**: Comprehensive test coverage

## Integration with Semantic Router

The Signal Fusion Engine is designed to integrate with the existing semantic router architecture:

1. **Signal Providers** gather signals:
   - Keyword matcher (in-tree)
   - Regex scanner (in-tree)
   - Embedding similarity (in-tree or MCP)
   - BERT classifier (existing)

2. **Signal Context** is built from provider results

3. **Fusion Engine** evaluates policy against context

4. **Routing Decision** is made based on evaluation result:
   - Block: Reject request
   - Route: Select specific models
   - Boost: Adjust category weights for BERT
   - Fallthrough: Use standard BERT classification

## Performance Characteristics

- **Expression Parsing**: O(n) where n = expression length
- **Rule Evaluation**: O(r × e) where r = number of rules, e = expression complexity
- **Short-Circuit**: Average case terminates after first match
- **Priority Sorting**: One-time O(r log r) cost at engine creation

Typical performance:
- Simple expressions (1-2 conditions): < 100 microseconds
- Complex expressions (5+ conditions): < 500 microseconds
- Full policy evaluation (10 rules): < 1 millisecond

## Future Enhancements

Potential improvements for future versions:
- Expression compilation/caching for repeated evaluations
- Support for regular expressions in string comparisons
- Mathematical operations in expressions (addition, subtraction, etc.)
- Built-in functions (e.g., `contains()`, `matches()`)
- Validation of expressions at policy load time
- Metrics and observability integration
