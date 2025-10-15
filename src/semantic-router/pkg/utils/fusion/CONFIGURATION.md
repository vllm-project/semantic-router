# Signal Fusion Engine Configuration Guide

This guide shows how to configure and use the Signal Fusion Engine with the semantic router.

## Table of Contents

1. [Configuration Structure](#configuration-structure)
2. [YAML Configuration Examples](#yaml-configuration-examples)
3. [Integration with Router](#integration-with-router)
4. [Complete Examples](#complete-examples)

## Configuration Structure

### Adding to RouterConfig

To integrate the fusion engine, add the following to your `RouterConfig` in `pkg/config/config.go`:

```go
type RouterConfig struct {
    // ... existing fields ...
    
    // Content Scanning and Signal Fusion configuration
    ContentScanning ContentScanningConfig `yaml:"content_scanning,omitempty"`
}

// ContentScanningConfig represents the signal fusion configuration
type ContentScanningConfig struct {
    // Enable/disable the entire content scanning system
    Enabled bool `yaml:"enabled"`
    
    // Default action when no rules match (fallthrough or block)
    DefaultAction string `yaml:"default_action,omitempty"`
    
    // Enable audit logging for policy decisions
    AuditLogging bool `yaml:"audit_logging,omitempty"`
    
    // In-tree signal providers
    Providers ProvidersConfig `yaml:"providers,omitempty"`
    
    // Fusion policy configuration
    FusionPolicy FusionPolicyConfig `yaml:"fusion_policy"`
}

// ProvidersConfig represents signal provider configurations
type ProvidersConfig struct {
    // Keyword matching provider
    Keyword KeywordProviderConfig `yaml:"keyword,omitempty"`
    
    // Regex scanning provider
    Regex RegexProviderConfig `yaml:"regex,omitempty"`
    
    // Embedding similarity provider
    Similarity SimilarityProviderConfig `yaml:"similarity,omitempty"`
}

// KeywordProviderConfig represents keyword matching configuration
type KeywordProviderConfig struct {
    Enabled   bool   `yaml:"enabled"`
    RulesPath string `yaml:"rules_path,omitempty"`
}

// RegexProviderConfig represents regex scanning configuration
type RegexProviderConfig struct {
    Enabled      bool   `yaml:"enabled"`
    PatternsPath string `yaml:"patterns_path,omitempty"`
    Engine       string `yaml:"engine,omitempty"` // "re2" or "stdlib"
}

// SimilarityProviderConfig represents embedding similarity configuration
type SimilarityProviderConfig struct {
    Enabled         bool    `yaml:"enabled"`
    ConceptsPath    string  `yaml:"concepts_path,omitempty"`
    DefaultThreshold float64 `yaml:"default_threshold,omitempty"`
}

// FusionPolicyConfig represents fusion policy configuration
type FusionPolicyConfig struct {
    // Path to fusion policy rules file
    RulesPath string `yaml:"rules_path"`
    
    // Inline rules (alternative to rules_path)
    Rules []FusionRule `yaml:"rules,omitempty"`
}

// FusionRule represents a single fusion policy rule
type FusionRule struct {
    Name        string   `yaml:"name"`
    Condition   string   `yaml:"condition"`
    Action      string   `yaml:"action"` // "block", "route", "boost_category", "fallthrough"
    Priority    int      `yaml:"priority"`
    Models      []string `yaml:"models,omitempty"`
    Category    string   `yaml:"category,omitempty"`
    BoostWeight float64  `yaml:"boost_weight,omitempty"`
    Message     string   `yaml:"message,omitempty"`
}
```

## YAML Configuration Examples

### Basic Configuration

Add this to your `config.yaml`:

```yaml
content_scanning:
  enabled: true
  default_action: fallthrough
  audit_logging: true
  
  providers:
    keyword:
      enabled: true
      rules_path: "config/fusion/keyword_rules.yaml"
    
    regex:
      enabled: true
      patterns_path: "config/fusion/regex_patterns.yaml"
      engine: "re2"
    
    similarity:
      enabled: true
      concepts_path: "config/fusion/similarity_concepts.yaml"
      default_threshold: 0.75
  
  fusion_policy:
    rules_path: "config/fusion/policy_rules.yaml"
```

### Inline Policy Rules

Alternatively, define rules inline:

```yaml
content_scanning:
  enabled: true
  default_action: fallthrough
  
  fusion_policy:
    rules:
      # Safety blocks - highest priority
      - name: "block-ssn"
        condition: "regex.ssn.matched"
        action: "block"
        priority: 200
        message: "Request contains SSN pattern and cannot be processed"
      
      - name: "block-credit-card"
        condition: "regex.credit-card.matched"
        action: "block"
        priority: 200
        message: "Request contains credit card pattern and cannot be processed"
      
      # High-confidence routing
      - name: "route-kubernetes"
        condition: "keyword.kubernetes.matched && similarity.infrastructure.score > 0.75"
        action: "route"
        priority: 150
        models: ["k8s-expert", "devops-model"]
      
      - name: "route-security"
        condition: "keyword.security.matched && bert.category.value == 'computer science'"
        action: "route"
        priority: 150
        models: ["security-hardened-model"]
      
      # Category boosting
      - name: "boost-reasoning"
        condition: "similarity.reasoning.score > 0.75"
        action: "boost_category"
        priority: 100
        category: "reasoning"
        boost_weight: 1.5
      
      # Default fallthrough
      - name: "default-fallthrough"
        condition: "!regex.ssn.matched"
        action: "fallthrough"
        priority: 0
```

### External Rules Files

**config/fusion/keyword_rules.yaml:**
```yaml
rules:
  - name: "kubernetes-infrastructure"
    keywords: ["kubernetes", "k8s", "kubectl", "helm", "pod", "deployment"]
    operator: "OR"
    case_sensitive: false
  
  - name: "security"
    keywords: ["security", "vulnerability", "CVE", "exploit", "firewall"]
    operator: "OR"
    case_sensitive: false
  
  - name: "docker"
    keywords: ["docker", "container", "dockerfile", "docker-compose"]
    operator: "OR"
    case_sensitive: false
```

**config/fusion/regex_patterns.yaml:**
```yaml
patterns:
  - name: "ssn"
    pattern: '\b\d{3}-\d{2}-\d{4}\b'
    description: "Social Security Number pattern"
  
  - name: "credit-card"
    pattern: '\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    description: "Credit card number pattern"
  
  - name: "cve-id"
    pattern: 'CVE-\d{4}-\d{4,7}'
    description: "CVE identifier pattern"
```

**config/fusion/similarity_concepts.yaml:**
```yaml
concepts:
  - name: "reasoning"
    keywords:
      - "step by step"
      - "solve this problem"
      - "explain your reasoning"
      - "break down the solution"
    threshold: 0.75
    aggregate_method: "mean"
  
  - name: "infrastructure"
    keywords:
      - "deploy kubernetes cluster"
      - "configure infrastructure"
      - "set up cloud resources"
    threshold: 0.75
    aggregate_method: "max"
```

**config/fusion/policy_rules.yaml:**
```yaml
rules:
  # Safety blocks (Priority 200)
  - name: "safety-pii-block"
    condition: "regex.ssn.matched || regex.credit-card.matched"
    action: "block"
    priority: 200
    message: "PII detected - request blocked for security"
  
  # High-confidence routing (Priority 150)
  - name: "k8s-expert-routing"
    condition: "keyword.kubernetes-infrastructure.matched && similarity.infrastructure.score > 0.8"
    action: "route"
    priority: 150
    models: ["k8s-expert", "devops-specialist"]
  
  - name: "security-routing"
    condition: "(keyword.security.matched || regex.cve-id.matched) && bert.category.value == 'computer science'"
    action: "route"
    priority: 150
    models: ["security-hardened-model"]
  
  # Category boosting (Priority 100)
  - name: "boost-reasoning-category"
    condition: "similarity.reasoning.score > 0.75"
    action: "boost_category"
    priority: 100
    category: "reasoning"
    boost_weight: 1.5
  
  # Consensus routing (Priority 50)
  - name: "multi-signal-consensus"
    condition: "keyword.kubernetes-infrastructure.matched && similarity.infrastructure.score > 0.8 && bert.category.value == 'computer science'"
    action: "route"
    priority: 50
    models: ["consensus-k8s-expert"]
  
  # Default fallthrough (Priority 0)
  - name: "default-bert"
    condition: "!regex.ssn.matched && !regex.credit-card.matched"
    action: "fallthrough"
    priority: 0
```

## Integration with Router

### Step 1: Load Configuration

Add config loading in your router initialization:

```go
import (
    "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
    "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/fusion"
)

// Load router configuration
cfg, err := config.LoadConfig("config/config.yaml")
if err != nil {
    return err
}

// Initialize fusion engine if enabled
var fusionEngine *fusion.Engine
if cfg.ContentScanning.Enabled {
    fusionEngine, err = initializeFusionEngine(cfg)
    if err != nil {
        return fmt.Errorf("failed to initialize fusion engine: %w", err)
    }
}
```

### Step 2: Initialize Fusion Engine

```go
func initializeFusionEngine(cfg *config.RouterConfig) (*fusion.Engine, error) {
    // Convert config rules to fusion policy
    policy := &fusion.Policy{
        Rules: make([]fusion.Rule, 0, len(cfg.ContentScanning.FusionPolicy.Rules)),
    }
    
    for _, cfgRule := range cfg.ContentScanning.FusionPolicy.Rules {
        rule := fusion.Rule{
            Name:        cfgRule.Name,
            Condition:   cfgRule.Condition,
            Action:      fusion.ActionType(cfgRule.Action),
            Priority:    cfgRule.Priority,
            Models:      cfgRule.Models,
            Category:    cfgRule.Category,
            BoostWeight: cfgRule.BoostWeight,
            Message:     cfgRule.Message,
        }
        policy.Rules = append(policy.Rules, rule)
    }
    
    // Create and return engine
    return fusion.NewEngine(policy), nil
}
```

### Step 3: Gather Signals

```go
func gatherSignals(query string, cfg *config.RouterConfig) (*fusion.SignalContext, error) {
    ctx := fusion.NewSignalContext()
    
    // Gather keyword signals
    if cfg.ContentScanning.Providers.Keyword.Enabled {
        keywordSignals := detectKeywords(query, cfg.ContentScanning.Providers.Keyword.RulesPath)
        for _, sig := range keywordSignals {
            ctx.AddSignal(sig)
        }
    }
    
    // Gather regex signals
    if cfg.ContentScanning.Providers.Regex.Enabled {
        regexSignals := detectRegexPatterns(query, cfg.ContentScanning.Providers.Regex.PatternsPath)
        for _, sig := range regexSignals {
            ctx.AddSignal(sig)
        }
    }
    
    // Gather similarity signals
    if cfg.ContentScanning.Providers.Similarity.Enabled {
        similaritySignals := computeSimilarity(query, cfg.ContentScanning.Providers.Similarity.ConceptsPath)
        for _, sig := range similaritySignals {
            ctx.AddSignal(sig)
        }
    }
    
    // Add BERT classification signal (existing classifier)
    bertResult := classifyWithBERT(query)
    ctx.AddSignal(fusion.Signal{
        Provider: "bert",
        Name:     "category",
        Value:    bertResult.Category,
        Score:    bertResult.Confidence,
        Matched:  bertResult.Confidence > cfg.Classifier.CategoryModel.Threshold,
    })
    
    return ctx, nil
}
```

### Step 4: Evaluate Policy and Route

```go
func routeRequest(query string, cfg *config.RouterConfig, engine *fusion.Engine) ([]string, error) {
    // Gather all signals
    signalCtx, err := gatherSignals(query, cfg)
    if err != nil {
        return nil, fmt.Errorf("failed to gather signals: %w", err)
    }
    
    // Evaluate fusion policy
    result, err := engine.Evaluate(signalCtx)
    if err != nil {
        return nil, fmt.Errorf("failed to evaluate fusion policy: %w", err)
    }
    
    // Log decision if audit logging is enabled
    if cfg.ContentScanning.AuditLogging {
        logPolicyDecision(result, query)
    }
    
    // Handle action
    switch result.Action {
    case fusion.ActionBlock:
        return nil, fmt.Errorf("request blocked: %s", result.Message)
    
    case fusion.ActionRoute:
        return result.Models, nil
    
    case fusion.ActionBoostCategory:
        // Apply boost to BERT category weights
        return routeWithBoost(query, result.Category, result.BoostWeight, cfg)
    
    case fusion.ActionFallthrough:
        // Use standard BERT classification
        return routeWithBERT(query, cfg)
    
    default:
        return nil, fmt.Errorf("unknown action type: %s", result.Action)
    }
}
```

## Complete Examples

### Example 1: Safety-First Configuration

Prioritize blocking PII and security threats:

```yaml
content_scanning:
  enabled: true
  default_action: fallthrough
  audit_logging: true
  
  providers:
    regex:
      enabled: true
      patterns_path: "config/fusion/security_patterns.yaml"
  
  fusion_policy:
    rules:
      - name: "block-pii"
        condition: "regex.ssn.matched || regex.credit-card.matched || regex.email.matched"
        action: "block"
        priority: 200
        message: "PII detected"
      
      - name: "default"
        condition: "!regex.ssn.matched"
        action: "fallthrough"
        priority: 0
```

### Example 2: Specialized Routing Configuration

Route to expert models based on topic detection:

```yaml
content_scanning:
  enabled: true
  
  providers:
    keyword:
      enabled: true
      rules_path: "config/fusion/topic_keywords.yaml"
    similarity:
      enabled: true
      concepts_path: "config/fusion/topic_concepts.yaml"
  
  fusion_policy:
    rules:
      - name: "kubernetes-expert"
        condition: "keyword.kubernetes.matched && similarity.infrastructure.score > 0.75"
        action: "route"
        priority: 150
        models: ["k8s-expert-v1", "k8s-expert-v2"]
      
      - name: "database-expert"
        condition: "keyword.database.matched && similarity.database.score > 0.8"
        action: "route"
        priority: 150
        models: ["db-expert", "sql-specialist"]
      
      - name: "fallback"
        condition: "keyword.kubernetes.matched == false"
        action: "fallthrough"
        priority: 0
```

### Example 3: Multi-Signal Consensus

Require multiple signals to agree before routing:

```yaml
content_scanning:
  enabled: true
  
  providers:
    keyword:
      enabled: true
      rules_path: "config/fusion/keywords.yaml"
    similarity:
      enabled: true
      concepts_path: "config/fusion/concepts.yaml"
  
  fusion_policy:
    rules:
      - name: "high-confidence-routing"
        condition: "keyword.topic.matched && similarity.topic.score > 0.85 && bert.category.value == 'computer science'"
        action: "route"
        priority: 100
        models: ["expert-model"]
      
      - name: "medium-confidence-boost"
        condition: "keyword.topic.matched && similarity.topic.score > 0.7"
        action: "boost_category"
        priority: 50
        category: "computer science"
        boost_weight: 1.3
      
      - name: "default"
        condition: "keyword.topic.matched == false"
        action: "fallthrough"
        priority: 0
```

## Testing Your Configuration

Test your configuration with a simple script:

```go
package main

import (
    "fmt"
    "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
    "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/fusion"
)

func main() {
    // Load config
    cfg, _ := config.LoadConfig("config/config.yaml")
    
    // Initialize engine
    engine, _ := initializeFusionEngine(cfg)
    
    // Test with sample signals
    ctx := fusion.NewSignalContext()
    ctx.AddSignal(fusion.Signal{
        Provider: "keyword",
        Name:     "kubernetes",
        Matched:  true,
    })
    ctx.AddSignal(fusion.Signal{
        Provider: "similarity",
        Name:     "infrastructure",
        Score:    0.85,
        Matched:  true,
    })
    
    // Evaluate
    result, _ := engine.Evaluate(ctx)
    
    fmt.Printf("Matched Rule: %s\n", result.MatchedRule)
    fmt.Printf("Action: %s\n", result.Action)
    if result.Action == fusion.ActionRoute {
        fmt.Printf("Models: %v\n", result.Models)
    }
}
```

## Best Practices

1. **Priority Levels**: Use consistent priority levels across your organization:
   - 200: Safety blocks
   - 150: High-confidence routing
   - 100: Category boosting
   - 50: Multi-signal consensus
   - 0: Default fallthrough

2. **Audit Logging**: Always enable audit logging in production to track policy decisions

3. **Testing**: Test your policies with various inputs before deploying to production

4. **Gradual Rollout**: Start with fallthrough actions and gradually add blocking/routing rules

5. **Performance**: Keep expression complexity reasonable - simpler expressions evaluate faster

6. **Documentation**: Document each rule's purpose and the business logic behind it
