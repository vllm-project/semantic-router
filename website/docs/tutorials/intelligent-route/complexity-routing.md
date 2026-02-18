---
sidebar_position: 9
---

# Complexity Routing Tutorial

This tutorial shows you how to use **Complexity Signals** to route requests based on their difficulty level.

This is useful for:

- Routing complex queries to powerful, specialized models
- Routing simple queries to fast, efficient models
- Optimizing cost by using cheaper models for easy tasks
- Improving response quality by matching query difficulty to model capability

## Scenario

We want to:

1. Route simple coding questions (e.g., "print hello world") to a fast model (`llama-3-8b`)
2. Route medium complexity questions to a standard model (`llama-3-70b`)
3. Route complex questions (e.g., "design distributed system") to a specialized model (`deepseek-coder-v2`)

## Step 1: Define Domain Signals (Prerequisite)

First, define domain signals to classify the query type. This is **required** for using composers with complexity signals:

```yaml
signals:
  domains:
    - name: "computer_science"
      description: "Programming, algorithms, software engineering, system design"
      mmlu_categories:
        - "computer_science"
        - "machine_learning"

    - name: "math"
      description: "Mathematics, calculus, algebra, statistics"
      mmlu_categories:
        - "mathematics"
        - "statistics"
```

## Step 2: Define Complexity Signals with Composers

Add `complexity` rules to your `signals` configuration. **IMPORTANT**: Always use `composer` to filter based on domain to prevent cross-domain misclassification:

```yaml
signals:
  complexity:
    - name: "code_complexity"
      composer:
        operator: "AND"
        conditions:
          - type: "domain"
            name: "computer_science"
      threshold: 0.1
      description: "Detects code complexity level based on task difficulty"
      hard:
        candidates:
          - "design distributed system"
          - "implement consensus algorithm"
          - "optimize for scale"
          - "architect microservices"
          - "fix race condition"
          - "implement garbage collector"
      easy:
        candidates:
          - "print hello world"
          - "loop through array"
          - "read file"
          - "sort list"
          - "string concatenation"
          - "simple function"

    - name: "math_complexity"
      composer:
        operator: "AND"
        conditions:
          - type: "domain"
            name: "math"
      threshold: 0.1
      description: "Detects mathematical problem complexity"
      hard:
        candidates:
          - "prove mathematically"
          - "derive the equation"
          - "formal proof"
          - "solve differential equation"
          - "prove by induction"
          - "analyze convergence"
      easy:
        candidates:
          - "add two numbers"
          - "calculate percentage"
          - "simple arithmetic"
          - "basic algebra"
          - "count items"
          - "find average"
```

**Why use composer?** Without composer, a math query like "prove by induction" might incorrectly match `code_complexity` if it has higher similarity to code candidates. The composer ensures `code_complexity` only activates when the domain is "computer_science".

## Step 3: Define Decisions

Create decisions that trigger based on complexity levels:

```yaml
decisions:
  - name: "simple_code_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "complexity"
          name: "code_complexity:easy"
    modelRefs:
      - model: "llama-3-8b"

  - name: "medium_code_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "complexity"
          name: "code_complexity:medium"
    modelRefs:
      - model: "llama-3-70b"

  - name: "complex_code_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "complexity"
          name: "code_complexity:hard"
    modelRefs:
      - model: "deepseek-coder-v2"
```

## Step 4: Combined Logic (Advanced)

You can combine complexity signals with other signals (like domain or language).

**Example**: Route complex **Chinese** coding tasks to a specialized model:

```yaml
decisions:
  - name: "complex_chinese_code"
    priority: 20  # Higher priority
    rules:
      operator: "AND"
      conditions:
        - type: "complexity"
          name: "code_complexity:hard"
        - type: "language"
          name: "zh"
    modelRefs:
      - model: "qwen-coder-plus"
```

## How Complexity Classification Works

The complexity signal uses a **two-phase evaluation process**:

### Phase 1: Parallel Signal Evaluation

All complexity rules are evaluated **independently and in parallel** with other signals:

1. Query is compared to all **hard candidates** → max_hard_similarity
2. Query is compared to all **easy candidates** → max_easy_similarity
3. Difficulty signal = max_hard_similarity - max_easy_similarity
4. Classification:
   - If signal > threshold (0.1): **hard**
   - If signal < -threshold (-0.1): **easy**
   - Otherwise: **medium**

At this stage, **all rules** that match are kept (e.g., both "code_complexity:hard" and "math_complexity:hard" might match).

### Phase 2: Composer Filtering

After all signals are computed, composer conditions are evaluated:

1. For each matched complexity rule, check if it has a `composer`
2. If composer exists, evaluate its conditions against other signal results
3. Only keep rules whose composer conditions are satisfied
4. This prevents cross-domain misclassification

### Example Flow

**Query**: "How do I implement a distributed consensus algorithm?"

**Phase 1 - Parallel Evaluation:**

1. **Domain Signal**: Matches "computer_science" (evaluated in parallel)
2. **code_complexity**:
   - max_hard_similarity = 0.85 (matches "implement consensus algorithm")
   - max_easy_similarity = 0.15 (low match to easy candidates)
   - difficulty_signal = 0.85 - 0.15 = 0.70
   - Result: 0.70 > 0.1 → **"code_complexity:hard"** (tentative)
3. **math_complexity**:
   - max_hard_similarity = 0.25 (some similarity to "algorithm")
   - max_easy_similarity = 0.10
   - difficulty_signal = 0.25 - 0.10 = 0.15
   - Result: 0.15 > 0.1 → **"math_complexity:hard"** (tentative)

**Phase 2 - Composer Filtering:**

1. **code_complexity** composer check:
   - Requires: domain = "computer_science"
   - Domain signal matched "computer_science" ✅
   - **KEPT**: "code_complexity:hard"
2. **math_complexity** composer check:
   - Requires: domain = "math"
   - Domain signal matched "computer_science" (not "math") ❌
   - **FILTERED OUT**

**Final Result**: "code_complexity:hard"

**Routing**: Matches decision → Routes to `deepseek-coder-v2`

## Best Practices

1. **Always Use Composer**: Configure a composer for each complexity rule to filter based on domain. This is **critical** to prevent cross-domain misclassification.
2. **Define Domain Signals First**: Complexity composers depend on domain signals, so define domains before complexity rules.
3. **Diverse Candidates**: Include varied examples in hard/easy candidates to cover different query patterns.
4. **Tune Threshold**: Adjust threshold (default 0.1) based on your use case. Lower threshold = more sensitive classification.
5. **Monitor Results**: Check routing decisions in response headers to refine candidates and thresholds.
6. **Multiple Composer Conditions**: You can use multiple conditions with AND/OR operators:

   ```yaml
   composer:
     operator: "OR"
     conditions:
       - type: "domain"
         name: "computer_science"
       - type: "keyword"
         name: "coding_keywords"
   ```

7. **Description is Optional**: The `description` field is now optional and only used for documentation. It does not affect classification.

## Monitoring

The complexity signal returns results in the format: `"rule_name:difficulty"`

You can check the routing decision in response headers:

```http
x-vsr-matched-complexity: code_complexity:hard
```

This helps you monitor and debug complexity-based routing decisions.
