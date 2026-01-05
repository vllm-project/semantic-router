# User Feedback Signal Routing

This guide shows you how to route requests based on user feedback and satisfaction signals. The user_feedback signal helps identify follow-up messages, corrections, and satisfaction levels.

## Key Advantages

- **Adaptive Routing**: Detect when users are unsatisfied and route to better models
- **Correction Handling**: Automatically handle "that's wrong" and "try again" messages
- **Satisfaction Analysis**: Identify positive vs negative feedback
- **Improved UX**: Provide better responses when users indicate dissatisfaction

## What Problem Does It Solve?

Users often provide feedback in follow-up messages:

- **Corrections**: "That's wrong", "No, that's not what I meant"
- **Satisfaction**: "Thank you", "That's helpful", "Perfect"
- **Clarifications**: "Can you explain more?", "I don't understand"
- **Retries**: "Try again", "Give me another answer"

The user_feedback signal automatically identifies these patterns, allowing you to:

1. Route corrections to more capable models
2. Detect satisfaction levels for monitoring
3. Handle follow-up questions appropriately
4. Improve response quality based on feedback

## Configuration

### Basic Configuration

Define user feedback signals in your `config.yaml`:

```yaml
signals:
  user_feedbacks:
    - name: "wrong_answer"
      description: "User indicates previous answer was incorrect"

    - name: "satisfied"
      description: "User is satisfied with the answer"

    - name: "need_clarification"
      description: "User needs more clarification on the answer"

    - name: "want_different"
      description: "User wants some other different answer"
```

### Use in Decision Rules

```yaml
decisions:
  - name: wrong_answer_route
    description: "Handle user feedback indicating wrong answer - rethink and provide correct response"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "wrong_answer"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The user has indicated that the previous answer was incorrect. Please carefully reconsider the question, identify what might have been wrong in the previous response, and provide a corrected and accurate answer. Think step-by-step and verify your reasoning before responding."

  - name: retry_with_different_approach
    description: "Route requests for different approach"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "want_different"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The user wants a different approach or perspective. Provide an alternative solution or explanation that differs from the previous response."
```

## Feedback Types

### 1. Wrong Answer

**Patterns**: "That's wrong", "No", "Incorrect", "Try again"

```yaml
signals:
  user_feedbacks:
    - name: "wrong_answer"
      description: "User indicates previous answer was incorrect"

decisions:
  - name: wrong_answer_route
    description: "Handle user feedback indicating wrong answer"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "wrong_answer"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The user has indicated that the previous answer was incorrect. Please carefully reconsider the question and provide a corrected answer."
```

**Example Queries**:

- "That's wrong, the answer is 42" → ✅ Correction detected
- "No, that's not what I meant" → ✅ Correction detected
- "Try again with a different approach" → ✅ Correction detected

### 2. Satisfied

**Patterns**: "Thank you", "Perfect", "That's helpful", "Great"

```yaml
signals:
  user_feedbacks:
    - name: "satisfied"
      description: "User is satisfied with the answer"

decisions:
  - name: track_satisfaction
    description: "Track satisfied users"
    priority: 50
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "satisfied"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The user is satisfied. Continue providing helpful assistance."
```

**Example Queries**:

- "Thank you, that's exactly what I needed" → ✅ Satisfaction detected
- "Perfect, that helps a lot" → ✅ Satisfaction detected
- "Great explanation" → ✅ Satisfaction detected

### 3. Need Clarification

**Patterns**: "Can you explain more?", "I don't understand", "What do you mean?"

```yaml
signals:
  user_feedbacks:
    - name: "need_clarification"
      description: "User needs more clarification on the answer"

decisions:
  - name: provide_clarification
    description: "Provide more detailed explanation"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "need_clarification"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The user needs more clarification. Provide a more detailed, step-by-step explanation with examples."
```

**Example Queries**:

- "Can you explain that in simpler terms?" → ✅ Clarification needed
- "I don't understand the last part" → ✅ Clarification needed
- "What do you mean by that?" → ✅ Clarification needed

### 4. Want Different Approach

**Patterns**: "Give me another answer", "Try a different way", "Show me alternatives"

```yaml
signals:
  user_feedbacks:
    - name: "want_different"
      description: "User wants some other different answer"

decisions:
  - name: retry_with_different_approach
    description: "Provide alternative solution"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "want_different"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The user wants a different approach or perspective. Provide an alternative solution or explanation that differs from the previous response."
```

**Example Queries**:

- "Give me another way to solve this" → ✅ Alternative wanted
- "Show me a different approach" → ✅ Alternative wanted
- "Can you try a different method?" → ✅ Alternative wanted

## Use Cases

### 1. Customer Support - Escalation

**Problem**: Unsatisfied customers need better responses

```yaml
signals:
  user_feedbacks:
    - name: "wrong_answer"
      description: "Customer indicates previous answer was incorrect"

decisions:
  - name: escalate_to_premium
    description: "Escalate to premium model"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "wrong_answer"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The customer was not satisfied with the previous answer. Provide a better, more accurate response."
```

### 2. Education - Adaptive Learning

**Problem**: Students need different explanations when confused

```yaml
signals:
  user_feedbacks:
    - name: "need_clarification"
      description: "Student needs more clarification on the answer"

decisions:
  - name: detailed_explanation
    description: "Provide detailed explanation"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "need_clarification"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "The student needs more clarification. Provide a detailed, step-by-step explanation with examples."
```

## Best Practices

### 1. Combine with Context

Use conversation history to improve detection:

```yaml
# Track conversation state
context:
  previous_response: true
  conversation_history: 3  # Last 3 messages
```

### 2. Set Escalation Priorities

Corrections should have high priority:

```yaml
decisions:
  - name: handle_correction
    priority: 100  # High priority for corrections
```

### 3. Monitor Satisfaction Rates

Track feedback patterns:

```yaml
logging:
  level: info
  user_feedback: true
  satisfaction_metrics: true
```

### 4. Use Appropriate Models

- **Corrections**: Route to more capable/expensive models
- **Clarifications**: Route to models good at explanations
- **Satisfaction**: Continue with current model

## Reference

See [Signal-Driven Decision Architecture](../../overview/signal-driven-decisions.md) for complete signal architecture.
