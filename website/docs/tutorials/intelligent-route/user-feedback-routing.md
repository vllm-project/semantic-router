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

```yaml
signals:
  user_feedbacks:
    - name: "correction_needed"
      description: "User indicates previous answer was wrong"
    
    - name: "satisfied"
      description: "User is satisfied with the answer"
    
    - name: "need_clarification"
      description: "User needs more explanation"
    
    - name: "want_different"
      description: "User wants a different approach"
```

### Use in Decision Rules

```yaml
decisions:
  - name: retry_with_better_model
    description: "Route corrections to more capable model"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "user_feedback"
          name: "correction_needed"
        - type: "user_feedback"
          name: "want_different"
    modelRefs:
      - model: premium-model  # More capable model
        weight: 1.0
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          prompt: "The user was not satisfied with the previous answer. Provide a better, more detailed response."
```

## Feedback Types

### 1. Correction Needed

**Patterns**: "That's wrong", "No", "Incorrect", "Try again"

```yaml
signals:
  user_feedbacks:
    - name: "correction_needed"
      description: "User indicates previous answer was wrong"

decisions:
  - name: handle_correction
    description: "Route corrections to better model"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "user_feedback"
          name: "correction_needed"
    modelRefs:
      - model: premium-model
        weight: 1.0
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
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "user_feedback"
          name: "satisfied"
    modelRefs:
      - model: standard-model
        weight: 1.0
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
      description: "User needs more explanation"

decisions:
  - name: provide_clarification
    description: "Provide more detailed explanation"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "user_feedback"
          name: "need_clarification"
    modelRefs:
      - model: detailed-model
        weight: 1.0
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          prompt: "The user needs more clarification. Provide a more detailed, step-by-step explanation."
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
      description: "User wants a different approach"

decisions:
  - name: alternative_approach
    description: "Provide alternative solution"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "user_feedback"
          name: "want_different"
    modelRefs:
      - model: creative-model
        weight: 1.0
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          prompt: "Provide an alternative approach or solution. Be creative and think differently."
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
    - name: "correction_needed"
      description: "Customer is unsatisfied"

decisions:
  - name: escalate_to_premium
    description: "Escalate to premium model"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "user_feedback"
          name: "correction_needed"
    modelRefs:
      - model: premium-support-model
        weight: 1.0
```

### 2. Education - Adaptive Learning

**Problem**: Students need different explanations when confused

```yaml
signals:
  user_feedbacks:
    - name: "need_clarification"
      description: "Student needs more explanation"

decisions:
  - name: detailed_explanation
    description: "Provide detailed explanation"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "user_feedback"
          name: "need_clarification"
    modelRefs:
      - model: educational-model
        weight: 1.0
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
