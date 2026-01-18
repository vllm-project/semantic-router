---
sidebar_position: 1
---

# What are our Goals?

We are building the **System Level Intelligence** for Mixture-of-Models (MoM), bringing **Collective Intelligence** into **LLM systems**.

## Core Questions

Our project addresses five fundamental challenges in LLM systems:

### 1. How to capture the missing signals?

In traditional LLM routing, we only look at the user's query text. But there's so much more information we're missing:

- **Context signals**: What domain is this query about? (math, code, creative writing?)
- **Quality signals**: Does this query need fact-checking? Is the user giving feedback?
- **User signals**: What are the user's preferences? What's their satisfaction level?

**Our solution**: A comprehensive signal extraction system that captures 8 types of signals from requests, responses, and context.

### 2. How to combine the signals?

Having multiple signals is great, but how do we use them together to make better decisions?

- Should we route to the math model if we detect **both** math keywords **and** math domain?
- Should we enable fact-checking if we detect **either** a factual question **or** a sensitive domain?

**Our solution**: A flexible decision engine with AND/OR operators that lets you combine signals in powerful ways.

### 3. How to collaborate more efficiently?

Different models are good at different things. How do we make them work together as a team?

- Route math questions to specialized math models
- Route creative writing to models with better creativity
- Route code questions to models trained on code
- Use smaller models for simple tasks, larger models for complex ones

**Our solution**: Intelligent routing that matches queries to the best model based on multiple signals, not just simple rules.

### 4. How to secure the system?

LLM systems face unique security challenges:

- **Jailbreak attacks**: Adversarial prompts trying to bypass safety guardrails
- **PII leaks**: Accidentally exposing sensitive personal information
- **Hallucinations**: Models generating false or misleading information

**Our solution**: A plugin chain architecture with multiple security layers (jailbreak detection, PII filtering, hallucination detection).

### 5. How to collect valuable signals?

The system should learn and improve over time:

- Track which signals lead to better routing decisions
- Collect user feedback to improve signal detection
- Build a self-learning system that gets smarter with use

**Our solution**: Comprehensive observability and feedback collection that feeds back into the signal extraction and decision engine.

## The Vision

We envision a future where:

- **LLM systems are intelligent at the system level**, not just at the model level
- **Multiple models collaborate seamlessly**, each contributing their strengths
- **Security is built-in**, not bolted on
- **Systems learn and improve** from every interaction
- **Collective intelligence emerges** from the combination of signals, decisions, and feedback

## Why This Matters

### For Developers

- Build more capable LLM applications with less effort
- Leverage multiple models without complex orchestration
- Get built-in security and compliance

### For Organizations

- Reduce costs by routing to appropriate models
- Improve quality through specialized model selection
- Meet compliance requirements with built-in PII and security controls

### For Users

- Get better, more accurate responses
- Experience faster response times through caching
- Benefit from improved safety and privacy

## Next Steps

Learn more about the core concepts:

- [What is Semantic Router?](semantic-router-overview.md) - Understanding semantic routing
- [What is Collective Intelligence?](collective-intelligence.md) - How signals create intelligence
- [What is Signal-Driven Decision?](signal-driven-decisions.md) - Deep dive into the decision engine
