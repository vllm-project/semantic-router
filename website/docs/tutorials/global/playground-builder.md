---
title: Build a Mixture in Playground
description: Design, test, and publish a Mixture-of-Models with the vLLM-SR Agent.
---

## Overview

Playground can turn a routing goal into a tested Mixture-of-Models without making
you assemble the Recipe by hand. The Agent works against the Models, component
schemas, and built-in examples installed in your Router, so its suggestions stay
aligned with the version you are running.

## What Problem Does It Solve?

Builder keeps design, evidence, and publication in one durable conversation. You
describe the workload; the Agent discovers the current Router vocabulary, prepares
a model-free Recipe, assigns authorized Models through an Entrypoint, tests the
result, and presents one exact publication review.

## When to Use

Use Builder when you want to create or tune a Mixture-of-Models interactively. Use
the visual and DSL editors when you already know the exact component graph you want.

You need:

- at least one connected Model available to your User or Team;
- an Agent model that supports tools and streaming; and
- permission to use Builder. Publishing also requires routing publish access.

Model credentials remain in the Router credential store. They are never copied
into the conversation, Recipe, or Entrypoint.

## Configuration

### Start in Builder

1. Open **Playground**.
2. Select **+**, then turn on **Builder**.
3. Choose an authorized Model or existing Mixture as the Agent model.
4. Describe the workload and the outcome you want.

A useful first request is concrete but does not prescribe the implementation:

```text
Build a model path for a multilingual support assistant. Keep routine replies
fast, send difficult reasoning to the strongest model, and route image questions
to a vision-capable model. Test English, Chinese, ambiguous, and image requests.
```

The Agent reads the live Signal, Projection, Decision, Algorithm, and Plugin
catalogs as it works. It can reuse built-in Recipes, inspect the Models you are
allowed to use, and explain a proposed path without exposing connection details.

### Tune with evidence

Ask the Agent to probe boundary cases instead of accepting the first draft. It can:

- validate the Recipe and show its topology;
- run bounded multilingual, modality, tool-use, and routing-boundary probes;
- evaluate the draft against an authorized suite; and
- revise the Recipe and Entrypoint assignments while preserving the conversation.

Probe and evaluation results are retained as session artifacts. Reopening the
conversation restores its checkpoint, current resource revisions, unresolved
goals, and previous evidence. A browser refresh or Router replica change does not
start the design over.

### Review and publish

When the draft is ready, Builder presents one review containing the exact Recipe
and Entrypoint changes, Model assignments, validation results, and publication
digest. Review it, then select **Publish**.

The Agent cannot approve its own work. The Router checks your permission again and
rejects the publication if any reviewed revision changed. A successful publication
appears in the Playground model selector and in authenticated `GET /v1/models`;
you can invoke it through the same inference API as every other Entrypoint.

### Access and usage

Every Agent model call, probe, and published-model request uses the ordinary Router
access path. Model visibility, Team and User policy, RPM and TPM, token and cost
budgets, request logs, and actual post-response usage settlement all apply. The
Dashboard is only a client of these Router APIs; another console can provide the
same workflow through the versioned Management API.

For the underlying resources and endpoints, see the
[Management API](../../api/management) and
[Models, Recipes, and Entrypoints](./models-entrypoints-serving).
