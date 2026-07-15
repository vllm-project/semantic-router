# Complexity Signal

## Overview

`complexity` estimates whether a prompt needs a harder reasoning path or a cheaper easy path. It maps to `config/signal/complexity/` and is declared under `routing.signals.complexity`.

This family is learned: the classifier compares requests against hard and easy examples using embedding similarity, and can optionally use multimodal candidates.

## Key Advantages

- Separates reasoning escalation from domain classification.
- Reuses one complexity policy across multiple decisions.
- Supports hard/easy examples that are easy to tune over time.
- Lets simple prompts stay on cheaper models while hard prompts escalate.

## What Problem Does It Solve?

Topic alone does not tell you whether a prompt needs strong reasoning. Two questions in the same domain can have very different reasoning depth.

`complexity` solves that by estimating task difficulty directly from example-driven signal rules.

## When to Use

Use `complexity` when:

- some prompts need stronger reasoning or longer chains of thought
- easy traffic should stay on cheaper models
- you want escalation policies that are independent of domain
- multimodal reasoning requests need different handling from simple prompts

## Configuration

Source fragment family: `config/signal/complexity/`

```yaml
global:
  model_catalog:
    modules:
      complexity:
        prototype_scoring:
          enabled: true
          cluster_similarity_threshold: 0.9
          max_prototypes: 8
          best_weight: 0.75
          top_m: 2
          margin_threshold: 0.0
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.75
        description: Escalate multi-step reasoning or synthesis-heavy prompts.
        hard:
          candidates:
            - solve this step by step
            - compare multiple tradeoffs
            - analyze the root cause
        easy:
          candidates:
            - answer briefly
            - quick summary
            - simple rewrite
```

Use `complexity` with representative hard and easy examples so the learned boundary matches your real routing cost profile. `prototype_scoring` is a family-level config under `global.model_catalog.modules.complexity`, so every complexity rule shares the same prototype-bank construction and label-scoring policy. Each rule still builds separate hard and easy prototype banks before computing the hard-vs-easy margin.

## Trained Classifier Mode (`method: model`)

By default a complexity rule is decided by the embedding-similarity path above. A rule can instead be decided by a fine-tuned **3-class text classifier** (`easy`/`medium`/`hard`) by setting `method: model`. This mirrors how the [Jailbreak](./jailbreak) signal supports both a trained model and a contrastive path: the choice is per rule, and both modes emit the same `rule:difficulty` output, so decisions and projections are unchanged.

Set `method: model` on the rule and bind the model under `global.model_catalog`:

```yaml
global:
  model_catalog:
    modules:
      complexity:
        # prototype_scoring still applies to embedding-mode rules.
        classifier:
          model_id: your-org/complexity-classifier # HF repo id or local path
          use_cpu: false
          # class-index -> difficulty mapping (required for model mode)
          complexity_mapping_path: models/complexity-classifier/complexity_mapping.json
routing:
  signals:
    complexity:
      - name: needs_reasoning
        method: model
        threshold: 0.6
        description: Escalate hard prompts using the trained complexity classifier.
```

The mapping file maps the model's class indices to difficulty labels. Several naming conventions are accepted, so a checkpoint's existing mapping file can be used directly — `idx_to_label`/`label_to_idx`, HuggingFace `id_to_label`/`label_to_id`, and the category-classifier convention `idx_to_category`/`category_to_idx` (the format shipped in `category_mapping.json` alongside merged classifier checkpoints):

```json
{ "idx_to_category": { "0": "easy", "1": "hard", "2": "medium" } }
```

How it works:

- The classifier runs once per request and predicts the top difficulty class plus a confidence.
- The rule reports `name:<difficulty>` using the predicted class label.
- `threshold` is a **confidence floor**: when the top-class confidence is below it, the rule reports the neutral `medium` band instead (a threshold of `0` always reports the predicted class). This matches the embedding path's neutral-band semantics.

Notes:

- Model mode is **text-only**; image/multimodal fusion remains an embedding-mode capability.
- You can mix modes: some complexity rules can use `method: model` while others stay on the default embedding path.
- If a rule sets `method: model` but no classifier is configured under `modules.complexity.classifier`, the rule is inert (a startup warning is logged).

## Routing on a complexity rule

A complexity rule classifies each request into one of three difficulty levels — `hard`, `easy`, or `medium` — and emits the match as `<rule>:<level>`. Decision conditions must therefore reference the rule **with the difficulty suffix**, not by the bare rule name:

```yaml
routing:
  decisions:
    - name: escalate_hard_prompts
      priority: 160
      rules:
        operator: OR
        conditions:
          - type: complexity
            name: needs_reasoning:hard # required form: <rule>:<hard|easy|medium>
      modelRefs:
        - model: qwen3-30b
          use_reasoning: true
```

A bare `name: needs_reasoning` never matches at runtime — the classifier only ever emits `needs_reasoning:hard|easy|medium` — so config validation rejects the suffix-less form with a clear error.
