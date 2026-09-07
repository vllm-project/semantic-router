# Complexity Signal

## Overview

`complexity` estimates whether a request is `easy`, `medium`, or `hard` by
comparing it with configured example sets. It is independent of topic: two
requests in the same domain can still need different model tiers.

## Key Advantages

- Separates estimated difficulty from topic classification.
- Reuses one easy/medium/hard policy across multiple decisions.
- Tunes routing with examples instead of a custom classifier schema.

## What Problem Does It Solve?

Domain routing alone cannot distinguish a short factual request from a
multi-step analysis request. Complexity supplies a reusable difficulty signal
so decisions can escalate only the traffic that needs it.

## When to Use

Use complexity when model cost or reasoning mode should vary with estimated
task difficulty. Do not treat it as a correctness or safety guarantee; use
domain-specific evaluation and safety signals for those concerns.

## Configuration

```yaml
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.10
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

`threshold` is a margin, not a similarity cutoff. The router scores the request
against the `hard` and the `easy` candidate banks, then compares the two:

```text
signal = hard_bank_score - easy_bank_score

signal >  threshold  -> hard
signal < -threshold  -> easy
otherwise            -> medium
```

Because the two banks can assign similar baseline scores, subtracting their
scores may produce a margin much smaller than either individual score. In one
calibration run using the candidate banks above, the largest observed absolute
margin across eight prompts was `0.197`; none reached the `hard` or `easy` band
with a threshold of `0.75`.

Treat `0.10` as a starting point rather than a universal default. Measure the
margin on representative traffic and tune the threshold for the configured
candidate banks and embedding model.

A rule emits a suffixed name. Decisions must reference
`<rule>:easy`, `<rule>:medium`, or `<rule>:hard`:

```yaml
routing:
  decisions:
    - name: escalate-hard-prompts
      description: Route hard prompts to the reasoning model.
      priority: 150
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: needs_reasoning:hard
      modelRefs:
        - model: reasoning-model
          use_reasoning: true
```

For optional prototype-bank tuning, configure the family-level module once:

```yaml
global:
  model_catalog:
    modules:
      complexity:
        prototype_scoring:
          enabled: true
          max_prototypes: 8
          top_m: 2
```

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
- A rule with `method: model` requires the classifier configuration (`model_id` and `complexity_mapping_path`); config validation rejects the rule at load time when either is missing, so a model-mode route can never be silently inert.

## Dependencies and Limitations

- Complexity uses the configured semantic embedding runtime. A remote embedding
  provider receives the request text used for classification.
- Candidate phrases and thresholds must be calibrated together against labeled
  traffic. Re-evaluate them whenever the embedding model changes.
- Ambiguous prompts can land in the `medium` band; always define a route or
  fallback for every band you rely on.
- See a complete example:
  [`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml).
