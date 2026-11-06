# Router DC (removed)

## Overview

The `router_dc` selector was removed, with its
`global.router.model_selection.router_dc` settings. The
[System One](./systemone) selector chooses by the candidates' descriptions.

## What Problem Does It Solve?

It embedded the request and each model description, then selected the
candidate with the strongest similarity. The `systemone` selector asks the
decision model which described candidate should answer.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[System One](./systemone).

## Configuration

Run `vllm-sr config migrate` to rewrite it. A decision with two or more
candidates becomes the `systemone` selector; a decision with one candidate
becomes `static`. Describe each candidate in its model card, or in
`algorithm.systemone.candidates`:

```yaml
algorithm:
  type: systemone
  systemone:
    instructions: Which candidate model best fits this request?
```
