---
title: vLLM Semantic Router
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: apache-2.0
---

## Semantic Router classifier playground

This Streamlit app lets you try the small classification models published by
the vLLM Semantic Router project. It includes prompt category, fact-check,
jailbreak, PII, feedback, and tool-call safety models. Choose a model, enter an
example, and inspect its predicted label or token annotations.

The playground demonstrates individual classifiers. It does not start the
Semantic Router, route requests to language models, or reproduce a complete
recipe.

## Run locally

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit. The first request for each classifier
downloads its public model and tokenizer from Hugging Face; later requests
reuse Streamlit's in-process cache. Internet access and enough local memory for
the selected model are therefore required on first use.

## What the results mean

- Sequence classifiers show the selected label, confidence, and scores for all
  labels.
- PII models highlight detected spans. Detection is not redaction and should
  not be treated as a compliance guarantee.
- The tool-call verifier highlights tokens it classifies as unauthorized. It is
  a model demonstration, not a replacement for application authorization.

Model IDs and example inputs are defined in `app.py`. For production routing,
start with the repository's [public documentation](https://vllm-semantic-router.com/docs/intro).
