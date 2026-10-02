---
license: apache-2.0
base_model: vllm-sr/Decision-1.0-Lux-9B
base_model_relation: finetune
library_name: transformers
tags:
- decision-model
- classification
- system-one
- safetensors
---

![Decision-2.0-Lux-9B](assets/banner.png)

# Decision-2.0-Lux-9B

**Decision-2.0-Lux-9B** is the 9B model of [Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00), the decision models of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). Give it an input (text or JSON) and the questions you need answered: pick one of several options, say yes or no, or rate on a scale. It answers them all at once and returns a probability for every answer, without generating text.

| | |
| --- | --- |
| **Parameters** | 7.94B |
| **Context length** | 16,384 tokens |
| **Decision types** | Choice · Yes / No · Score |
| **License** | Apache-2.0 |

## Highlights

- **Top JevArena score of its size:** 68.0, ahead of the 2 other same-size models compared.
- **Ahead of Decision 1.0 Lux:** +2.2 on JevArena and +1.8 on the Jev Decision Index.
- **Speed:** a median of 19.5 ms per single-question request on a single GPU.
- **Many questions, one pass:** Choice, Yes / No and Score questions about the same input are answered together in one forward pass, with a probability for every option.

## Quickstart

```bash
pip install "transformers>=5.17" torch safetensors
```

```python
import json

from transformers import AutoModel

model = AutoModel.from_pretrained("vllm-sr/Decision-2.0-Lux-9B", trust_remote_code=True)
result = model.system_one(
    state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults"
            }
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?"
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "Routine",
                "Soon",
                "Today"
            ]
        }
    },
)
print(json.dumps(result["answers"], indent=2))

# Or as a pipeline:
# transformers.pipeline("decision", model="vllm-sr/Decision-2.0-Lux-9B", trust_remote_code=True)(state=..., questions=...)
```

## Evaluation

| Model | JevArena ↑ | Human-labelled transfer ↑ | Jev Decision Index ↑ |
| --- | ---: | ---: | ---: |
| **Decision-2.0-Lux-9B** | **68.0** | **57.3** | **45.3** |
| Decision 1.0 Lux | 65.8 | 55.8 | 43.5 |
| Nimble v2 | 62.1 | 53.6 | — |

### JevArena

![JevArena: Decision-2.0-Lux-9B and same-size models](assets/jevarena.png)

![JevArena by decision type: Decision-2.0-Lux-9B and same-size models](assets/jevarena-types.png)

<sub>Every model answers the same frozen prompts, scored the same way; missing or invalid answers count as errors. Human-labelled transfer is the median macro-F1 over 15 human-labelled tasks (×100).</sub>

### Jev Decision Index

![Jev Decision Index against model size](assets/index-pareto.png)

![Jev Decision Index by area: Decision-2.0-Lux-9B and Decision 1.0 Lux](assets/index-areas.png)

<sub>Decision 2.0: independent reproduction with the official 0.2.1 kit on the released weights; others: public board snapshot, 2026-09-28. Training data audited at row level against all Index test items.</sub>

## License

Apache-2.0 ([LICENSE](LICENSE)).

## Citation

```bibtex
@misc{decision_2_0_lux_9b_2026,
  title        = {{Decision-2.0-Lux-9B}: A Decision 2.0 Model for Structured Decisions},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B}}
}
```
