# Decision 2.0 4B independent release preview review

This review concerns `llm-semantic-router/DEV2.0-4B`, revision
`checkpoint-0232`, with 4,205,751,296 loaded parameters. It is bound to the
frozen JevArena v3 release context
`6ac24db123b6fd4b2ffb7a05d9b4a87f16f624d0ce981603760c92c376b8bb19`
and release record
`1d1287f0c62cf283de2a663500e148f2657553cac682df9fec0bfe50a2d96ddd`.
The reviewed deterministic package preview manifest is
`e3dedcb84a089c56996e20fa8bd76e74dae9329ac2ffadc8ee92e9f829d299cb`.

## Independent checks

| Check | Result | Private review receipt SHA-256 |
| --- | --- | --- |
| Training/evaluation overlap | Passed for observable record IDs and exact/normalized/near context checks | `77ad87d02bce5c1d94389038331c7fd7174bc8a12956af069f631c71a8386fba` |
| Native inference parity | Passed on 1,600 DEV and 1,430 human pilot items, with zero category changes and zero measured probability drift | `53b4ee033936d088dcb427c81fc082480445a1911e0e8aced98526250441112a` |
| Rights and provenance preview | Passed the exact Apache-2.0 preview and inherited notice-byte review | `8840c7e50c21b1eb199ee1f6bc2b848783f177ba418d39f385c574c5cad53c1a` |

The overlap audit compared 7,455 training rows with typed FINAL (1,600),
15-task human transfer (6,547), and the public decision subset (231). It found
zero matching record IDs, raw or normalized contexts, or candidates satisfying
the preregistered near-duplicate rule in each panel. Some training and
evaluation items belong to related task families. Approximate screening
cannot prove the absence of semantic paraphrases or exposure during upstream
pretraining.

The release record binds the same 22 native files, the same calibration file,
the same pinned [Eikos-4B](https://huggingface.co/caiovicentino1/Eikos-4B)
revision, and its [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
ancestry. All 22 candidate files were independently rehashed. The preview
has a standard Apache-2.0 top-level `LICENSE` and a modification `NOTICE`;
the inherited model licenses and original notice retain their source bytes
inside the native package. The model card identifies those inherited terms,
does not distribute training records, and contains no private infrastructure
or credentials. Source-level rights and permissions remain in the private
training provenance. This is a fact-specific provenance review, not a legal
opinion; downloadable weights retain a language-model head, so regeneration
of training text is not ruled out.

**Publication is still pending exact-byte review of the assembled package and
the exported Hugging Face repository.** The three passed receipts are bound to
the deterministic preview and release context. They do not substitute for
verifying that the emitted weights, calibration, model card, licenses, notice,
and artwork match their reviewed hashes.
