package config

import (
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// loadOnBase loads a configuration fragment merged onto the model catalog
// documentation base.
func loadOnBase(t *testing.T, fragment string) (*RouterConfig, error) {
	t.Helper()
	var base, overlay map[string]interface{}
	data, err := os.ReadFile(modelCatalogDocsBase)
	if err != nil {
		t.Fatal(err)
	}
	if err = yaml.Unmarshal(data, &base); err != nil {
		t.Fatal(err)
	}
	if err = yaml.Unmarshal([]byte(fragment), &overlay); err != nil {
		t.Fatal(err)
	}
	encoded, err := yaml.Marshal(mergeYAML(base, overlay))
	if err != nil {
		t.Fatal(err)
	}
	return ParseYAMLBytesWithoutEnvExpansion(encoded)
}

const remoteEmbedDeployment = `
      remote-embed: {provider: openai_embeddings, endpoint: https://embedding.example.com/v1, model: BAAI/bge-m3, auth: {api_key_env: EMBEDDING_API_KEY}}`

func TestModelCatalogRefusesWhatCannotWork(t *testing.T) {
	for name, tc := range map[string]struct{ fragment, want string }{
		"a binding to an undeclared deployment": {
			"global: {model_catalog: {bindings: {pii: missing}}}",
			`unknown deployment "missing"; declared deployments: `,
		},
		"a rule naming an undeclared deployment": {
			"routing: {signals: {classifiers: [{name: risk, deployment: missing, labels: [low, high]}]}}",
			`unknown deployment "missing"; declared deployments: `,
		},
		"an embedding model bound to a question-style signal": {
			"global:\n  model_catalog:\n    deployments:" + remoteEmbedDeployment + "\n    bindings: {pii: remote-embed}",
			"an openai_embeddings deployment serves embeddings only",
		},
		"a built-in embedding model bound to a question-style signal": {
			"global: {model_catalog: {bindings: {pii: vela-embedding}}}",
			"answers no question",
		},
		"an early exit on a remote reranker": {
			"global:\n  model_catalog:\n    deployments:\n      remote-ranker: {provider: rerank, endpoint: https://rerank.example.com, model: bge-reranker-v2-m3}\n" +
				"routing:\n  decisions:\n    - name: rag\n      plugins:\n        - type: rag\n          configuration: {enabled: true, backend: vectorstore, rerank: {deployment: remote-ranker, top_k: 3, layer: 22}}",
			"a rerank deployment answers at its own depth",
		},
		"a question model as the default embedding": {
			"global: {model_catalog: {embeddings: {default: primary}}}",
			"serves no embeddings",
		},
		"a remote decision model as the default embedding": {
			"global: {model_catalog: {embeddings: {default: d3-remote}}}",
			"embeddings and reranking need",
		},
		"remote embedding dimensions that disagree with the target dimension": {
			"global:\n  model_catalog:\n    deployments:\n      remote-embed: {provider: openai_embeddings, endpoint: https://embedding.example.com/v1, model: BAAI/bge-m3, capabilities: {dimensions: 1024}}\n" +
				"    embeddings:\n      default: remote-embed\n      semantic: {embedding_config: {target_dimension: 768}}",
			"answers with 1024 dimensions (capabilities.dimensions), but embeddings.semantic.embedding_config.target_dimension asks for 768",
		},
		"a key written inline": {
			"global: {model_catalog: {deployments: {d3-remote: {auth: {api_key: sk-test}}}}}",
			"a key is never written in the configuration",
		},
		"a credential header": {
			"global: {model_catalog: {deployments: {d3-remote: {headers: {Authorization: Bearer sk-test}}}}}",
			"carries a credential",
		},
		"credentials in the endpoint": {
			"global: {model_catalog: {deployments: {d3-remote: {endpoint: 'https://user:secret@app.vllm-sr.ai/v1/systemone'}}}}",
			"must not carry credentials",
		},
		"an unknown signal type as a binding key": {
			"global: {model_catalog: {bindings: {piii: d3-remote}}}",
			"unknown signal type (did you mean pii?)",
		},
		"a deployment on a rule of a shared-answer type": {
			"routing: {signals: {domains: [{name: math, mmlu_categories: [math], deployment: d3-remote}]}}",
			"domain rules share one answer, so their deployment is chosen for the signal type",
		},
		"a binding for a type whose rules ask their own question": {
			"global: {model_catalog: {bindings: {classifier: d3-remote}}}",
			"classifier rules each ask their own question",
		},
		"a binding for the hazard head": {
			"routing: {model_bindings: {hazard: content-risk}}",
			"leave it out to use the pre-declared hazard deployment",
		},
		"a rule with a deployment and a deprecated model path": {
			"routing: {signals: {classifiers: [{name: risk, deployment: content-risk, model_path: models/x, labels: [low, high]}]}}",
			"deployment replaces type, model and model_path",
		},
		"a binding written under its signal type and its earlier name": {
			"global: {model_catalog: {bindings: {pii: kai, pii_classifier: kai}}}",
			"bind the same signal",
		},
		"a systemone question the remote model does not answer": {
			"global: {model_catalog: {deployments: {d3-remote: {capabilities: {question_types: [choice]}}}}}\nrouting: {signals: {systemone: [{name: tools, deployment: d3-remote, question: {type: noul, instructions: \"Does it need a tool?\"}}]}}",
			"noul",
		},
		"complexity rules split between a remote scorer and the decision model": {
			"routing: {signals: {complexity: [{name: a, deployment: content-risk, hard_above: 0.6, easy_below: 0.4}, {name: b, hard_above: 0.6, easy_below: 0.4}]}}",
			"one remote score drives every complexity rule",
		},
		"a masking plugin on PII without spans": {
			"global: {model_catalog: {deployments: {d3-remote: {capabilities: {presets: [halu]}}}, system: {decision_model: {deployment: d3-remote}}}}\nrouting: {signals: {pii: [{name: personal, threshold: 0.5}]}, decisions: [{name: mask, rules: {type: pii, name: personal}, modelRefs: [{model: general}], plugins: [{type: masking, configuration: {enabled: true}}]}]}",
			"the masking plugin replaces PII spans",
		},
		"operating_point on a remote deployment": {
			"global: {model_catalog: {deployments: {content-risk: {operating_point: {path: operating_point.json, sha256: e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5}}}}}",
			"operating_point pins a managed model_runtime head",
		},
		"generation on a deployment that is not openai_chat": {
			"global: {model_catalog: {deployments: {content-risk: {generation: {max_tokens: 64}}}}}",
			"generation applies to openai_chat deployments",
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := loadOnBase(t, tc.fragment)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want one containing %q", err, tc.want)
			}
		})
	}
}

func TestModelCatalogWarnsWhereItStillWorks(t *testing.T) {
	t.Run("a systemone deployment without spans for PII", func(t *testing.T) {
		cfg, err := loadOnBase(t, "global: {model_catalog: {deployments: {d3-remote: {capabilities: {presets: [halu]}}}, system: {decision_model: {deployment: d3-remote}}}}\nrouting: {signals: {pii: [{name: personal, threshold: 0.5}]}, decisions: [{name: guard, rules: {type: pii, name: personal}, modelRefs: [{model: general}]}]}")
		if err != nil {
			t.Fatal(err)
		}
		if warningCodes(cfg)["span_fallback"] == 0 {
			t.Fatalf("no span_fallback warning: %+v", Warnings(cfg))
		}
	})
	t.Run("an earlier module model next to a binding", func(t *testing.T) {
		cfg, err := loadOnBase(t, "global: {model_catalog: {bindings: {domain: kai}, modules: {classifier: {domain: {model_id: models/Vela-1.0-Encoder-307M-Domain}}}}}")
		if err != nil {
			t.Fatal(err)
		}
		if got := resolveDocKey(cfg, "domain"); got != "kai" {
			t.Fatalf("domain resolves to %q; the binding must win", got)
		}
		found := false
		for _, warning := range Warnings(cfg) {
			found = found || strings.Contains(warning.Message, "already names the deployment")
		}
		if !found {
			t.Fatalf("no warning that the model_id is ignored: %+v", Warnings(cfg))
		}
	})
}
