package config

import "testing"

func TestRerankerBindingRequiresNativePairContract(t *testing.T) {
	decl := ModelBinding{Deployment: "model", Contract: RelevanceScoresContract, Adapter: "vela_reranker", PairScorer: &PairScorerSelection{Layer: 22, Dimension: 768}}
	deployment := ModelDeployment{Artifact: "model", Provider: "candle"}.WithDefaults()
	if err := validateTaskModelBinding(RAGRerankerConsumer, decl, deployment); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"probability", "foreign_task", "mapping", "separate_head", "negative", "truncate", "remote"} {
		t.Run(kind, func(t *testing.T) {
			d, b, name := deployment, decl, RAGRerankerConsumer
			switch kind {
			case "probability":
				b.Contract = RemoteClassifierContractLabelDistribution
			case "foreign_task":
				name = "embedding"
				b.Contract = "embedding.v1"
			case "mapping":
				b.MappingPath = "map.json"
			case "separate_head":
				b.Head = "head"
			case "negative":
				b.PairScorer = &PairScorerSelection{Layer: -1}
			case "truncate":
				d.Input.Overflow = "truncate"
			case "remote":
				d.Provider = "http"
			}
			if err := validateTaskModelBinding(name, b, d); err == nil {
				t.Fatal("accepted incompatible contract")
			}
		})
	}
}

func TestRAGNeuralRerankBoundsAndRecipeBinding(t *testing.T) {
	two, three := 2, 3
	rag := RAGPluginConfig{Enabled: true, Backend: "vectorstore", TopK: &three, Rerank: &RAGRerankConfig{TopK: &two}}
	if err := rag.Validate(); err != nil {
		t.Fatal(err)
	}
	decision := Decision{Name: "retrieve", Plugins: []DecisionPlugin{{Type: "rag", Configuration: MustStructuredPayload(rag)}}}
	cfg := &RouterConfig{}
	if err := validateDecisionRAGAndMemoryPlugins(cfg, &decision); err == nil {
		t.Fatal("accepted missing same-recipe scorer")
	}
	cfg.ModelBindings = map[string]ModelBinding{RAGRerankerConsumer: {}}
	if err := validateDecisionRAGAndMemoryPlugins(cfg, &decision); err != nil {
		t.Fatal(err)
	}
	rag.Backend = "qdrant"
	if err := rag.Validate(); err == nil {
		t.Fatal("accepted string-only retriever rerank")
	}
	rag.Backend = "vectorstore"
	rag.TopK = &two
	rag.Rerank.TopK = &three
	if err := rag.Validate(); err == nil {
		t.Fatal("accepted more retained hits than candidates")
	}
}
