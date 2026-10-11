package config

import "testing"

func TestRerankerBindingRequiresNativePairContract(t *testing.T) {
	decl := ModelBinding{Deployment: "model", Contract: RelevanceScoresContract, Adapter: "vela_reranker", PairScorer: &PairScorerSelection{Layer: 22, Dimension: 768}}
	deployment := ModelDeployment{Artifact: "model", Provider: ModelRuntimeProvider}.WithDefaults()
	if err := validateTaskModelBinding(RAGRerankerConsumer, decl, deployment); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"probability", "foreign_task", "mapping", "negative", "truncate", "remote"} {
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

func TestRAGDecisionBindingIsExplicitAndRejectsNativeOptions(t *testing.T) {
	deployment := ModelDeployment{Provider: ModelRuntimeProvider}.WithDefaults()
	binding := ModelBinding{Deployment: "judge", Contract: DecisionTaskContract}
	if err := validateTaskModelBinding(RAGRerankerConsumer, binding, deployment); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"adapter", "head", "mapping", "exit", "operating", "remote", "truncate"} {
		t.Run(kind, func(t *testing.T) {
			b, d := binding, deployment
			switch kind {
			case "adapter":
				b.Adapter = "vela_reranker"
			case "head":
				b.Head = "default"
			case "mapping":
				b.MappingPath = "labels.json"
			case "exit":
				b.PairScorer = &PairScorerSelection{Layer: 22}
			case "operating":
				b.OperatingPoint = &OperatingPointReference{}
			case "remote":
				d.Provider = "http"
			case "truncate":
				d.Input.Overflow = "truncate"
			}
			if err := validateTaskModelBinding(RAGRerankerConsumer, b, d); err == nil {
				t.Fatal("accepted incompatible decision binding")
			}
		})
	}
}

func TestRAGDecisionCandidateCapIsValidatedBeforeServing(t *testing.T) {
	count := MaxDecisionRAGCandidates + 1
	rag := RAGPluginConfig{Enabled: true, Backend: "vectorstore", TopK: &count, Rerank: &RAGRerankConfig{}}
	decision := Decision{Name: "retrieve", Plugins: []DecisionPlugin{{Type: "rag", Configuration: MustStructuredPayload(rag)}}}
	cfg := &RouterConfig{}
	cfg.ModelBindings = map[string]ModelBinding{RAGRerankerConsumer: {Contract: DecisionTaskContract}}
	if err := validateDecisionRAGAndMemoryPlugins(cfg, &decision); err == nil {
		t.Fatal("accepted oversized decision ranking batch")
	}
	cfg.ModelBindings[RAGRerankerConsumer] = ModelBinding{Contract: RelevanceScoresContract}
	if err := validateDecisionRAGAndMemoryPlugins(cfg, &decision); err != nil {
		t.Fatalf("changed native reranker bounds: %v", err)
	}
}
