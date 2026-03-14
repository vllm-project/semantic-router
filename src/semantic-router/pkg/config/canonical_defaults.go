package config

// DefaultCanonicalGlobal returns the router-owned runtime defaults used when
// canonical config omits all or part of the global block.
func DefaultCanonicalGlobal() CanonicalGlobal {
	defaults := CanonicalGlobal{
		AutoModelName:             "MoM",
		IncludeConfigModelsInList: false,
		ClearRouteCache:           true,
		InlineModels: InlineModels{
			EmbeddingModels: EmbeddingModels{
				MmBertModelPath: "models/mom-embedding-ultra",
				UseCPU:          true,
				HNSWConfig: HNSWConfig{
					ModelType:         "mmbert",
					PreloadEmbeddings: true,
					TargetDimension:   768,
					TargetLayer:       22,
					MinScoreThreshold: 0.5,
				},
			},
		},
		ResponseAPI: ResponseAPIConfig{
			Enabled:      true,
			StoreBackend: "memory",
			TTLSeconds:   86400,
			MaxResponses: 1000,
		},
		RouterReplay: RouterReplayConfig{
			StoreBackend: "memory",
			TTLSeconds:   2592000,
			AsyncWrites:  false,
		},
		Memory: MemoryConfig{
			Enabled:                    false,
			AutoStore:                  false,
			Milvus:                     MemoryMilvusConfig{Collection: "agentic_memory", Dimension: 384},
			DefaultRetrievalLimit:      5,
			DefaultSimilarityThreshold: 0.70,
			ExtractionBatchSize:        10,
		},
		SemanticCache: SemanticCache{
			Enabled:        true,
			BackendType:    "memory",
			MaxEntries:     1000,
			TTLSeconds:     3600,
			EvictionPolicy: "fifo",
		},
		Tools: ToolsConfig{
			Enabled:         false,
			TopK:            3,
			ToolsDBPath:     "config/tools_db.json",
			FallbackToEmpty: true,
		},
		Observability: ObservabilityConfig{
			Metrics: MetricsConfig{
				Enabled: canonicalBoolPtr(true),
			},
			Tracing: TracingConfig{
				Enabled:  true,
				Provider: "opentelemetry",
				Exporter: TracingExporterConfig{
					Type:     "otlp",
					Endpoint: "vllm-sr-jaeger:4317",
					Insecure: true,
				},
				Sampling: TracingSamplingConfig{
					Type: "always_on",
					Rate: 1.0,
				},
				Resource: TracingResourceConfig{
					ServiceName:           "vllm-sr",
					ServiceVersion:        "v0.3.0",
					DeploymentEnvironment: "development",
				},
			},
		},
		Looper: LooperConfig{
			Endpoint:       "http://localhost:8899/v1/chat/completions",
			TimeoutSeconds: 1200,
			Headers:        map[string]string{},
		},
		ModelSelection: ModelSelectionConfig{
			Enabled: true,
			Method:  "knn",
		},
		SystemModels: DefaultSystemModels(),
	}

	defaults.InlineModels.PromptGuard = PromptGuardConfig{
		Enabled:              true,
		ModelID:              defaults.SystemModels.PromptGuard,
		Threshold:            0.7,
		UseCPU:               true,
		UseMmBERT32K:         true,
		JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
	}
	defaults.InlineModels.Classifier = Classifier{
		CategoryModel: CategoryModel{
			ModelID:             defaults.SystemModels.DomainClassifier,
			Threshold:           0.5,
			UseCPU:              true,
			UseMmBERT32K:        true,
			CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
		},
		PIIModel: PIIModel{
			ModelID:        defaults.SystemModels.PIIClassifier,
			Threshold:      0.9,
			UseCPU:         true,
			UseMmBERT32K:   true,
			PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
		},
	}
	defaults.InlineModels.HallucinationMitigation = HallucinationMitigationConfig{
		Enabled: false,
		FactCheckModel: FactCheckModelConfig{
			ModelID:      defaults.SystemModels.FactCheckClassifier,
			Threshold:    0.6,
			UseCPU:       true,
			UseMmBERT32K: true,
		},
		HallucinationModel: HallucinationModelConfig{
			ModelID:                defaults.SystemModels.HallucinationDetector,
			Threshold:              0.8,
			UseCPU:                 true,
			MinSpanLength:          2,
			MinSpanConfidence:      0.6,
			ContextWindowSize:      50,
			EnableNLIFiltering:     true,
			NLIEntailmentThreshold: 0.75,
		},
		NLIModel: NLIModelConfig{
			ModelID:   defaults.SystemModels.HallucinationExplainer,
			Threshold: 0.9,
			UseCPU:    true,
		},
	}
	defaults.InlineModels.FeedbackDetector = FeedbackDetectorConfig{
		Enabled:      true,
		ModelID:      defaults.SystemModels.FeedbackDetector,
		Threshold:    0.7,
		UseCPU:       true,
		UseMmBERT32K: true,
	}

	enabledSoftMatching := true
	defaults.InlineModels.EmbeddingModels.HNSWConfig.EnableSoftMatching = &enabledSoftMatching

	return defaults
}

// DefaultSystemModels returns stable capability bindings for built-in runtime models.
func DefaultSystemModels() CanonicalSystemModels {
	return CanonicalSystemModels{
		PromptGuard:            "models/mmbert32k-jailbreak-detector-merged",
		DomainClassifier:       "models/mmbert32k-intent-classifier-merged",
		PIIClassifier:          "models/mmbert32k-pii-detector-merged",
		FactCheckClassifier:    "models/mmbert32k-factcheck-classifier-merged",
		HallucinationDetector:  "models/mom-halugate-detector",
		HallucinationExplainer: "models/mom-halugate-explainer",
		FeedbackDetector:       "models/mmbert32k-feedback-detector-merged",
	}
}

// DefaultGlobalConfig materializes canonical global defaults into the runtime RouterConfig.
func DefaultGlobalConfig() RouterConfig {
	global := DefaultCanonicalGlobal()
	cfg := RouterConfig{}
	_ = applyCanonicalGlobal(&cfg, &global)
	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}
	return cfg
}

func applySystemModelOverrides(resolved *CanonicalGlobal, defaults *CanonicalGlobal) {
	if resolved == nil || defaults == nil {
		return
	}

	if resolved.SystemModels.PromptGuard != defaults.SystemModels.PromptGuard &&
		resolved.PromptGuard.ModelID == defaults.PromptGuard.ModelID {
		resolved.PromptGuard.ModelID = resolved.SystemModels.PromptGuard
	}
	if resolved.SystemModels.DomainClassifier != defaults.SystemModels.DomainClassifier &&
		resolved.Classifier.CategoryModel.ModelID == defaults.Classifier.CategoryModel.ModelID {
		resolved.Classifier.CategoryModel.ModelID = resolved.SystemModels.DomainClassifier
	}
	if resolved.SystemModels.PIIClassifier != defaults.SystemModels.PIIClassifier &&
		resolved.Classifier.PIIModel.ModelID == defaults.Classifier.PIIModel.ModelID {
		resolved.Classifier.PIIModel.ModelID = resolved.SystemModels.PIIClassifier
	}
	if resolved.SystemModels.FactCheckClassifier != defaults.SystemModels.FactCheckClassifier &&
		resolved.HallucinationMitigation.FactCheckModel.ModelID == defaults.HallucinationMitigation.FactCheckModel.ModelID {
		resolved.HallucinationMitigation.FactCheckModel.ModelID = resolved.SystemModels.FactCheckClassifier
	}
	if resolved.SystemModels.HallucinationDetector != defaults.SystemModels.HallucinationDetector &&
		resolved.HallucinationMitigation.HallucinationModel.ModelID == defaults.HallucinationMitigation.HallucinationModel.ModelID {
		resolved.HallucinationMitigation.HallucinationModel.ModelID = resolved.SystemModels.HallucinationDetector
	}
	if resolved.SystemModels.HallucinationExplainer != defaults.SystemModels.HallucinationExplainer &&
		resolved.HallucinationMitigation.NLIModel.ModelID == defaults.HallucinationMitigation.NLIModel.ModelID {
		resolved.HallucinationMitigation.NLIModel.ModelID = resolved.SystemModels.HallucinationExplainer
	}
	if resolved.SystemModels.FeedbackDetector != defaults.SystemModels.FeedbackDetector &&
		resolved.FeedbackDetector.ModelID == defaults.FeedbackDetector.ModelID {
		resolved.FeedbackDetector.ModelID = resolved.SystemModels.FeedbackDetector
	}
}

func canonicalBoolPtr(value bool) *bool {
	return &value
}
