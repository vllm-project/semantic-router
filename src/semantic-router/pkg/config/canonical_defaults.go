package config

// DefaultCanonicalGlobal returns the router-owned runtime defaults used when
// canonical config omits all or part of the global block.
func DefaultCanonicalGlobal() CanonicalGlobal {
	defaults := CanonicalGlobal{
		Router: CanonicalRouterGlobal{
			AutoModelName:             "MoM",
			IncludeConfigModelsInList: false,
			ClearRouteCache:           true,
			ModelSelection: ModelSelectionConfig{
				Enabled: true,
				Method:  "knn",
			},
		},
		Services: CanonicalServiceGlobal{
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
		},
		Stores: CanonicalStoreGlobal{
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
		},
		Integrations: CanonicalIntegrationGlobal{
			Tools: ToolsConfig{
				Enabled:         false,
				TopK:            3,
				ToolsDBPath:     "config/tools_db.json",
				FallbackToEmpty: true,
			},
			Looper: LooperConfig{
				Endpoint:       "http://localhost:8899/v1/chat/completions",
				TimeoutSeconds: 1200,
				Headers:        map[string]string{},
			},
		},
		ModelCatalog: CanonicalModelCatalog{
			Embeddings: CanonicalEmbeddingModels{
				Semantic: EmbeddingModels{
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
			System: DefaultSystemModels(),
		},
	}

	defaults.ModelCatalog.Modules.PromptGuard = CanonicalPromptGuardModule{
		ModelRef: "prompt_guard",
		PromptGuardConfig: PromptGuardConfig{
			Enabled:              true,
			Threshold:            0.7,
			UseCPU:               true,
			UseMmBERT32K:         true,
			JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
		},
	}
	defaults.ModelCatalog.Modules.Classifier = CanonicalClassifierModule{
		Domain: CanonicalCategoryModule{
			ModelRef: "domain_classifier",
			CategoryModel: CategoryModel{
				Threshold:           0.5,
				UseCPU:              true,
				UseMmBERT32K:        true,
				CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
			},
		},
		PII: CanonicalPIIModule{
			ModelRef: "pii_classifier",
			PIIModel: PIIModel{
				Threshold:      0.9,
				UseCPU:         true,
				UseMmBERT32K:   true,
				PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
			},
		},
	}
	defaults.ModelCatalog.Modules.HallucinationMitigation = CanonicalHallucinationModule{
		Enabled: false,
		FactCheck: CanonicalFactCheckModule{
			ModelRef: "fact_check_classifier",
			FactCheckModelConfig: FactCheckModelConfig{
				Threshold:    0.6,
				UseCPU:       true,
				UseMmBERT32K: true,
			},
		},
		Detector: CanonicalHallucinationDetector{
			ModelRef: "hallucination_detector",
			HallucinationModelConfig: HallucinationModelConfig{
				Threshold:              0.8,
				UseCPU:                 true,
				MinSpanLength:          2,
				MinSpanConfidence:      0.6,
				ContextWindowSize:      50,
				EnableNLIFiltering:     true,
				NLIEntailmentThreshold: 0.75,
			},
		},
		Explainer: CanonicalExplainerModule{
			ModelRef: "hallucination_explainer",
			NLIModelConfig: NLIModelConfig{
				Threshold: 0.9,
				UseCPU:    true,
			},
		},
	}
	defaults.ModelCatalog.Modules.FeedbackDetector = CanonicalFeedbackDetectorModule{
		ModelRef: "feedback_detector",
		FeedbackDetectorConfig: FeedbackDetectorConfig{
			Enabled:      true,
			Threshold:    0.7,
			UseCPU:       true,
			UseMmBERT32K: true,
		},
	}

	enabledSoftMatching := true
	defaults.ModelCatalog.Embeddings.Semantic.HNSWConfig.EnableSoftMatching = &enabledSoftMatching

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
	_ = resolveModuleModelRefs(&global)
	cfg := RouterConfig{}
	_ = applyCanonicalGlobal(&cfg, &global)
	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}
	return cfg
}

func canonicalBoolPtr(value bool) *bool {
	return &value
}
