package config

// DefaultCanonicalGlobal returns the router-owned runtime defaults used when
// canonical config omits all or part of the global block.
func DefaultCanonicalGlobal() CanonicalGlobal {
	defaults := CanonicalGlobal{
		Router:       defaultCanonicalRouterGlobal(),
		Services:     defaultCanonicalServiceGlobal(),
		Stores:       defaultCanonicalStoreGlobal(),
		Integrations: defaultCanonicalIntegrationGlobal(),
		ModelCatalog: defaultCanonicalModelCatalog(),
	}
	return defaults
}

func defaultCanonicalRouterGlobal() CanonicalRouterGlobal {
	return CanonicalRouterGlobal{
		ConfigSource:              ConfigSourceFile,
		AutoModelName:             "MoM",
		IncludeConfigModelsInList: false,
		ClearRouteCache:           true,
		ModelSelection: ModelSelectionConfig{
			Enabled: true,
			Method:  "knn",
			RouterDC: RouterDCSelectionConfig{
				Temperature:         0.07,
				DimensionSize:       768,
				MinSimilarity:       0.3,
				UseQueryContrastive: true,
				UseModelContrastive: true,
				UseCapabilities:     true,
			},
			Hybrid: HybridSelectionConfig{
				ExperienceWeight:    0.3,
				RouterDCWeight:      0.3,
				AutoMixWeight:       0.2,
				CostWeight:          0.2,
				QualityGapThreshold: 0.1,
				NormalizeScores:     true,
			},
		},
	}
}

func defaultCanonicalServiceGlobal() CanonicalServiceGlobal {
	return CanonicalServiceGlobal{
		API: APIConfig{RoutingPreview: RoutingPreviewConfig{
			RequestTimeoutSeconds: canonicalIntPtr(DefaultRoutingPreviewTimeoutSeconds),
			MaxConcurrency:        canonicalIntPtr(DefaultRoutingPreviewMaxConcurrency),
		}},
		ResponseAPI: ResponseAPIConfig{
			Enabled:      true,
			StoreBackend: "redis",
			TTLSeconds:   86400,
			MaxResponses: 1000,
		},
		RouterReplay: RouterReplayConfig{
			Enabled:      false,
			StoreBackend: "memory",
			TTLSeconds:   2592000,
			AsyncWrites:  false,
		},
		StartupStatus: StartupStatusConfig{
			StoreBackend: "file",
		},
		ManagementAPI: DefaultManagementAPIConfig(),
		Observability: ObservabilityConfig{
			Metrics: MetricsConfig{
				Enabled: canonicalBoolPtr(true),
			},
			Profiling: ProfilingConfig{
				Enabled: false,
				Port:    DefaultProfilingPort,
				Bind:    DefaultProfilingBind,
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
	}
}

func defaultCanonicalStoreGlobal() CanonicalStoreGlobal {
	return CanonicalStoreGlobal{
		Memory: MemoryConfig{
			Enabled:                    false,
			AutoStore:                  false,
			Milvus:                     MemoryMilvusConfig{Collection: "agentic_memory", Dimension: 384},
			DefaultRetrievalLimit:      5,
			DefaultSimilarityThreshold: 0.70,
			Persistence: MemoryPersistenceConfig{
				TimeoutSeconds:       30,
				Concurrency:          8,
				Queue:                64,
				ShutdownGraceSeconds: 5,
			},
		},
		ResponseCache: SemanticCache{
			Enabled:        true,
			BackendType:    "memory",
			MaxEntries:     1000,
			TTLSeconds:     3600,
			EvictionPolicy: "fifo",
		},
	}
}

func defaultCanonicalIntegrationGlobal() CanonicalIntegrationGlobal {
	return CanonicalIntegrationGlobal{
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
			Flow: FlowRuntimeConfig{
				State: WorkflowStateRuntimeConfig{
					StoreBackend: WorkflowStateBackendFile,
					TTLSeconds:   DefaultWorkflowStateTTLSeconds,
				},
			},
		},
	}
}

func defaultCanonicalModelCatalog() CanonicalModelCatalog {
	catalog := CanonicalModelCatalog{
		Embeddings: defaultCanonicalEmbeddingModels(),
		System:     DefaultSystemModels(),
		KBs:        defaultCanonicalKnowledgeBases(),
		Modules:    defaultCanonicalModelModules(),
	}
	// Declaring a deployment does not activate it. Recipes opt in through an
	// independent-score binding that names the artifact's operating point.
	hazard := ModelDeployment{
		Artifact:  catalog.System.Hazard,
		Provider:  "candle",
		Device:    "cpu",
		Precision: "fp32",
		Input:     ModelInputBudget{MaxTokens: 32768, Overflow: "reject"},
	}
	if model := GetModelByPath(hazard.Artifact); model != nil {
		hazard.Revision = model.Revision
	}
	catalog.Deployments = map[string]ModelDeployment{"hazard": hazard}
	enabledSoftMatching := false
	catalog.Embeddings.Semantic.EmbeddingConfig.EnableSoftMatching = &enabledSoftMatching
	return catalog
}

func defaultCanonicalKnowledgeBases() []KnowledgeBaseConfig {
	return []KnowledgeBaseConfig{
		defaultPrivacyKnowledgeBase(),
		defaultCalibrationKnowledgeBase(),
	}
}

func defaultPrivacyKnowledgeBase() KnowledgeBaseConfig {
	return KnowledgeBaseConfig{
		Name: "privacy_kb",
		Source: KnowledgeBaseSource{
			Path:     "knowledge_bases/privacy/",
			Manifest: "labels.json",
		},
		Threshold: 0.55,
		LabelThresholds: map[string]float32{
			"prompt_injection": 0.7,
		},
		Groups: map[string][]string{
			"security_containment": {"prompt_injection", "credential_exfiltration", "jailbreak_role", "system_prompt_extraction"},
			"privacy_policy":       {"proprietary_code", "internal_document", "pii", "business_strategy", "trade_secret_ip", "operational_infrastructure", "locality_directive", "customer_data"},
			"frontier_reasoning":   {"architecture_analysis", "root_cause_analysis", "multi_step_tradeoffs"},
			"local_standard":       {"generic_coding", "simple_task", "general_knowledge"},
			"private":              {"prompt_injection", "credential_exfiltration", "jailbreak_role", "system_prompt_extraction", "proprietary_code", "internal_document", "pii", "business_strategy", "trade_secret_ip", "operational_infrastructure", "locality_directive", "customer_data"},
			"public":               {"architecture_analysis", "root_cause_analysis", "multi_step_tradeoffs", "generic_coding", "simple_task", "general_knowledge"},
		},
		Metrics: []KnowledgeBaseMetricConfig{
			{
				Name:          "private_vs_public",
				Type:          KBMetricTypeGroupMargin,
				PositiveGroup: "private",
				NegativeGroup: "public",
			},
		},
	}
}

func defaultCalibrationKnowledgeBase() KnowledgeBaseConfig {
	return KnowledgeBaseConfig{
		Name: "mmlu_kb",
		Source: KnowledgeBaseSource{
			Path:     "knowledge_bases/mmlu/",
			Manifest: "labels.json",
		},
		Threshold: 0.25,
		Groups: map[string][]string{
			"escalate": {"biology", "business", "computer science", "economics", "history", "math", "other", "philosophy", "psychology"},
			"keep":     {"chemistry", "engineering", "health", "law", "physics"},
		},
		Metrics: []KnowledgeBaseMetricConfig{
			{
				Name:          "escalate_vs_keep",
				Type:          KBMetricTypeGroupMargin,
				PositiveGroup: "escalate",
				NegativeGroup: "keep",
			},
		},
	}
}

func defaultCanonicalEmbeddingModels() CanonicalEmbeddingModels {
	return CanonicalEmbeddingModels{
		Semantic: EmbeddingModels{
			MmBertModelPath:     "models/Vela-1.0-Encoder-307M-Embedding",
			MultiModalModelPath: "models/vela-1.0-omni-nano",
			UseCPU:              true,
			EmbeddingConfig: HNSWConfig{
				// Keep representative routing samples; full 32K text is explicitly opt-in.
				FullContext:       false,
				ModelType:         "mmbert",
				PreloadEmbeddings: true,
				TargetDimension:   0,
				TargetLayer:       22,
				TopK:              canonicalIntPtr(0),
				MinScoreThreshold: 0.5,
			},
		},
	}
}

func defaultCanonicalModelModules() CanonicalModelModules {
	return CanonicalModelModules{
		Safety:                  SafetyModelsConfig{Safety: SequenceHeadModelConfig{ModelRef: "safety", UseCPU: true}},
		PromptGuard:             defaultPromptGuardModule(),
		Classifier:              defaultClassifierModule(),
		Complexity:              ComplexityModelConfig{}.WithDefaults(),
		HallucinationMitigation: defaultHallucinationModule(),
		FeedbackDetector:        defaultFeedbackDetectorModule(),
	}
}

func defaultPromptGuardModule() CanonicalPromptGuardModule {
	return CanonicalPromptGuardModule{
		ModelRef: "prompt_guard",
		PromptGuardConfig: PromptGuardConfig{
			Enabled:              true,
			Threshold:            0.5,
			UseCPU:               true,
			Variant:              PromptGuardVariantMmBERT32K,
			JailbreakMappingPath: "models/Vela-1.0-Encoder-307M-Guard/jailbreak_type_mapping.json",
		},
	}
}

func defaultClassifierModule() CanonicalClassifierModule {
	return CanonicalClassifierModule{
		Domain: CanonicalCategoryModule{
			ModelRef: "domain_classifier",
			CategoryModel: CategoryModel{
				Threshold:           0.5,
				UseCPU:              true,
				Variant:             CategoryVariantMmBERT32K,
				CategoryMappingPath: "models/Vela-1.0-Encoder-307M-Domain/category_mapping.json",
			},
		},
		PII: CanonicalPIIModule{
			ModelRef: "pii_classifier",
			PIIModel: PIIModel{
				Threshold:      0.9,
				UseCPU:         true,
				UseMmBERT32K:   true,
				PIIMappingPath: "models/Vela-1.0-Encoder-307M-PII/pii_mapping.json",
			},
		},
		Preference: PreferenceModelConfig{
			UseContrastive: canonicalBoolPtr(true),
		},
	}
}

func defaultHallucinationModule() CanonicalHallucinationModule {
	return CanonicalHallucinationModule{
		Enabled: false,
		FactCheck: CanonicalFactCheckModule{
			ModelRef: "fact_check_classifier",
			FactCheckModelConfig: FactCheckModelConfig{
				Threshold:    0.95,
				UseCPU:       true,
				UseMmBERT32K: true,
			},
		},
		Detector: CanonicalHallucinationDetector{
			ModelRef: "hallucination_detector",
			HallucinationModelConfig: HallucinationModelConfig{
				Threshold:              0.5,
				UseCPU:                 true,
				MinSpanLength:          1,
				MinSpanConfidence:      0,
				ContextWindowSize:      50,
				EnableNLIFiltering:     false,
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
}

func defaultFeedbackDetectorModule() CanonicalFeedbackDetectorModule {
	return CanonicalFeedbackDetectorModule{
		ModelRef: "feedback_detector",
		FeedbackDetectorConfig: FeedbackDetectorConfig{
			Enabled:      true,
			Threshold:    0.7,
			UseCPU:       true,
			UseMmBERT32K: true,
		},
	}
}

// DefaultSystemModels returns stable capability bindings for built-in runtime models.
func DefaultSystemModels() CanonicalSystemModels {
	return CanonicalSystemModels{
		Safety:                 "models/Vela-1.0-Encoder-307M-Safety",
		Hazard:                 "models/Vela-1.0-Encoder-307M-Hazard",
		PromptGuard:            "models/Vela-1.0-Encoder-307M-Guard",
		DomainClassifier:       "models/Vela-1.0-Encoder-307M-Domain",
		PIIClassifier:          "models/Vela-1.0-Encoder-307M-PII",
		FactCheckClassifier:    "models/Vela-1.0-Encoder-307M-FactCheck",
		HallucinationDetector:  "models/Vela-1.0-Encoder-307M-Halu",
		HallucinationExplainer: "models/mom-halugate-explainer",
		FeedbackDetector:       "models/Vela-1.0-Encoder-307M-Feedback",
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

func canonicalIntPtr(value int) *int {
	return &value
}
