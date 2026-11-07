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
		ConfigSource:      ConfigSourceFile,
		ListBackendModels: false,
		ClearRouteCache:   true,
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
			CaptureRequestBody:  canonicalBoolPtr(true),
			CaptureResponseBody: canonicalBoolPtr(true),
			CapturePersonalData: canonicalBoolPtr(true),
			CaptureToolResults:  canonicalBoolPtr(false),
			MaxRecords:          canonicalIntPtr(defaultRouterReplayMaxRecords),
			MaxBodyBytes:        canonicalIntPtr(defaultRouterReplayMaxBodyBytes),
			MaxToolTraceBytes:   canonicalIntPtr(0),
			MaxToolTraceSteps:   canonicalIntPtr(defaultRouterReplayMaxToolTraceSteps),
			Enabled:             false,
			StoreBackend:        "memory",
			TTLSeconds:          2592000,
			AsyncWrites:         false,
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
			Milvus:                     MemoryMilvusConfig{Collection: "agentic_memory"},
			DefaultRetrievalLimit:      5,
			DefaultSimilarityThreshold: DefaultMemorySimilarityThreshold,
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

// The deployments every configuration declares. Declaring one does not start
// it: a deployment runs only when a consumer resolves to it.
const (
	HazardDeployment              = "hazard"
	EmbeddingDefaultDeployment    = "vela-embedding"
	MultimodalEmbeddingDeployment = "omni-nano"
)

func defaultCanonicalModelCatalog() CanonicalModelCatalog {
	catalog := CanonicalModelCatalog{
		Embeddings: defaultCanonicalEmbeddingModels(),
		System:     CanonicalSystemModels{DecisionModel: DecisionModelBinding{Deployment: DefaultDecisionDeployment}},
		KBs:        defaultCanonicalKnowledgeBases(),
		Modules:    defaultCanonicalModelModules(),
	}
	catalog.Deployments = DefaultModelDeployments()
	enabledSoftMatching := false
	catalog.Embeddings.Semantic.EmbeddingConfig.EnableSoftMatching = &enabledSoftMatching
	return catalog
}

// DefaultModelDeployments are the pre-declared deployments: primary, the
// default decision model (vela3-fast); hazard, Vela 1.0 Hazard, which a
// safety rule's hazard part names to score categories with its operating
// point; vela-embedding and omni-nano, the default text and multimodal
// embedding models.
func DefaultModelDeployments() map[string]ModelDeployment {
	deployments := make(map[string]ModelDeployment)
	if primary, err := ImplicitModelRuntimeDeployment(Vela3SignalModel, true); err == nil {
		deployments[DefaultDecisionDeployment] = primary
	}
	if hazard, err := ImplicitModelRuntimeDeployment(vela1HazardModel, true); err == nil {
		hazard.Input = ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}
		deployments[HazardDeployment] = hazard
	}
	embeddings := defaultCanonicalEmbeddingModels().Semantic
	for name, model := range map[string]string{EmbeddingDefaultDeployment: embeddings.MmBertModelPath, MultimodalEmbeddingDeployment: embeddings.MultiModalModelPath} {
		if deployment, err := ImplicitModelRuntimeDeployment(model, true); err == nil {
			deployment.Input.Overflow = "truncate"
			deployments[name] = deployment
		}
	}
	return deployments
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
		Safety:                  SafetyModelsConfig{Safety: SequenceHeadModelConfig{UseCPU: true}},
		PromptGuard:             defaultPromptGuardModule(),
		Classifier:              defaultClassifierModule(),
		Complexity:              ComplexityModelConfig{}.WithDefaults(),
		HallucinationMitigation: defaultHallucinationModule(),
		FeedbackDetector:        defaultFeedbackDetectorModule(),
	}
}

func defaultPromptGuardModule() CanonicalPromptGuardModule {
	return CanonicalPromptGuardModule{
		PromptGuardConfig: PromptGuardConfig{
			Enabled:   true,
			Threshold: ModuleThresholdsOf(Vela3SignalModel).PromptGuard,
			UseCPU:    true,
		},
	}
}

func defaultClassifierModule() CanonicalClassifierModule {
	return CanonicalClassifierModule{
		Domain: CanonicalCategoryModule{
			CategoryModel: CategoryModel{
				Threshold: ModuleThresholdsOf(Vela3SignalModel).Domain,
				UseCPU:    true,
			},
		},
		PII: CanonicalPIIModule{
			PIIModel: PIIModel{
				Threshold: ModuleThresholdsOf(Vela3SignalModel).PII,
				UseCPU:    true,
			},
		},
		Preference: PreferenceModelConfig{},
	}
}

func defaultHallucinationModule() CanonicalHallucinationModule {
	return CanonicalHallucinationModule{
		Enabled: false,
		FactCheck: CanonicalFactCheckModule{
			FactCheckModelConfig: FactCheckModelConfig{
				Threshold: ModuleThresholdsOf(Vela3SignalModel).FactCheck,
				UseCPU:    true,
			},
		},
		Detector: CanonicalHallucinationDetector{
			HallucinationModelConfig: HallucinationModelConfig{
				Threshold:         0.5,
				UseCPU:            true,
				MinSpanLength:     1,
				MinSpanConfidence: 0,
				ContextWindowSize: 50,
			},
		},
	}
}

func defaultFeedbackDetectorModule() CanonicalFeedbackDetectorModule {
	return CanonicalFeedbackDetectorModule{
		FeedbackDetectorConfig: FeedbackDetectorConfig{
			Enabled:   true,
			Threshold: ModuleThresholdsOf(Vela3SignalModel).Feedback,
			UseCPU:    true,
		},
	}
}

// DefaultSystemModels returns the models the built-in modules run when no
// binding names one: Vela 3.0 vela3-fast, the default decision model's, for
// every built-in signal it answers, and Vela 1.0 Hazard, the pre-declared
// hazard deployment's.
func DefaultSystemModels() CanonicalSystemModels {
	return CanonicalSystemModels{
		DecisionModel:         DecisionModelBinding{Deployment: DefaultDecisionDeployment},
		Safety:                Vela3SignalModel,
		Hazard:                "models/Vela-1.0-Encoder-307M-Hazard",
		PromptGuard:           Vela3SignalModel,
		DomainClassifier:      Vela3SignalModel,
		PIIClassifier:         Vela3SignalModel,
		FactCheckClassifier:   Vela3SignalModel,
		HallucinationDetector: Vela3SignalModel,
		FeedbackDetector:      Vela3SignalModel,
	}
}

// Vela1SystemModels returns the Vela 1.0 specialists the built-in signals ran
// before Vela 2.0 0.3B; global.model_catalog.system restores them.
func Vela1SystemModels() CanonicalSystemModels {
	return CanonicalSystemModels{
		Safety:                "models/Vela-1.0-Encoder-307M-Safety",
		Hazard:                "models/Vela-1.0-Encoder-307M-Hazard",
		PromptGuard:           "models/Vela-1.0-Encoder-307M-Guard",
		DomainClassifier:      "models/Vela-1.0-Encoder-307M-Domain",
		PIIClassifier:         "models/Vela-1.0-Encoder-307M-PII",
		FactCheckClassifier:   "models/Vela-1.0-Encoder-307M-FactCheck",
		HallucinationDetector: "models/Vela-1.0-Encoder-307M-Halu",
		FeedbackDetector:      "models/Vela-1.0-Encoder-307M-Feedback",
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

func canonicalBoolPtr(value bool) *bool {
	return &value
}

func canonicalIntPtr(value int) *int {
	return &value
}
