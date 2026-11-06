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
	EmbeddingDefaultDeployment    = "vela-embedding"
	MultimodalEmbeddingDeployment = "omni-nano"
)

func defaultCanonicalModelCatalog() CanonicalModelCatalog {
	catalog := CanonicalModelCatalog{
		Embeddings: defaultCanonicalEmbeddingModels(),
		System:     CanonicalSystemModels{DecisionModel: DecisionModelBinding{Deployment: DefaultDecisionDeployment}},
	}
	catalog.Deployments = DefaultModelDeployments()
	enabledSoftMatching := false
	catalog.Embeddings.Semantic.EmbeddingConfig.EnableSoftMatching = &enabledSoftMatching
	return catalog
}

// DefaultModelDeployments are the pre-declared deployments: primary, the
// default decision model (vela3-fast); vela-embedding and omni-nano, the
// default text and multimodal embedding models.
func DefaultModelDeployments() map[string]ModelDeployment {
	deployments := make(map[string]ModelDeployment)
	if primary, err := ImplicitModelRuntimeDeployment(Vela3SignalModel, true); err == nil {
		deployments[DefaultDecisionDeployment] = primary
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
