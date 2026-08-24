package config

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// CanonicalConfigFromRouterConfig exports the canonical v0.4 config surface
// from the internal runtime config.
func CanonicalConfigFromRouterConfig(cfg *RouterConfig) CanonicalConfig {
	if cfg == nil {
		return CanonicalConfig{Version: "v0.4"}
	}
	var models []AuthoringModel
	var recipes []AuthoringRecipe
	var entrypoints []AuthoringEntrypoint
	if cfg.ControlPlane.Mode != ControlPlaneModeManaged {
		models = routingModelsFromRouterConfig(cfg)
		recipes = canonicalRecipesFromRouterConfig(cfg)
		entrypoints = canonicalEntrypointsFromRouterConfig(cfg)
	}

	return CanonicalConfig{
		Version:     "v0.4",
		Listeners:   append([]Listener(nil), cfg.Listeners...),
		Models:      models,
		Entrypoints: entrypoints,
		Recipes:     recipes,
		Global:      CanonicalGlobalFromRouterConfig(cfg),
	}
}

// CanonicalRoutingFromRouterConfig exports the routing-owned canonical surface
// from the internal runtime config. Deployment bindings and router-global
// runtime settings intentionally stay outside this view.
func CanonicalRoutingFromRouterConfig(cfg *RouterConfig) CanonicalRouting {
	if cfg == nil {
		return CanonicalRouting{}
	}

	return CanonicalRouting{
		Signals:     canonicalSignalsFromSignals(cfg.Signals),
		Projections: canonicalProjectionsFromProjections(cfg.Projections),
		Decisions:   copyDecisions(cfg.Decisions),
		Strategy:    cfg.Strategy,
	}
}

func canonicalSignalsFromSignals(signals Signals) CanonicalSignals {
	return CanonicalSignals{
		Keywords:      append([]KeywordRule(nil), signals.KeywordRules...),
		Embeddings:    append([]EmbeddingRule(nil), signals.EmbeddingRules...),
		Domains:       append([]Category(nil), signals.Categories...),
		FactCheck:     append([]FactCheckRule(nil), signals.FactCheckRules...),
		UserFeedbacks: append([]UserFeedbackRule(nil), signals.UserFeedbackRules...),
		Reasks:        append([]ReaskRule(nil), signals.ReaskRules...),
		Preferences:   append([]PreferenceRule(nil), signals.PreferenceRules...),
		Language:      append([]LanguageRule(nil), signals.LanguageRules...),
		Context:       append([]ContextRule(nil), signals.ContextRules...),
		Structure:     append([]StructureRule(nil), signals.StructureRules...),
		Complexity:    append([]ComplexityRule(nil), signals.ComplexityRules...),
		Modality:      append([]ModalityRule(nil), signals.ModalityRules...),
		RoleBindings:  append([]RoleBinding(nil), signals.RoleBindings...),
		Jailbreak:     append([]JailbreakRule(nil), signals.JailbreakRules...),
		PII:           append([]PIIRule(nil), signals.PIIRules...),
		KB:            append([]KBSignalRule(nil), signals.KBRules...),
		Conversation:  append([]ConversationRule(nil), signals.ConversationRules...),
		EventRules:    append([]EventRule(nil), signals.EventRules...),
		Metadata:      append([]MetadataRule(nil), signals.MetadataRules...),
		Classifiers:   append([]ClassifierSignalRule(nil), signals.ClassifierRules...),
	}
}

func canonicalProjectionsFromProjections(projections Projections) CanonicalProjections {
	return CanonicalProjections{
		Partitions: append([]ProjectionPartition(nil), projections.Partitions...),
		Scores:     append([]ProjectionScore(nil), projections.Scores...),
		Mappings:   append([]ProjectionMapping(nil), projections.Mappings...),
	}
}

func routingModelsFromRouterConfig(cfg *RouterConfig) []AuthoringModel {
	// Managed bootstrap never owns inline routing source. Its immutable
	// snapshot may contain credential UIDs and other publication state that
	// must not be projected back into human authoring fields.
	if cfg == nil || cfg.ControlPlane.Mode == ControlPlaneModeManaged || cfg.RoutingSnapshot == nil {
		return nil
	}
	return authoringModelsFromSnapshot(cfg.RoutingSnapshot.Models)
}

func authoringModelsFromSnapshot(models []routingsnapshot.Model) []AuthoringModel {
	result := make([]AuthoringModel, 0, len(models))
	for _, model := range models {
		connections := make([]modelauthoring.Connection, 0, len(model.Backends))
		for _, backend := range model.Backends {
			weight := backend.Weight
			if weight == "1" {
				weight = ""
			}
			connections = append(connections, modelauthoring.Connection{
				Provider: backend.ProviderID, Endpoint: backend.Origin,
				Model: backend.ProviderModelID, Credential: backend.ProviderCredentialID,
				Weight: weight,
			})
		}
		execution := ModelExecutionSettings{
			MaxRetries: model.Execution.MaxRetries,
		}
		if model.Execution.RequestTimeout != defaultModelInvocationTimeout {
			execution.RequestTimeout = model.Execution.RequestTimeout
		}
		if model.Execution.StreamTimeout != defaultModelInvocationTimeout {
			execution.StreamTimeout = model.Execution.StreamTimeout
		}
		pricing := ModelRuntimePricing{
			InputCostPerMillionTokens:  cloneStringPointer(model.Pricing.InputCostPerMillionTokens),
			OutputCostPerMillionTokens: cloneStringPointer(model.Pricing.OutputCostPerMillionTokens),
		}
		if !sameModelPrice(model.Pricing.CacheReadCostPerMillionTokens, model.Pricing.InputCostPerMillionTokens) {
			pricing.CacheReadCostPerMillionTokens = cloneStringPointer(model.Pricing.CacheReadCostPerMillionTokens)
		}
		if !sameModelPrice(model.Pricing.CacheWriteCostPerMillionTokens, model.Pricing.InputCostPerMillionTokens) {
			pricing.CacheWriteCostPerMillionTokens = cloneStringPointer(model.Pricing.CacheWriteCostPerMillionTokens)
		}
		result = append(result, AuthoringModel{
			Name: model.Name,
			Card: AuthoringModelCard{
				Aliases: append([]string(nil), model.Aliases...), ParamSize: model.ParamSize,
				ContextWindowSize: model.ContextWindowSize, Description: model.Description,
				Capabilities: append([]string(nil), model.Capabilities...), Reasoning: model.Reasoning,
				LoRAs: append([]string(nil), model.LoRAs...), QualityScore: model.QualityScore,
				Modality: model.Modality, Tags: append([]string(nil), model.Tags...),
			},
			Connections:    connections,
			Execution:      execution,
			RuntimePricing: pricing,
		})
	}
	return result
}

func sameModelPrice(left, right *string) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}

// CanonicalGlobalFromRouterConfig exports the router-wide canonical global
// block from the internal runtime config.
func CanonicalGlobalFromRouterConfig(cfg *RouterConfig) *CanonicalGlobal {
	if cfg == nil {
		return nil
	}

	currency := cfg.BillingCurrency
	if cfg.ControlPlane.Mode == ControlPlaneModeManaged {
		currency = ""
	}
	var billing *CanonicalBillingGlobal
	if currency != "" {
		billing = &CanonicalBillingGlobal{Currency: currency}
	}
	global := &CanonicalGlobal{
		ControlPlane: cloneControlPlaneConfig(cfg.ControlPlane),
		Billing:      billing,
		Router: CanonicalRouterGlobal{
			ClearRouteCache: cfg.ClearRouteCache,
			StreamedBody: CanonicalStreamedBody{
				Enabled:    cfg.StreamedBodyMode,
				MaxBytes:   cfg.MaxStreamedBodyBytes,
				TimeoutSec: cfg.StreamedBodyTimeoutSec,
			},
			Learning: cfg.RouterLearning,
		},
		Services: CanonicalServiceGlobal{
			API:                cfg.API,
			ResponseAPI:        cfg.ResponseAPI,
			Agent:              cfg.Agent,
			Observability:      cfg.Observability,
			ManagementAPI:      cfg.ManagementAPI,
			Access:             cfg.Access,
			BackendCredentials: cloneBackendCredentialsConfig(cfg.BackendCredentials),
			BackendEgress:      cfg.BackendEgress,
			BackendDispatch:    cfg.BackendDispatch,
			RouterReplay:       cfg.RouterReplay,
			StartupStatus:      cfg.StartupStatus,
		},
		Stores: CanonicalStoreGlobal{
			ResponseCache: cfg.SemanticCache,
			Memory:        cfg.Memory,
			VectorStore:   cloneVectorStoreConfig(cfg.VectorStore),
			Access:        cloneAccessStoreConfig(cfg.AccessStore),
			AccessRuntime: cloneAccessRuntimeStoreConfig(cfg.AccessRuntimeStore),
		},
		Integrations: CanonicalIntegrationGlobal{
			Tools:  cfg.Tools,
			Looper: cfg.Looper,
		},
		ModelCatalog: canonicalModelCatalogFromRouterConfig(cfg),
	}

	return global
}

func canonicalModelCatalogFromRouterConfig(cfg *RouterConfig) CanonicalModelCatalog {
	return CanonicalModelCatalog{
		Embeddings: CanonicalEmbeddingModels{
			Semantic: cfg.EmbeddingModels,
		},
		System: CanonicalSystemModels{
			PromptGuard:            cfg.PromptGuard.ModelID,
			DomainClassifier:       cfg.CategoryModel.ModelID,
			PIIClassifier:          cfg.PIIModel.ModelID,
			FactCheckClassifier:    cfg.HallucinationMitigation.FactCheckModel.ModelID,
			HallucinationDetector:  cfg.HallucinationMitigation.HallucinationModel.ModelID,
			HallucinationExplainer: cfg.HallucinationMitigation.NLIModel.ModelID,
			FeedbackDetector:       cfg.FeedbackDetector.ModelID,
		},
		External: append([]ExternalModelConfig(nil), cfg.ExternalModels...),
		KBs:      append([]KnowledgeBaseConfig(nil), cfg.KnowledgeBases...),
		Modules: CanonicalModelModules{
			PromptCompression: cfg.PromptCompression,
			PromptGuard: CanonicalPromptGuardModule{
				PromptGuardConfig: cfg.PromptGuard,
				ModelRef:          "prompt_guard",
			},
			Classifier: CanonicalClassifierModule{
				Domain: CanonicalCategoryModule{
					CategoryModel: cfg.CategoryModel,
					ModelRef:      "domain_classifier",
				},
				MCP: cfg.MCPCategoryModel,
				PII: CanonicalPIIModule{
					PIIModel: cfg.PIIModel,
					ModelRef: "pii_classifier",
				},
				Preference: cfg.PreferenceModel.WithDefaults(),
			},
			Complexity: cfg.ComplexityModel.WithDefaults(),
			HallucinationMitigation: CanonicalHallucinationModule{
				Enabled:                 cfg.HallucinationMitigation.Enabled,
				OnHallucinationDetected: cfg.HallucinationMitigation.OnHallucinationDetected,
				FactCheck: CanonicalFactCheckModule{
					FactCheckModelConfig: cfg.HallucinationMitigation.FactCheckModel,
					ModelRef:             "fact_check_classifier",
				},
				Detector: CanonicalHallucinationDetector{
					HallucinationModelConfig: cfg.HallucinationMitigation.HallucinationModel,
					ModelRef:                 "hallucination_detector",
				},
				Explainer: CanonicalExplainerModule{
					NLIModelConfig: cfg.HallucinationMitigation.NLIModel,
					ModelRef:       "hallucination_explainer",
				},
			},
			FeedbackDetector: CanonicalFeedbackDetectorModule{
				FeedbackDetectorConfig: cfg.FeedbackDetector,
				ModelRef:               "feedback_detector",
			},
			ModalityDetector: cfg.ModalityDetector,
		},
	}
}

func cloneVectorStoreConfig(cfg *VectorStoreConfig) *VectorStoreConfig {
	if cfg == nil {
		return nil
	}
	cloned := *cfg
	return &cloned
}
