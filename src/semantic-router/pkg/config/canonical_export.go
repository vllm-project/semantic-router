package config

import (
	"fmt"

	"gopkg.in/yaml.v2"
)

// CanonicalConfigFromRouterConfig exports the public v0.3 file-authoring
// source. Runtime snapshots are deliberately not reverse-engineered into
// authoring YAML because they no longer contain source-only names and secret
// references.
func CanonicalConfigFromRouterConfig(cfg *RouterConfig) CanonicalConfig {
	if cfg == nil {
		return CanonicalConfig{Version: "v0.3"}
	}
	if cfg.fileAuthoring != nil {
		cloned, err := cloneCanonicalConfig(cfg.fileAuthoring)
		if err == nil && cloned != nil {
			return *cloned
		}
	}

	return CanonicalConfig{
		Version: "v0.3", Listeners: append([]Listener(nil), cfg.Listeners...),
		Global: CanonicalGlobalFromRouterConfig(cfg),
	}
}

func cloneCanonicalConfig(source *CanonicalConfig) (*CanonicalConfig, error) {
	if source == nil {
		return nil, nil
	}
	payload, err := yaml.Marshal(source)
	if err != nil {
		return nil, fmt.Errorf("encode public v0.3 authoring source: %w", err)
	}
	var cloned CanonicalConfig
	if err := yaml.UnmarshalStrict(payload, &cloned); err != nil {
		return nil, fmt.Errorf("clone public v0.3 authoring source: %w", err)
	}
	return &cloned, nil
}

// CanonicalRoutingFromRouterConfig exports the routing-owned canonical surface
// from the internal runtime config. Deployment bindings and router-global
// runtime settings intentionally stay outside this view.
func CanonicalRoutingFromRouterConfig(cfg *RouterConfig) CanonicalRouting {
	if cfg == nil {
		return CanonicalRouting{}
	}

	return CanonicalRouting{
		ModelCards:  routingModelCardsFromRouterConfig(cfg),
		Signals:     canonicalSignalsFromSignals(cfg.Signals),
		Projections: canonicalProjectionsFromProjections(cfg.Projections),
		Decisions:   copyDecisions(cfg.Decisions),
		Strategy:    cfg.Strategy,
	}
}

func routingModelCardsFromRouterConfig(cfg *RouterConfig) []RoutingModel {
	if cfg == nil || cfg.RoutingSnapshot == nil {
		return nil
	}
	models := make([]RoutingModel, 0, len(cfg.RoutingSnapshot.Models))
	for _, model := range cfg.RoutingSnapshot.Models {
		loras := make([]LoRAAdapter, 0, len(model.LoRAs))
		for _, name := range model.LoRAs {
			loras = append(loras, LoRAAdapter{Name: name})
		}
		models = append(models, RoutingModel{
			Name: model.Name, ParamSize: model.ParamSize,
			ContextWindowSize: model.ContextWindowSize, Description: model.Description,
			Capabilities: append([]string(nil), model.Capabilities...), LoRAs: loras,
			Reasoning: ModelReasoning{
				Type: model.Reasoning.Type, Efforts: append([]string(nil), model.Reasoning.Efforts...),
			},
			QualityScore: model.QualityScore, Modality: model.Modality,
			Tags: append([]string(nil), model.Tags...),
		})
	}
	return models
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

// CanonicalGlobalFromRouterConfig exports the router-wide canonical global
// block from the internal runtime config.
func CanonicalGlobalFromRouterConfig(cfg *RouterConfig) *CanonicalGlobal {
	if cfg == nil {
		return nil
	}

	currency := cfg.BillingCurrency
	var billing *CanonicalBillingGlobal
	if currency != "" {
		billing = &CanonicalBillingGlobal{Currency: currency}
	}
	global := &CanonicalGlobal{
		Billing: billing,
		Router: CanonicalRouterGlobal{
			Strategy:                  cfg.Strategy,
			AutoModelName:             cfg.AutoModelName,
			AutoModelNames:            canonicalAutoModelNames(cfg.AutoModelNames),
			IncludeConfigModelsInList: cfg.IncludeConfigModelsInList,
			ClearRouteCache:           cfg.ClearRouteCache,
			StreamedBody: CanonicalStreamedBody{
				Enabled:    cfg.StreamedBodyMode,
				MaxBytes:   cfg.MaxStreamedBodyBytes,
				TimeoutSec: cfg.StreamedBodyTimeoutSec,
			},
			ModelSelection: cfg.ModelSelection,
			Learning:       cfg.RouterLearning,
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
			RoutingSecurity:    cfg.RoutingSecurity,
			RouterReplay:       cfg.RouterReplay,
			StartupStatus:      cfg.StartupStatus,
		},
		Stores: CanonicalStoreGlobal{
			ResponseCache: cfg.SemanticCache,
			Memory:        cfg.Memory,
			VectorStore:   cloneVectorStoreConfig(cfg.VectorStore),
			Management:    canonicalManagementStore(cfg.AccessStore),
			Runtime:       canonicalRuntimeStore(cfg.AccessRuntimeStore),
		},
		Integrations: CanonicalIntegrationGlobal{
			Tools:  cfg.Tools,
			Looper: cfg.Looper,
		},
		ModelCatalog: canonicalModelCatalogFromRouterConfig(cfg),
	}

	return global
}

func canonicalAutoModelNames(names []string) *[]string {
	if names == nil {
		return nil
	}
	cloned := append([]string(nil), names...)
	return &cloned
}

func canonicalManagementStore(source *AccessStoreConfig) *CanonicalManagementStore {
	if source == nil {
		return nil
	}
	postgres := source.Postgres
	return &CanonicalManagementStore{Postgres: &postgres}
}

func canonicalRuntimeStore(source *AccessRuntimeStoreConfig) *CanonicalRuntimeStore {
	if source == nil {
		return nil
	}
	redis := source.Redis
	return &CanonicalRuntimeStore{Redis: &redis}
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
