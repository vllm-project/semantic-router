package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

// InspectPluginBinding observes handles in this leased router generation.
// Classifier checks use the binding's exact recipe, never aggregate readiness.
func (r *OpenAIRouter) InspectPluginBinding(binding pluginruntime.Binding, pluginType string) ([]pluginruntime.Dependency, error) {
	request, err := r.pluginPreviewContext(context.Background(), binding, pluginType)
	if err != nil {
		return nil, err
	}
	payload, err := config.DecodeDecisionPlugin(*request.VSRSelectedDecision.GetPlugin(pluginType))
	if err != nil {
		return nil, err
	}
	result := []pluginruntime.Dependency{}
	add := func(name string, available bool) {
		status := "unavailable"
		if available {
			status = "available"
		}
		result = append(result, pluginruntime.Dependency{Name: name, Availability: status, Health: "not_probed"})
	}
	unknown := func(name string) {
		result = append(result, pluginruntime.Dependency{Name: name, Availability: "unknown", Health: "not_probed"})
	}
	classifier := r.classifierForRequest(request)
	switch policy := payload.(type) {
	case *config.ResponseCachePluginConfig:
		add("response_cache", r.ResponseCache != nil)
	case *config.ContextCompressionPluginConfig:
		add("context_compression", r.ContextCompression != nil)
		if policy.Recovery != nil && policy.Recovery.Enabled {
			add("context_recovery", r.CompressionRecovery != nil)
		}
	case *config.MemoryPluginConfig:
		add("memory_store", r.MemoryStore != nil)
	case *config.RouterReplayPluginConfig:
		add("replay_store", r.ReplayRecorder != nil || r.ReplayRecorders[config.RoutingDecisionKey(binding.Recipe, binding.Decision)] != nil)
	case *config.HallucinationPluginConfig:
		add("hallucination_detector", classifier != nil && classifier.IsHallucinationDetectorReady())
		useNLI := policy.UseNLI
		if rules := r.hallucinationRules(request); len(rules) > 0 {
			useNLI = hallucinationRulesUseNLI(rules)
		}
		if useNLI {
			add("nli", classifier != nil && classifier.IsHallucinationExplainerReady())
		}
	case *config.ResponseJailbreakPluginConfig:
		add("response_jailbreak_classifier", classifier != nil && classifier.IsJailbreakEnabled())
	case *config.ToolsPluginConfig:
		if policy.SelectionEnabled() {
			add("tool_retriever", r.ToolsDatabase != nil && r.ToolsDatabase.IsEnabled())
		}
	case *config.ToolSelectionPluginConfig:
		if policy.Mode == config.ToolSelectionModeFilter {
			add("tool_embeddings", r.toolEmbedder != nil)
		} else if policy.ToolsDBPath != "" {
			r.toolSelectionDBMu.Lock()
			database := r.toolSelectionDBByPath[policy.ToolsDBPath]
			r.toolSelectionDBMu.Unlock()
			if database == nil {
				unknown("tool_retriever")
			} else {
				add("tool_retriever", database.IsEnabled())
			}
		} else {
			add("tool_retriever", r.ToolsDatabase != nil && r.ToolsDatabase.IsEnabled())
		}
	case *config.RAGPluginConfig:
		if policy.Backend == "vectorstore" {
			add("vector_store", r.RuntimeRegistry != nil && r.RuntimeRegistry.VectorStoreRuntime() != nil)
		} else {
			unknown("rag_backend")
		}
	case *config.ShadowDispatchPluginConfig:
		add("shadow_dispatcher", r.ShadowDispatcher != nil)
	}
	return result, nil
}
