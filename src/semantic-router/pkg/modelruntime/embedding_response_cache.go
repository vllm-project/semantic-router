package modelruntime

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// PrepareOwnedResponseCacheEmbeddings gives the shared cache its own typed
// consumer handle over the global serving binding. Its resource reference uses
// the same pool as recipe handles, without borrowing any recipe's override.
func PrepareOwnedResponseCacheEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime) (*embedding.Set, error) {
	if cfg == nil {
		return nil, fmt.Errorf("response cache requires model configuration")
	}
	if !cfg.NeedsSemanticResponseCache() {
		return embedding.NewSet(nil, ""), nil
	}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	spec, ok := plan.LookupGlobal("embedding")
	model := config.SemanticCacheEmbeddingModel(cfg)
	primary := strings.ToLower(strings.TrimSpace(cfg.EmbeddingConfig.ModelType))
	if primary == "" {
		primary = "qwen3"
	}
	if ok && (model != primary || (spec.Deployment.Provider != "http" && spec.Binding.Adapter != model)) {
		return nil, fmt.Errorf("response cache embedding_model %q must match the global embedding model %q and adapter %q", model, primary, spec.Binding.Adapter)
	}
	// An empty routing profile carries service settings without routing signals,
	// selectors, KBs, or recipe overrides. Prepare only the response-cache demand.
	scoped := cfg.ConfigForGlobalModelServices()
	scoped.Tools.Enabled = false
	scoped.Memory.Enabled = false
	scoped.VectorStore = nil
	scoped.ModelSelection.Enabled = false
	var view embedding.Options
	for _, requirement := range config.EmbeddingRequirements(scoped, primary, true) {
		if requirement.Consumer == "response cache" {
			view.Dimension, view.Layer = requirement.Dimension, requirement.Layer
		}
	}
	return prepareEmbeddings(ctx, scoped, runtime, true, view)
}
