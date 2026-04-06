package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func (r *OpenAIRouter) currentConfig() *config.RouterConfig {
	if r != nil && r.RuntimeRegistry != nil {
		if cfg := r.RuntimeRegistry.CurrentConfig(); cfg != nil {
			return cfg
		}
	}
	if r != nil && r.Config != nil {
		return r.Config
	}
	return config.Get()
}

func (r *OpenAIRouter) currentVectorStoreManager() *vectorstore.Manager {
	if r == nil || r.RuntimeRegistry == nil {
		return nil
	}
	runtime := r.RuntimeRegistry.VectorStoreRuntime()
	if runtime == nil {
		return nil
	}
	return runtime.Manager
}

func (r *OpenAIRouter) currentVectorStoreEmbedder() vectorstore.Embedder {
	if r == nil || r.RuntimeRegistry == nil {
		return nil
	}
	runtime := r.RuntimeRegistry.VectorStoreRuntime()
	if runtime == nil {
		return nil
	}
	return runtime.Embedder
}
