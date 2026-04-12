package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"

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
