package config

import (
	"fmt"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func validateNativeBackendContracts(cfg *RouterConfig) error {
	// Add config validation logic here checking native.Registry
	// For instance, check if the embedding backend is supported
	backend := native.Backend(cfg.EmbeddingModels.EmbeddingBackend())
	if backend != "" && backend != "openai_compatible" {
		if _, ok := native.Registry.Get(backend); !ok {
			return fmt.Errorf("configured embedding backend %q is not registered in native runtime", backend)
		}
	}
	return nil
}
