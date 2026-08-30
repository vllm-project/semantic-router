package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

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

func (r *OpenAIRouter) currentFileStore() *vectorstore.FileStore {
	if r == nil || r.RuntimeRegistry == nil {
		return nil
	}
	runtime := r.RuntimeRegistry.VectorStoreRuntime()
	if runtime == nil {
		return nil
	}
	return runtime.FileStore
}

// routerImageFileResolver resolves Response API input_image file_id references
// against the router's own uploaded-file store (POST /v1/files with
// purpose=vision). It consults the registry on every call because the
// vector-store runtime is attached after the router components are built.
type routerImageFileResolver struct {
	router *OpenAIRouter
}

func (res routerImageFileResolver) ResolveImageFile(fileID string) ([]byte, error) {
	fileStore := res.router.currentFileStore()
	if fileStore == nil {
		return nil, fmt.Errorf("the router file store is not enabled (vector_store.enabled); send the image as image_url or file_data")
	}
	return fileStore.Read(fileID)
}
