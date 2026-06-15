/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import "context"

// VectorStoreBackend abstracts the underlying vector database.
type VectorStoreBackend interface {
	// CreateCollection creates a new collection for a vector store.
	CreateCollection(ctx context.Context, vectorStoreID string, dimension int) error

	// DeleteCollection deletes a collection and all its data.
	DeleteCollection(ctx context.Context, vectorStoreID string) error

	// CollectionExists checks whether a collection exists.
	CollectionExists(ctx context.Context, vectorStoreID string) (bool, error)

	// InsertChunks inserts embedded chunks into a collection.
	InsertChunks(ctx context.Context, vectorStoreID string, chunks []EmbeddedChunk) error

	// DeleteByFileID removes all chunks associated with a file.
	DeleteByFileID(ctx context.Context, vectorStoreID string, fileID string) error

	// Search performs a vector similarity search.
	Search(ctx context.Context, vectorStoreID string, queryEmbedding []float32,
		topK int, threshold float32, filter map[string]interface{}) ([]SearchResult, error)

	// Close releases all resources held by the backend.
	Close() error
}
