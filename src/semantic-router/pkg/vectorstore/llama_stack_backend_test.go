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

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Compile-time interface check.
var _ VectorStoreBackend = (*LlamaStackBackend)(nil)

var _ = Describe("LlamaStackBackend", func() {
	Context("NewLlamaStackBackend (constructor)", func() {
		It("should create backend with valid config", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint: "http://localhost:8321",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(b).NotTo(BeNil())
			Expect(b.endpoint).To(Equal("http://localhost:8321"))
			Expect(b.storeIDs).NotTo(BeNil())
		})

		It("should fail when endpoint is empty", func() {
			_, err := NewLlamaStackBackend(LlamaStackBackendConfig{})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("endpoint is required"))
		})

		It("should strip trailing slash from endpoint", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint: "http://localhost:8321/",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(b.endpoint).To(Equal("http://localhost:8321"))
		})

		It("should use default timeout of 30 seconds when not specified", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint: "http://localhost:8321",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(b.httpClient.Timeout.Seconds()).To(Equal(30.0))
		})

		It("should use custom timeout when specified", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:              "http://localhost:8321",
				RequestTimeoutSeconds: 60,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(b.httpClient.Timeout.Seconds()).To(Equal(60.0))
		})

		It("should store embedding model and dimension", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:           "http://localhost:8321",
				EmbeddingModel:     "all-MiniLM-L6-v2",
				EmbeddingDimension: 384,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(b.embeddingModel).To(Equal("all-MiniLM-L6-v2"))
			Expect(b.embeddingDim).To(Equal(384))
		})
	})

	Context("CreateCollection", func() {
		It("should POST to /v1/vector_stores and cache the generated ID", func() {
			var receivedBody map[string]interface{}
			var receivedMethod, receivedPath string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedMethod = r.Method
				receivedPath = r.URL.Path
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123", "name": "my-store"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:           server.URL,
				EmbeddingModel:     "all-MiniLM-L6-v2",
				EmbeddingDimension: 384,
			})
			Expect(err).NotTo(HaveOccurred())

			err = b.CreateCollection(context.Background(), "my-store", 0)
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedMethod).To(Equal("POST"))
			Expect(receivedPath).To(Equal("/v1/vector_stores"))
			Expect(receivedBody["name"]).To(Equal("my-store"))
			Expect(receivedBody["embedding_model"]).To(Equal("all-MiniLM-L6-v2"))
			Expect(receivedBody["embedding_dimension"]).To(BeNumerically("==", 384))

			// Verify the generated ID was cached.
			Expect(b.storeIDs["my-store"]).To(Equal("vs_gen_abc123"))
		})

		It("should prefer passed dimension over config dimension", func() {
			var receivedBody map[string]interface{}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:           server.URL,
				EmbeddingDimension: 384,
			})
			Expect(err).NotTo(HaveOccurred())

			err = b.CreateCollection(context.Background(), "my-store", 768)
			Expect(err).NotTo(HaveOccurred())
			Expect(receivedBody["embedding_dimension"]).To(BeNumerically("==", 768))
		})

		It("should not include embedding fields when not configured", func() {
			var receivedBody map[string]interface{}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.CreateCollection(context.Background(), "my-store", 0)
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedBody).NotTo(HaveKey("embedding_model"))
			Expect(receivedBody).NotTo(HaveKey("embedding_dimension"))
		})

		It("should return error on server failure", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"error": "db unavailable"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.CreateCollection(context.Background(), "my-store", 768)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to create vector store"))
		})

		It("should return error when response has empty ID", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"name": "my-store"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.CreateCollection(context.Background(), "my-store", 768)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("empty ID"))
		})
	})

	Context("resolveStoreID", func() {
		It("should return cached ID when available", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://unused"})
			Expect(err).NotTo(HaveOccurred())

			b.storeIDs["my-store"] = "vs_cached_123"

			id, err := b.resolveStoreID(context.Background(), "my-store")
			Expect(err).NotTo(HaveOccurred())
			Expect(id).To(Equal("vs_cached_123"))
		})

		It("should list stores and resolve by name on cache miss", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "GET" && r.URL.Path == "/v1/vector_stores" {
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(`{
						"data": [
							{"id": "vs_older", "name": "my-store", "created_at": 100},
							{"id": "vs_newer", "name": "my-store", "created_at": 200},
							{"id": "vs_other", "name": "other-store", "created_at": 300}
						]
					}`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			id, err := b.resolveStoreID(context.Background(), "my-store")
			Expect(err).NotTo(HaveOccurred())
			// Should pick the newest one (created_at=200).
			Expect(id).To(Equal("vs_newer"))
			// Should be cached now.
			Expect(b.storeIDs["my-store"]).To(Equal("vs_newer"))
		})

		It("should return error when store name not found in list", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			_, err = b.resolveStoreID(context.Background(), "nonexistent")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not found"))
		})
	})

	Context("DeleteCollection", func() {
		It("should resolve the store ID and send DELETE", func() {
			var receivedMethod, receivedPath string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedMethod = r.Method
				receivedPath = r.URL.Path
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			err = b.DeleteCollection(context.Background(), "my-store")
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedMethod).To(Equal("DELETE"))
			Expect(receivedPath).To(Equal("/v1/vector_stores/vs_gen_abc123"))

			// Cache should be cleared after delete.
			_, cached := b.storeIDs["my-store"]
			Expect(cached).To(BeFalse())
		})

		It("should treat missing store as no-op", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// List returns empty — store not found.
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.DeleteCollection(context.Background(), "nonexistent")
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return error on server failure during delete", func() {
			callCount := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				callCount++
				if r.Method == "DELETE" {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error": "internal"}`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			err = b.DeleteCollection(context.Background(), "my-store")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to delete vector store"))
		})
	})

	Context("CollectionExists", func() {
		It("should return true when store exists (resolved from cache)", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "GET" && r.URL.Path == "/v1/vector_stores/vs_gen_abc123" {
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			exists, err := b.CollectionExists(context.Background(), "my-store")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())
		})

		It("should return false when store not found (resolve fails)", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// List returns empty.
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			exists, err := b.CollectionExists(context.Background(), "nonexistent")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should return false when GET returns 404", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/vector_stores/vs_gen_abc123" {
					w.WriteHeader(http.StatusNotFound)
					_, _ = w.Write([]byte(`{"error": "not found"}`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			exists, err := b.CollectionExists(context.Background(), "my-store")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should return false when GET returns 400 (Llama Stack not-found variant)", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/vector_stores/vs_gen_abc123" {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = w.Write([]byte(`{"error": {"detail": "Vector_store not found"}}`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			exists, err := b.CollectionExists(context.Background(), "my-store")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should return error for other server errors", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/vector_stores/vs_gen_abc123" {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error": "db down"}`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			_, err = b.CollectionExists(context.Background(), "my-store")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to check vector store"))
		})
	})

	Context("InsertChunks", func() {
		It("should POST to /v1/vector-io/insert with correct body format", func() {
			var receivedBody map[string]interface{}
			var receivedPath string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedPath = r.URL.Path
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusNoContent)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:       server.URL,
				EmbeddingModel: "all-MiniLM-L6-v2",
			})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			chunks := []EmbeddedChunk{
				{
					ID:         "c1",
					FileID:     "file_001",
					Filename:   "doc.txt",
					Content:    "hello world",
					Embedding:  []float32{0.1, 0.2, 0.3},
					ChunkIndex: 0,
				},
			}
			err = b.InsertChunks(context.Background(), "my-store", chunks)
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedPath).To(Equal("/v1/vector-io/insert"))
			Expect(receivedBody["vector_store_id"]).To(Equal("vs_gen_abc123"))

			receivedChunks := receivedBody["chunks"].([]interface{})
			Expect(receivedChunks).To(HaveLen(1))

			chunk := receivedChunks[0].(map[string]interface{})
			Expect(chunk["content"]).To(Equal("hello world"))
			Expect(chunk["chunk_id"]).To(Equal("c1"))
			Expect(chunk["embedding_model"]).To(Equal("all-MiniLM-L6-v2"))
			Expect(chunk["embedding_dimension"]).To(BeNumerically("==", 3))

			embedding := chunk["embedding"].([]interface{})
			Expect(embedding).To(HaveLen(3))
			Expect(embedding[0]).To(BeNumerically("~", 0.1, 0.001))

			chunkMeta := chunk["chunk_metadata"].(map[string]interface{})
			Expect(chunkMeta["document_id"]).To(Equal("c1"))

			meta := chunk["metadata"].(map[string]interface{})
			Expect(meta["file_id"]).To(Equal("file_001"))
			Expect(meta["filename"]).To(Equal("doc.txt"))
			Expect(meta["chunk_index"]).To(BeNumerically("==", 0))
		})

		It("should send multiple chunks in one request", func() {
			var receivedBody map[string]interface{}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusNoContent)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			chunks := []EmbeddedChunk{
				{ID: "c1", Content: "first", Embedding: []float32{0.1}},
				{ID: "c2", Content: "second", Embedding: []float32{0.2}},
				{ID: "c3", Content: "third", Embedding: []float32{0.3}},
			}
			err = b.InsertChunks(context.Background(), "my-store", chunks)
			Expect(err).NotTo(HaveOccurred())

			receivedChunks := receivedBody["chunks"].([]interface{})
			Expect(receivedChunks).To(HaveLen(3))
		})

		It("should skip request when chunks are empty", func() {
			requestCount := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requestCount++
				w.WriteHeader(http.StatusNoContent)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.InsertChunks(context.Background(), "my-store", []EmbeddedChunk{})
			Expect(err).NotTo(HaveOccurred())
			Expect(requestCount).To(Equal(0))
		})

		It("should return error on server failure", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"error": "insert failed"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			chunks := []EmbeddedChunk{{ID: "c1", Content: "hello", Embedding: []float32{0.1}}}
			err = b.InsertChunks(context.Background(), "my-store", chunks)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to insert"))
		})
	})

	Context("DeleteByFileID", func() {
		It("should resolve store ID and send DELETE to correct path", func() {
			var receivedMethod, receivedPath string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedMethod = r.Method
				receivedPath = r.URL.Path
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			err = b.DeleteByFileID(context.Background(), "my-store", "file_001")
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedMethod).To(Equal("DELETE"))
			Expect(receivedPath).To(Equal("/v1/vector_stores/vs_gen_abc123/files/file_001"))
		})

		It("should treat 404 as success (idempotent)", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"error": "not found"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			err = b.DeleteByFileID(context.Background(), "my-store", "file_gone")
			Expect(err).NotTo(HaveOccurred())
		})

		It("should treat missing store as no-op", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.DeleteByFileID(context.Background(), "nonexistent", "file_001")
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return error for other server errors", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "DELETE" {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error": "db error"}`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			err = b.DeleteByFileID(context.Background(), "my-store", "file_001")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to delete file"))
		})
	})

	Context("Search", func() {
		It("should resolve store ID and POST to search endpoint", func() {
			var receivedBody map[string]interface{}
			var receivedPath string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedPath = r.URL.Path
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			filter := map[string]interface{}{"_query_text": "what is kubernetes?"}
			_, err = b.Search(context.Background(), "my-store", nil, 5, 0, filter)
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedPath).To(Equal("/v1/vector_stores/vs_gen_abc123/search"))
			Expect(receivedBody["query"]).To(Equal("what is kubernetes?"))
			Expect(receivedBody["max_num_results"]).To(BeNumerically("==", 5))
		})

		It("should parse results correctly", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				resp := `{
					"data": [
						{
							"content": [{"type": "text", "text": "Kubernetes is a container orchestrator"}],
							"file_id": "file_001",
							"filename": "k8s.txt",
							"score": 0.95
						},
						{
							"content": [{"type": "text", "text": "Docker runs containers"}],
							"file_id": "file_002",
							"filename": "docker.txt",
							"score": 0.82
						}
					]
				}`
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(resp))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			filter := map[string]interface{}{"_query_text": "what is kubernetes?"}
			results, err := b.Search(context.Background(), "my-store", nil, 10, 0, filter)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(2))

			Expect(results[0].Content).To(Equal("Kubernetes is a container orchestrator"))
			Expect(results[0].FileID).To(Equal("file_001"))
			Expect(results[0].Filename).To(Equal("k8s.txt"))
			Expect(results[0].Score).To(BeNumerically("~", 0.95, 0.001))

			Expect(results[1].Content).To(Equal("Docker runs containers"))
			Expect(results[1].Score).To(BeNumerically("~", 0.82, 0.001))
		})

		It("should apply threshold filtering", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				resp := `{
					"data": [
						{"content": [{"type": "text", "text": "high score"}], "score": 0.95},
						{"content": [{"type": "text", "text": "low score"}], "score": 0.30}
					]
				}`
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(resp))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			filter := map[string]interface{}{"_query_text": "test"}
			results, err := b.Search(context.Background(), "my-store", nil, 10, 0.5, filter)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
			Expect(results[0].Content).To(Equal("high score"))
		})

		It("should fail when _query_text is missing from filter", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://unused"})
			Expect(err).NotTo(HaveOccurred())

			_, err = b.Search(context.Background(), "vs_abc123", nil, 5, 0, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("_query_text"))
		})

		It("should fail when _query_text is empty string", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://unused"})
			Expect(err).NotTo(HaveOccurred())

			filter := map[string]interface{}{"_query_text": ""}
			_, err = b.Search(context.Background(), "vs_abc123", nil, 5, 0, filter)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("_query_text"))
		})

		It("should include file_id filter when provided", func() {
			var receivedBody map[string]interface{}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			filter := map[string]interface{}{
				"_query_text": "test",
				"file_id":     "file_001",
			}
			_, err = b.Search(context.Background(), "my-store", nil, 5, 0, filter)
			Expect(err).NotTo(HaveOccurred())

			filters := receivedBody["filters"].(map[string]interface{})
			Expect(filters["type"]).To(Equal("eq"))
			Expect(filters["key"]).To(Equal("file_id"))
			Expect(filters["value"]).To(Equal("file_001"))
		})

		It("should not include filters when file_id is not provided", func() {
			var receivedBody map[string]interface{}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &receivedBody)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			filter := map[string]interface{}{"_query_text": "test"}
			_, err = b.Search(context.Background(), "my-store", nil, 5, 0, filter)
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedBody).NotTo(HaveKey("filters"))
		})

		It("should handle empty data array", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": []}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			filter := map[string]interface{}{"_query_text": "no matches"}
			results, err := b.Search(context.Background(), "my-store", nil, 5, 0, filter)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})

		It("should return error on server failure", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"error": "search failed"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())
			b.storeIDs["my-store"] = "vs_gen_abc123"

			filter := map[string]interface{}{"_query_text": "test"}
			_, err = b.Search(context.Background(), "my-store", nil, 5, 0, filter)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to search"))
		})
	})

	Context("Close", func() {
		It("should not return an error", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://localhost:8321"})
			Expect(err).NotTo(HaveOccurred())

			err = b.Close()
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("doRequest (HTTP helper via public methods)", func() {
		It("should add Authorization header when auth token is set", func() {
			var receivedAuthHeader string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedAuthHeader = r.Header.Get("Authorization")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:  server.URL,
				AuthToken: "my-secret-token",
			})
			Expect(err).NotTo(HaveOccurred())

			_ = b.CreateCollection(context.Background(), "test", 768)
			Expect(receivedAuthHeader).To(Equal("Bearer my-secret-token"))
		})

		It("should not add Authorization header when auth token is empty", func() {
			var receivedAuthHeader string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedAuthHeader = r.Header.Get("Authorization")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			_ = b.CreateCollection(context.Background(), "test", 768)
			Expect(receivedAuthHeader).To(BeEmpty())
		})

		It("should set Content-Type to application/json", func() {
			var receivedContentType string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedContentType = r.Header.Get("Content-Type")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			_ = b.CreateCollection(context.Background(), "test", 768)
			Expect(receivedContentType).To(Equal("application/json"))
		})

		It("should truncate long error messages from server", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(strings.Repeat("x", 1000)))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.CreateCollection(context.Background(), "test", 768)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("..."))
		})
	})

	Context("end-to-end flow with mock server", func() {
		It("should create, insert, search, and delete through a single mock", func() {
			var createdID string
			stores := make(map[string]string) // id -> name

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				// CreateCollection
				case r.Method == "POST" && r.URL.Path == "/v1/vector_stores":
					var body map[string]interface{}
					_ = json.NewDecoder(r.Body).Decode(&body)
					createdID = fmt.Sprintf("vs_%s", body["name"])
					stores[createdID] = body["name"].(string)
					w.WriteHeader(http.StatusOK)
					fmt.Fprintf(w, `{"id": "%s", "name": "%s"}`, createdID, body["name"])

				// ListStores (for resolveStoreID)
				case r.Method == "GET" && r.URL.Path == "/v1/vector_stores":
					data := "["
					i := 0
					for id, name := range stores {
						if i > 0 {
							data += ","
						}
						data += fmt.Sprintf(`{"id": "%s", "name": "%s", "created_at": %d}`, id, name, 100+i)
						i++
					}
					data += "]"
					w.WriteHeader(http.StatusOK)
					fmt.Fprintf(w, `{"data": %s}`, data)

				// CollectionExists
				case r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/v1/vector_stores/"):
					id := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
					if _, ok := stores[id]; ok {
						w.WriteHeader(http.StatusOK)
						fmt.Fprintf(w, `{"id": "%s"}`, id)
					} else {
						w.WriteHeader(http.StatusNotFound)
						_, _ = w.Write([]byte(`{"error": "not found"}`))
					}

				// InsertChunks
				case r.Method == "POST" && r.URL.Path == "/v1/vector-io/insert":
					w.WriteHeader(http.StatusNoContent)

				// Search
				case r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/search"):
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(`{
						"data": [
							{"content": [{"type": "text", "text": "found it"}], "score": 0.9}
						]
					}`))

				// DeleteCollection
				case r.Method == "DELETE" && strings.HasPrefix(r.URL.Path, "/v1/vector_stores/"):
					id := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
					delete(stores, id)
					w.WriteHeader(http.StatusOK)

				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:       server.URL,
				EmbeddingModel: "test-model",
			})
			Expect(err).NotTo(HaveOccurred())
			ctx := context.Background()

			// Create
			err = b.CreateCollection(ctx, "e2e-store", 384)
			Expect(err).NotTo(HaveOccurred())

			// Exists
			exists, err := b.CollectionExists(ctx, "e2e-store")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())

			// Insert
			chunks := []EmbeddedChunk{
				{ID: "c1", Content: "test content", Embedding: []float32{0.1, 0.2}},
			}
			err = b.InsertChunks(ctx, "e2e-store", chunks)
			Expect(err).NotTo(HaveOccurred())

			// Search
			filter := map[string]interface{}{"_query_text": "test"}
			results, err := b.Search(ctx, "e2e-store", nil, 5, 0, filter)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
			Expect(results[0].Content).To(Equal("found it"))

			// Delete
			err = b.DeleteCollection(ctx, "e2e-store")
			Expect(err).NotTo(HaveOccurred())

			// Exists after delete
			exists, err = b.CollectionExists(ctx, "e2e-store")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})
	})

	Context("integration tests (require Llama Stack)", func() {
		skipLlamaStack := os.Getenv("SKIP_LLAMA_STACK_TESTS") != "false"

		var (
			endpoint       string
			embeddingModel string
		)

		BeforeEach(func() {
			if skipLlamaStack {
				Skip("Skipping Llama Stack tests (set SKIP_LLAMA_STACK_TESTS=false to enable)")
			}

			endpoint = os.Getenv("LLAMA_STACK_ENDPOINT")
			if endpoint == "" {
				endpoint = "http://localhost:8321"
			}
			embeddingModel = os.Getenv("LLAMA_STACK_EMBEDDING_MODEL")
			if embeddingModel == "" {
				embeddingModel = "sentence-transformers/all-MiniLM-L6-v2"
			}
		})

		It("should create, check, and delete a collection", func() {
			backend, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:       endpoint,
				EmbeddingModel: embeddingModel,
			})
			Expect(err).NotTo(HaveOccurred())
			defer backend.Close()

			ctx := context.Background()
			vsID := "test_integration_ls"

			// Clean up in case previous test left data.
			_ = backend.DeleteCollection(ctx, vsID)

			err = backend.CreateCollection(ctx, vsID, 384)
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = backend.DeleteCollection(ctx, vsID) }()

			exists, err := backend.CollectionExists(ctx, vsID)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())

			err = backend.DeleteCollection(ctx, vsID)
			Expect(err).NotTo(HaveOccurred())

			exists, err = backend.CollectionExists(ctx, vsID)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should insert chunks with embeddings and search", func() {
			backend, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:           endpoint,
				EmbeddingModel:     embeddingModel,
				EmbeddingDimension: 384,
			})
			Expect(err).NotTo(HaveOccurred())
			defer backend.Close()

			ctx := context.Background()
			vsID := "test_search_ls"

			_ = backend.DeleteCollection(ctx, vsID)

			err = backend.CreateCollection(ctx, vsID, 384)
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = backend.DeleteCollection(ctx, vsID) }()

			// Generate embeddings via Llama Stack's inference API.
			embeddings, err := generateTestEmbeddings(endpoint, embeddingModel, []string{
				"Kubernetes orchestrates containers across clusters",
				"Docker builds and runs container images",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).To(HaveLen(2))

			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "k8s.txt", Content: "Kubernetes orchestrates containers across clusters", Embedding: embeddings[0], ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "k8s.txt", Content: "Docker builds and runs container images", Embedding: embeddings[1], ChunkIndex: 1},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			filter := map[string]interface{}{"_query_text": "what is kubernetes?"}
			results, err := backend.Search(ctx, vsID, nil, 5, 0, filter)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))
		})
	})
})

// generateTestEmbeddings calls Llama Stack's inference API to generate
// embeddings for the given texts. Used only in integration tests.
func generateTestEmbeddings(endpoint, model string, texts []string) ([][]float32, error) {
	body := map[string]interface{}{
		"model": model,
		"input": texts,
	}
	jsonBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(endpoint+"/v1/embeddings", "application/json", strings.NewReader(string(jsonBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to call embeddings API: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read embeddings response: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("embeddings API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var embResp struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(respBody, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse embeddings response: %w", err)
	}

	result := make([][]float32, len(embResp.Data))
	for i, d := range embResp.Data {
		result[i] = d.Embedding
	}
	return result, nil
}
