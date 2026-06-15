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
	"io"
	"net/http"
	"net/http/httptest"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LlamaStackBackend collection lifecycle #1", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #2", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #3", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #4", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #5", func() {
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

var _ = Describe("LlamaStackBackend collection lifecycle #6", func() {
	It("should return cached ID when available", func() {
		b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://unused"})
		Expect(err).NotTo(HaveOccurred())

		b.storeIDs["my-store"] = "vs_cached_123"

		id, err := b.resolveStoreID(context.Background(), "my-store")
		Expect(err).NotTo(HaveOccurred())
		Expect(id).To(Equal("vs_cached_123"))
	})
})

var _ = Describe("LlamaStackBackend collection lifecycle #7", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #8", func() {
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

var _ = Describe("LlamaStackBackend collection lifecycle #9", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #10", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #11", func() {
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

var _ = Describe("LlamaStackBackend collection lifecycle #12", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #13", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #14", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #15", func() {
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
})

var _ = Describe("LlamaStackBackend collection lifecycle #16", func() {
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
