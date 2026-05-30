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

var _ = Describe("LlamaStackBackend chunk mutation #1", func() {
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
})

var _ = Describe("LlamaStackBackend chunk mutation #2", func() {
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
})

var _ = Describe("LlamaStackBackend chunk mutation #3", func() {
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
})

var _ = Describe("LlamaStackBackend chunk mutation #4", func() {
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

var _ = Describe("LlamaStackBackend chunk mutation #5", func() {
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
})

var _ = Describe("LlamaStackBackend chunk mutation #6", func() {
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
})

var _ = Describe("LlamaStackBackend chunk mutation #7", func() {
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
})

var _ = Describe("LlamaStackBackend chunk mutation #8", func() {
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
