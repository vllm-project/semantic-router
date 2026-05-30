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

var _ = Describe("LlamaStackBackend search #1", func() {
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
		Expect(receivedBody).NotTo(HaveKey("ranking_options"))
	})
})

var _ = Describe("LlamaStackBackend search #2", func() {
	It("should include ranking_options with rrf ranker for hybrid search", func() {
		var receivedBody map[string]interface{}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &receivedBody)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"data": []}`))
		}))
		defer server.Close()

		b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
			Endpoint:   server.URL,
			SearchType: "hybrid",
		})
		Expect(err).NotTo(HaveOccurred())
		b.storeIDs["my-store"] = "vs_gen_abc123"

		filter := map[string]interface{}{"_query_text": "what is kubernetes?"}
		_, err = b.Search(context.Background(), "my-store", nil, 5, 0, filter)
		Expect(err).NotTo(HaveOccurred())

		Expect(receivedBody).To(HaveKey("ranking_options"))
		opts, ok := receivedBody["ranking_options"].(map[string]interface{})
		Expect(ok).To(BeTrue())
		Expect(opts["ranker"]).To(Equal("rrf"))
	})
})

var _ = Describe("LlamaStackBackend search #3", func() {
	It("should not include ranking_options for default vector search", func() {
		var receivedBody map[string]interface{}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &receivedBody)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"data": []}`))
		}))
		defer server.Close()

		b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
			Endpoint:   server.URL,
			SearchType: "vector",
		})
		Expect(err).NotTo(HaveOccurred())
		b.storeIDs["my-store"] = "vs_gen_abc123"

		filter := map[string]interface{}{"_query_text": "kubernetes"}
		_, err = b.Search(context.Background(), "my-store", nil, 5, 0, filter)
		Expect(err).NotTo(HaveOccurred())

		Expect(receivedBody).NotTo(HaveKey("ranking_options"))
	})
})

var _ = Describe("LlamaStackBackend search #4", func() {
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
})

var _ = Describe("LlamaStackBackend search #5", func() {
	It("should apply threshold filtering for vector search", func() {
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
})

var _ = Describe("LlamaStackBackend search #6", func() {
	It("should skip threshold filtering for hybrid search (RRF scores are not similarity)", func() {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			resp := `{
					"data": [
						{"content": [{"type": "text", "text": "rrf result 1"}], "score": 0.039},
						{"content": [{"type": "text", "text": "rrf result 2"}], "score": 0.028},
						{"content": [{"type": "text", "text": "rrf result 3"}], "score": 0.010}
					]
				}`
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(resp))
		}))
		defer server.Close()

		b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
			Endpoint:   server.URL,
			SearchType: "hybrid",
		})
		Expect(err).NotTo(HaveOccurred())
		b.storeIDs["my-store"] = "vs_gen_abc123"

		filter := map[string]interface{}{"_query_text": "test"}
		results, err := b.Search(context.Background(), "my-store", nil, 10, 0.7, filter)
		Expect(err).NotTo(HaveOccurred())
		Expect(results).To(HaveLen(3), "hybrid search must return all results regardless of threshold")
		Expect(results[0].Content).To(Equal("rrf result 1"))
		Expect(results[0].Score).To(BeNumerically("~", 0.039, 0.001))
	})
})

var _ = Describe("LlamaStackBackend search #7", func() {
	It("should fail when _query_text is missing from filter", func() {
		b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://unused"})
		Expect(err).NotTo(HaveOccurred())

		_, err = b.Search(context.Background(), "vs_abc123", nil, 5, 0, nil)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("_query_text"))
	})
})

var _ = Describe("LlamaStackBackend search #8", func() {
	It("should fail when _query_text is empty string", func() {
		b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://unused"})
		Expect(err).NotTo(HaveOccurred())

		filter := map[string]interface{}{"_query_text": ""}
		_, err = b.Search(context.Background(), "vs_abc123", nil, 5, 0, filter)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("_query_text"))
	})
})

var _ = Describe("LlamaStackBackend search #9", func() {
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
})

var _ = Describe("LlamaStackBackend search #10", func() {
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
})

var _ = Describe("LlamaStackBackend search #11", func() {
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
})

var _ = Describe("LlamaStackBackend search #12", func() {
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
