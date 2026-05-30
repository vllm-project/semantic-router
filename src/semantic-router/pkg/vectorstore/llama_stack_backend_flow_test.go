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
	"net/http"
	"net/http/httptest"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LlamaStackBackend mock flow", func() {
	It("should create, insert, search, and delete through a single mock", func() {
		server := newLlamaStackBackendFlowServer()
		defer server.Close()

		b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
			Endpoint:       server.URL,
			EmbeddingModel: "test-model",
		})
		Expect(err).NotTo(HaveOccurred())
		ctx := context.Background()

		err = b.CreateCollection(ctx, "e2e-store", 384)
		Expect(err).NotTo(HaveOccurred())

		exists, err := b.CollectionExists(ctx, "e2e-store")
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeTrue())

		chunks := []EmbeddedChunk{
			{ID: "c1", Content: "test content", Embedding: []float32{0.1, 0.2}},
		}
		err = b.InsertChunks(ctx, "e2e-store", chunks)
		Expect(err).NotTo(HaveOccurred())

		filter := map[string]interface{}{"_query_text": "test"}
		results, err := b.Search(ctx, "e2e-store", nil, 5, 0, filter)
		Expect(err).NotTo(HaveOccurred())
		Expect(results).To(HaveLen(1))
		Expect(results[0].Content).To(Equal("found it"))

		err = b.DeleteCollection(ctx, "e2e-store")
		Expect(err).NotTo(HaveOccurred())

		exists, err = b.CollectionExists(ctx, "e2e-store")
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeFalse())
	})
})

func newLlamaStackBackendFlowServer() *httptest.Server {
	stores := make(map[string]string)
	routes := mockFlowRoutes(stores)
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, route := range routes {
			if route.matches(r) {
				route.handle(w, r)
				return
			}
		}
		w.WriteHeader(http.StatusNotFound)
	}))
}

type mockFlowRoute struct {
	matches func(*http.Request) bool
	handle  func(http.ResponseWriter, *http.Request)
}

func mockFlowRoutes(stores map[string]string) []mockFlowRoute {
	return []mockFlowRoute{
		{
			matches: func(r *http.Request) bool {
				return r.Method == "POST" && r.URL.Path == "/v1/vector_stores"
			},
			handle: func(w http.ResponseWriter, r *http.Request) {
				handleMockFlowCreate(w, r, stores)
			},
		},
		{
			matches: func(r *http.Request) bool {
				return r.Method == "GET" && r.URL.Path == "/v1/vector_stores"
			},
			handle: func(w http.ResponseWriter, _ *http.Request) {
				handleMockFlowList(w, stores)
			},
		},
		{
			matches: func(r *http.Request) bool {
				return r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/v1/vector_stores/")
			},
			handle: func(w http.ResponseWriter, r *http.Request) {
				handleMockFlowGet(w, r, stores)
			},
		},
		{
			matches: func(r *http.Request) bool {
				return r.Method == "POST" && r.URL.Path == "/v1/vector-io/insert"
			},
			handle: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusNoContent)
			},
		},
		{
			matches: func(r *http.Request) bool {
				return r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/search")
			},
			handle: handleMockFlowSearch,
		},
		{
			matches: func(r *http.Request) bool {
				return r.Method == "DELETE" && strings.HasPrefix(r.URL.Path, "/v1/vector_stores/")
			},
			handle: func(w http.ResponseWriter, r *http.Request) {
				handleMockFlowDelete(w, r, stores)
			},
		},
	}
}

func handleMockFlowCreate(w http.ResponseWriter, r *http.Request, stores map[string]string) {
	var body map[string]interface{}
	_ = json.NewDecoder(r.Body).Decode(&body)
	createdID := fmt.Sprintf("vs_%s", body["name"])
	stores[createdID] = body["name"].(string)
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"id": "%s", "name": "%s"}`, createdID, body["name"])
}

func handleMockFlowList(w http.ResponseWriter, stores map[string]string) {
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
}

func handleMockFlowSearch(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{
		"data": [
			{"content": [{"type": "text", "text": "found it"}], "score": 0.9}
		]
	}`))
}

func handleMockFlowDelete(w http.ResponseWriter, r *http.Request, stores map[string]string) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
	delete(stores, id)
	w.WriteHeader(http.StatusOK)
}

func handleMockFlowGet(w http.ResponseWriter, r *http.Request, stores map[string]string) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
	if _, ok := stores[id]; ok {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"id": "%s"}`, id)
		return
	}
	w.WriteHeader(http.StatusNotFound)
	_, _ = w.Write([]byte(`{"error": "not found"}`))
}
