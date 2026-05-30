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
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LlamaStackBackend integration", func() {
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
