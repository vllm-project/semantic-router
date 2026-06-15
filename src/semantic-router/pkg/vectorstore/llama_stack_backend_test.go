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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Compile-time interface check.
var _ VectorStoreBackend = (*LlamaStackBackend)(nil)

var _ = Describe("LlamaStackBackend construction", func() {
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

		It("should default searchType to 'vector' when not specified", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint: "http://localhost:8321",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(b.searchType).To(Equal("vector"))
		})

		It("should accept 'hybrid' search type", func() {
			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:   "http://localhost:8321",
				SearchType: "hybrid",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(b.searchType).To(Equal("hybrid"))
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
})
