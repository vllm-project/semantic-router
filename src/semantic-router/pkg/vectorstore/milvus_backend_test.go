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
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("MilvusBackend", func() {
	skipMilvus := os.Getenv("SKIP_MILVUS_TESTS") != "false"

	Context("integration tests (require Milvus)", func() {
		BeforeEach(func() {
			if skipMilvus {
				Skip("Skipping Milvus tests (set SKIP_MILVUS_TESTS=false to enable)")
			}
		})

		It("should create and delete collection", func() {
			addr := os.Getenv("MILVUS_ADDRESS")
			if addr == "" {
				addr = "localhost:19530"
			}

			backend, err := NewMilvusBackend(MilvusBackendConfig{
				Address: addr,
			})
			Expect(err).NotTo(HaveOccurred())
			defer backend.Close()

			ctx := context.Background()

			err = backend.CreateCollection(ctx, "test_integration", 3)
			Expect(err).NotTo(HaveOccurred())

			exists, err := backend.CollectionExists(ctx, "test_integration")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())

			err = backend.DeleteCollection(ctx, "test_integration")
			Expect(err).NotTo(HaveOccurred())

			exists, err = backend.CollectionExists(ctx, "test_integration")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should insert and search chunks", func() {
			addr := os.Getenv("MILVUS_ADDRESS")
			if addr == "" {
				addr = "localhost:19530"
			}

			backend, err := NewMilvusBackend(MilvusBackendConfig{
				Address: addr,
			})
			Expect(err).NotTo(HaveOccurred())
			defer backend.Close()

			ctx := context.Background()
			vsID := "test_search"

			// Clean up in case previous test left data.
			_ = backend.DeleteCollection(ctx, vsID)

			err = backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = backend.DeleteCollection(ctx, vsID) }()

			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0.5, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))
			Expect(results[0].Content).To(Equal("hello"))
		})
	})

	Context("unit tests (no Milvus required)", func() {
		It("should require address", func() {
			_, err := NewMilvusBackend(MilvusBackendConfig{})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("milvus address is required"))
		})

		It("should use default config values", func() {
			// We can't actually connect, but we can verify the config defaults
			// by passing an invalid address and checking the error.
			_, err := NewMilvusBackend(MilvusBackendConfig{
				Address:        "invalid:99999",
				ConnectTimeout: 1,
			})
			Expect(err).To(HaveOccurred())
			// Should fail on connection, not on config validation.
			Expect(err.Error()).To(ContainSubstring("failed to connect"))
		})

		It("should generate correct collection names", func() {
			// Test the collection name generation without connecting.
			mb := &MilvusBackend{collectionPrefix: "vsr_vs_"}
			Expect(mb.collectionName("store123")).To(Equal("vsr_vs_store123"))
		})

		It("should use custom prefix", func() {
			mb := &MilvusBackend{collectionPrefix: "custom_"}
			Expect(mb.collectionName("abc")).To(Equal("custom_abc"))
		})
	})
})
