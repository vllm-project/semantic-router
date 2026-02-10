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
	"math"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("MemoryBackend", func() {
	var (
		backend *MemoryBackend
		ctx     context.Context
	)

	BeforeEach(func() {
		backend = NewMemoryBackend(MemoryBackendConfig{})
		ctx = context.Background()
	})

	Context("CreateCollection", func() {
		It("should create a new collection", func() {
			err := backend.CreateCollection(ctx, "vs_test", 768)
			Expect(err).NotTo(HaveOccurred())

			exists, err := backend.CollectionExists(ctx, "vs_test")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())
		})

		It("should return error for duplicate collection", func() {
			err := backend.CreateCollection(ctx, "vs_dup", 768)
			Expect(err).NotTo(HaveOccurred())

			err = backend.CreateCollection(ctx, "vs_dup", 768)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("already exists"))
		})
	})

	Context("DeleteCollection", func() {
		It("should delete an existing collection", func() {
			err := backend.CreateCollection(ctx, "vs_del", 768)
			Expect(err).NotTo(HaveOccurred())

			err = backend.DeleteCollection(ctx, "vs_del")
			Expect(err).NotTo(HaveOccurred())

			exists, err := backend.CollectionExists(ctx, "vs_del")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should return error for non-existent collection", func() {
			err := backend.DeleteCollection(ctx, "vs_nonexistent")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not found"))
		})
	})

	Context("CollectionExists", func() {
		It("should return false for non-existent collection", func() {
			exists, err := backend.CollectionExists(ctx, "vs_nope")
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})
	})

	Context("InsertChunks", func() {
		BeforeEach(func() {
			err := backend.CreateCollection(ctx, "vs_insert", 3)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should insert chunks successfully", func() {
			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: []float32{1, 0, 0}, ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: []float32{0, 1, 0}, ChunkIndex: 1},
			}
			err := backend.InsertChunks(ctx, "vs_insert", chunks)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return error for non-existent collection", func() {
			chunks := []EmbeddedChunk{{ID: "c1"}}
			err := backend.InsertChunks(ctx, "vs_nonexistent", chunks)
			Expect(err).To(HaveOccurred())
		})

		It("should respect max entries limit", func() {
			limited := NewMemoryBackend(MemoryBackendConfig{MaxEntriesPerStore: 2})
			err := limited.CreateCollection(ctx, "vs_limit", 3)
			Expect(err).NotTo(HaveOccurred())

			chunks := []EmbeddedChunk{
				{ID: "c1", Embedding: []float32{1, 0, 0}},
				{ID: "c2", Embedding: []float32{0, 1, 0}},
			}
			err = limited.InsertChunks(ctx, "vs_limit", chunks)
			Expect(err).NotTo(HaveOccurred())

			extra := []EmbeddedChunk{{ID: "c3", Embedding: []float32{0, 0, 1}}}
			err = limited.InsertChunks(ctx, "vs_limit", extra)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("maximum entries"))
		})
	})

	Context("DeleteByFileID", func() {
		BeforeEach(func() {
			err := backend.CreateCollection(ctx, "vs_delf", 3)
			Expect(err).NotTo(HaveOccurred())

			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Content: "a", Embedding: []float32{1, 0, 0}},
				{ID: "c2", FileID: "f1", Content: "b", Embedding: []float32{0, 1, 0}},
				{ID: "c3", FileID: "f2", Content: "c", Embedding: []float32{0, 0, 1}},
			}
			err = backend.InsertChunks(ctx, "vs_delf", chunks)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should delete only chunks for the specified file", func() {
			err := backend.DeleteByFileID(ctx, "vs_delf", "f1")
			Expect(err).NotTo(HaveOccurred())

			// Search for remaining - only f2 chunks should remain.
			results, err := backend.Search(ctx, "vs_delf", []float32{0, 0, 1}, 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
			Expect(results[0].FileID).To(Equal("f2"))
		})

		It("should return error for non-existent collection", func() {
			err := backend.DeleteByFileID(ctx, "vs_nonexistent", "f1")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("Search", func() {
		BeforeEach(func() {
			err := backend.CreateCollection(ctx, "vs_search", 3)
			Expect(err).NotTo(HaveOccurred())

			// Insert normalized vectors for predictable cosine similarity.
			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "alpha", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "beta", Embedding: normalizeVec([]float32{0.9, 0.1, 0}), ChunkIndex: 1},
				{ID: "c3", FileID: "f2", Filename: "b.txt", Content: "gamma", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, "vs_search", chunks)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return results sorted by score descending", func() {
			query := normalizeVec([]float32{1, 0, 0})
			results, err := backend.Search(ctx, "vs_search", query, 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 2))

			// First result should be the most similar.
			Expect(results[0].Content).To(Equal("alpha"))
			Expect(results[0].Score).To(BeNumerically(">", results[1].Score))
		})

		It("should respect threshold", func() {
			query := normalizeVec([]float32{1, 0, 0})
			results, err := backend.Search(ctx, "vs_search", query, 10, 0.999, nil)
			Expect(err).NotTo(HaveOccurred())

			// Only the exact match should pass a very high threshold.
			Expect(results).To(HaveLen(1))
			Expect(results[0].Content).To(Equal("alpha"))
		})

		It("should respect topK", func() {
			query := normalizeVec([]float32{1, 0, 0})
			results, err := backend.Search(ctx, "vs_search", query, 1, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
		})

		It("should filter by file_id", func() {
			query := normalizeVec([]float32{1, 0, 0})
			results, err := backend.Search(ctx, "vs_search", query, 10, 0,
				map[string]interface{}{"file_id": "f2"})
			Expect(err).NotTo(HaveOccurred())

			for _, r := range results {
				Expect(r.FileID).To(Equal("f2"))
			}
		})

		It("should return error for non-existent collection", func() {
			_, err := backend.Search(ctx, "vs_nonexistent", []float32{1, 0, 0}, 10, 0, nil)
			Expect(err).To(HaveOccurred())
		})

		It("should return empty results for empty collection", func() {
			err := backend.CreateCollection(ctx, "vs_empty", 3)
			Expect(err).NotTo(HaveOccurred())

			results, err := backend.Search(ctx, "vs_empty", []float32{1, 0, 0}, 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})
	})

	Context("Close", func() {
		It("should not return an error", func() {
			err := backend.Close()
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("interface compliance", func() {
		It("should implement VectorStoreBackend", func() {
			var _ VectorStoreBackend = backend
		})
	})
})

var _ = Describe("cosineSimilarity", func() {
	It("should return 1.0 for identical normalized vectors", func() {
		v := normalizeVec([]float32{1, 2, 3})
		sim := cosineSimilarity(v, v)
		Expect(sim).To(BeNumerically("~", 1.0, 0.001))
	})

	It("should return 0.0 for orthogonal vectors", func() {
		a := []float32{1, 0, 0}
		b := []float32{0, 1, 0}
		sim := cosineSimilarity(a, b)
		Expect(sim).To(BeNumerically("~", 0.0, 0.001))
	})

	It("should return 0.0 for mismatched dimensions", func() {
		a := []float32{1, 0}
		b := []float32{1, 0, 0}
		sim := cosineSimilarity(a, b)
		Expect(sim).To(Equal(0.0))
	})

	It("should return 0.0 for empty vectors", func() {
		sim := cosineSimilarity([]float32{}, []float32{})
		Expect(sim).To(Equal(0.0))
	})

	It("should return 0.0 for zero vectors", func() {
		sim := cosineSimilarity([]float32{0, 0, 0}, []float32{1, 0, 0})
		Expect(sim).To(Equal(0.0))
	})
})

var _ = Describe("NewBackend factory", func() {
	It("should create memory backend", func() {
		b, err := NewBackend(BackendTypeMemory, MemoryBackendConfig{}, MilvusBackendConfig{})
		Expect(err).NotTo(HaveOccurred())
		Expect(b).NotTo(BeNil())

		_, ok := b.(*MemoryBackend)
		Expect(ok).To(BeTrue())
	})

	It("should return error for unsupported type", func() {
		_, err := NewBackend("redis", MemoryBackendConfig{}, MilvusBackendConfig{})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("unsupported backend type"))
	})

	It("should return error for milvus without address", func() {
		_, err := NewBackend(BackendTypeMilvus, MemoryBackendConfig{}, MilvusBackendConfig{})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("milvus address is required"))
	})
})

// normalizeVec returns a unit vector.
func normalizeVec(v []float32) []float32 {
	var norm float64
	for _, val := range v {
		norm += float64(val) * float64(val)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = float32(float64(val) / norm)
	}
	return result
}

// Verify MemoryBackend satisfies the interface at compile time.
var _ VectorStoreBackend = (*MemoryBackend)(nil)

// Verify MilvusBackend satisfies the interface at compile time.
var _ VectorStoreBackend = (*MilvusBackend)(nil)
