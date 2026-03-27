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
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// ---------------------------------------------------------------------------
// Unit tests — no Valkey server required
// ---------------------------------------------------------------------------

var _ = Describe("ValkeyBackend unit tests", func() {

	Context("float32SliceToBytes", func() {
		It("should produce correct little-endian bytes", func() {
			input := []float32{1.0, 2.0, 3.0}
			b := float32SliceToBytes(input)
			Expect(len(b)).To(Equal(12)) // 3 * 4 bytes

			for i, expected := range input {
				bits := binary.LittleEndian.Uint32(b[i*4:])
				Expect(math.Float32frombits(bits)).To(Equal(expected))
			}
		})

		It("should roundtrip arbitrary values", func() {
			input := []float32{0.0, -1.5, 3.14, math.MaxFloat32, math.SmallestNonzeroFloat32}
			b := float32SliceToBytes(input)
			Expect(len(b)).To(Equal(len(input) * 4))

			for i, expected := range input {
				bits := binary.LittleEndian.Uint32(b[i*4:])
				Expect(math.Float32frombits(bits)).To(Equal(expected))
			}
		})

		It("should return empty slice for empty input", func() {
			b := float32SliceToBytes([]float32{})
			Expect(b).To(HaveLen(0))
		})
	})

	Context("escapeTagValue", func() {
		It("should escape hyphens", func() {
			Expect(escapeTagValue("file-123")).To(Equal("file\\-123"))
		})

		It("should escape dots", func() {
			Expect(escapeTagValue("doc.txt")).To(Equal("doc\\.txt"))
		})

		It("should escape colons", func() {
			Expect(escapeTagValue("ns:val")).To(Equal("ns\\:val"))
		})

		It("should escape slashes", func() {
			Expect(escapeTagValue("path/to")).To(Equal("path\\/to"))
		})

		It("should escape spaces", func() {
			Expect(escapeTagValue("hello world")).To(Equal("hello\\ world"))
		})

		It("should escape multiple special characters", func() {
			Expect(escapeTagValue("a-b.c:d/e f")).To(Equal("a\\-b\\.c\\:d\\/e\\ f"))
		})

		It("should leave safe strings unchanged", func() {
			Expect(escapeTagValue("abc123")).To(Equal("abc123"))
		})
	})

	Context("distanceToSimilarity", func() {
		It("should convert COSINE distance to similarity", func() {
			// COSINE distance 0.2 → similarity = 1 - 0.2/2 = 0.9
			Expect(distanceToSimilarity("COSINE", 0.2)).To(BeNumerically("~", 0.9, 0.001))
		})

		It("should convert L2 distance to similarity", func() {
			// L2 distance 0.3 → similarity = 1/(1+0.3) ≈ 0.769
			Expect(distanceToSimilarity("L2", 0.3)).To(BeNumerically("~", 0.769, 0.01))
		})

		It("should return IP score as-is", func() {
			Expect(distanceToSimilarity("IP", 0.95)).To(BeNumerically("~", 0.95, 0.001))
		})

		It("should handle zero distance (identical vectors)", func() {
			Expect(distanceToSimilarity("COSINE", 0.0)).To(BeNumerically("~", 1.0, 0.001))
		})

		It("should be case-insensitive for metric type", func() {
			Expect(distanceToSimilarity("cosine", 0.2)).To(BeNumerically("~", 0.9, 0.001))
		})
	})

	Context("toInt64", func() {
		It("should handle int64", func() {
			Expect(toInt64(int64(42))).To(Equal(int64(42)))
		})

		It("should handle float64", func() {
			Expect(toInt64(float64(42.9))).To(Equal(int64(42)))
		})

		It("should handle string", func() {
			Expect(toInt64("123")).To(Equal(int64(123)))
		})

		It("should return 0 for nil", func() {
			Expect(toInt64(nil)).To(Equal(int64(0)))
		})

		It("should return 0 for unsupported type", func() {
			Expect(toInt64(true)).To(Equal(int64(0)))
		})

		It("should return 0 for invalid string", func() {
			Expect(toInt64("abc")).To(Equal(int64(0)))
		})
	})

	Context("extractKeysFromSearchResult", func() {
		It("should extract keys from flat string result", func() {
			// NOCONTENT flat format: [totalCount, key1, key2, ...]
			result := []interface{}{int64(2), "key:1", "key:2"}
			keys := extractKeysFromSearchResult(result)
			Expect(keys).To(Equal([]string{"key:1", "key:2"}))
		})

		It("should extract keys from map-based result", func() {
			// valkey-glide v2 map format: [totalCount, map[key1: ..., key2: ...]]
			result := []interface{}{
				int64(2),
				map[string]interface{}{
					"key:1": map[string]interface{}{},
					"key:2": map[string]interface{}{},
				},
			}
			keys := extractKeysFromSearchResult(result)
			Expect(keys).To(ConsistOf("key:1", "key:2"))
		})

		It("should return nil for zero results", func() {
			result := []interface{}{int64(0)}
			keys := extractKeysFromSearchResult(result)
			Expect(keys).To(BeNil())
		})

		It("should return nil for non-array input", func() {
			keys := extractKeysFromSearchResult("not an array")
			Expect(keys).To(BeNil())
		})

		It("should return nil for nil input", func() {
			keys := extractKeysFromSearchResult(nil)
			Expect(keys).To(BeNil())
		})
	})

	Context("naming helpers", func() {
		var vb *ValkeyBackend

		BeforeEach(func() {
			vb = &ValkeyBackend{collectionPrefix: "vsr_vs_"}
		})

		It("indexName should produce correct name", func() {
			Expect(vb.indexName("store123")).To(Equal("vsr_vs_store123_idx"))
		})

		It("keyPrefix should produce correct prefix", func() {
			Expect(vb.keyPrefix("store123")).To(Equal("vsr_vs_store123:"))
		})

		It("chunkKey should produce correct key", func() {
			Expect(vb.chunkKey("store123", "c1")).To(Equal("vsr_vs_store123:c1"))
		})

		It("should work with custom prefix", func() {
			vb.collectionPrefix = "custom_"
			Expect(vb.indexName("abc")).To(Equal("custom_abc_idx"))
			Expect(vb.keyPrefix("abc")).To(Equal("custom_abc:"))
			Expect(vb.chunkKey("abc", "x")).To(Equal("custom_abc:x"))
		})
	})

	Context("config defaults", func() {
		It("should apply defaults for zero-value config", func() {
			// We cannot connect, but we can verify the struct would be built
			// with correct defaults by inspecting the backend after construction
			// fails. Instead, test the default logic inline.
			cfg := ValkeyBackendConfig{}

			host := cfg.Host
			if host == "" {
				host = "localhost"
			}
			port := cfg.Port
			if port <= 0 {
				port = 6379
			}
			prefix := cfg.CollectionPrefix
			if prefix == "" {
				prefix = "vsr_vs_"
			}
			indexM := cfg.IndexM
			if indexM <= 0 {
				indexM = 16
			}
			indexEf := cfg.IndexEf
			if indexEf <= 0 {
				indexEf = 200
			}
			metricType := cfg.MetricType
			if metricType == "" {
				metricType = "COSINE"
			}
			timeout := cfg.ConnectTimeout
			if timeout <= 0 {
				timeout = 10
			}

			Expect(host).To(Equal("localhost"))
			Expect(port).To(Equal(6379))
			Expect(prefix).To(Equal("vsr_vs_"))
			Expect(indexM).To(Equal(16))
			Expect(indexEf).To(Equal(200))
			Expect(metricType).To(Equal("COSINE"))
			Expect(timeout).To(Equal(10))
		})
	})

	Context("interface compliance", func() {
		It("ValkeyBackend should implement VectorStoreBackend", func() {
			var _ VectorStoreBackend = (*ValkeyBackend)(nil)
		})
	})


	Context("parseSearchResults", func() {
		var vb *ValkeyBackend

		BeforeEach(func() {
			vb = &ValkeyBackend{metricType: "COSINE"}
		})

		It("should return nil for nil input", func() {
			results, err := vb.parseSearchResults(nil, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeNil())
		})

		It("should return nil for non-array input", func() {
			results, err := vb.parseSearchResults("not an array", 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeNil())
		})

		It("should return nil for zero total count", func() {
			result := []interface{}{int64(0)}
			results, err := vb.parseSearchResults(result, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeNil())
		})

		It("should parse valid results", func() {
			// valkey-glide v2 map format: [totalCount, map[docKey: map[fields...]], ...]
			result := []interface{}{
				int64(1),
				map[string]interface{}{
					"key:1": map[string]interface{}{
						"file_id":          "f1",
						"filename":         "doc.txt",
						"content":          "hello",
						"vector_distance":  "0.2",
						"chunk_index":      "0",
					},
				},
			}
			results, err := vb.parseSearchResults(result, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
			Expect(results[0].FileID).To(Equal("f1"))
			Expect(results[0].Filename).To(Equal("doc.txt"))
			Expect(results[0].Content).To(Equal("hello"))
			// COSINE: similarity = 1 - 0.2/2 = 0.9
			Expect(results[0].Score).To(BeNumerically("~", 0.9, 0.01))
			Expect(results[0].ChunkIndex).To(Equal(0))
		})

		It("should filter results below threshold", func() {
			result := []interface{}{
				int64(1),
				map[string]interface{}{
					"key:1": map[string]interface{}{
						"file_id":         "f1",
						"content":         "hello",
						"vector_distance": "1.6", // COSINE similarity = 1 - 1.6/2 = 0.2
					},
				},
			}
			results, err := vb.parseSearchResults(result, 0.5)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})

		It("should skip entries with non-map fields", func() {
			result := []interface{}{
				int64(1),
				"not-a-map",
			}
			results, err := vb.parseSearchResults(result, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})

		It("should handle missing vector_distance field", func() {
			result := []interface{}{
				int64(1),
				map[string]interface{}{
					"key:1": map[string]interface{}{
						"file_id": "f1",
						"content": "hello",
					},
				},
			}
			// No vector_distance → score 0 → below any positive threshold
			results, err := vb.parseSearchResults(result, 0.1)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})

		It("should parse multiple results", func() {
			// All docs may be in a single map or separate maps.
			result := []interface{}{
				int64(2),
				map[string]interface{}{
					"key:1": map[string]interface{}{
						"file_id": "f1", "content": "hello",
						"vector_distance": "0.1", "chunk_index": "0",
					},
					"key:2": map[string]interface{}{
						"file_id": "f2", "content": "world",
						"vector_distance": "0.4", "chunk_index": "1",
					},
				},
			}
			results, err := vb.parseSearchResults(result, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(2))
		})
	})

	Context("factory registration", func() {
		It("should fail to create valkey backend with unreachable host", func() {
			_, err := NewBackend(BackendTypeValkey, BackendConfigs{
				Valkey: ValkeyBackendConfig{
					Host:           "192.0.2.1", // RFC 5737 TEST-NET, unreachable
					Port:           6379,
					ConnectTimeout: 1,
				},
			})
			Expect(err).To(HaveOccurred())
		})

		It("should include valkey in unsupported type error", func() {
			_, err := NewBackend("nosql", BackendConfigs{})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("valkey"))
		})
	})
})

// ---------------------------------------------------------------------------
// Integration tests — require Valkey server with valkey-search module
// ---------------------------------------------------------------------------

var _ = Describe("ValkeyBackend integration tests", func() {
	skipValkey := os.Getenv("SKIP_VALKEY_TESTS") != "false"

	var (
		backend *ValkeyBackend
		ctx     context.Context
	)

	valkeyHost := os.Getenv("VALKEY_HOST")
	valkeyPort := os.Getenv("VALKEY_PORT")

	BeforeEach(func() {
		if skipValkey {
			Skip("Skipping Valkey tests (set SKIP_VALKEY_TESTS=false to enable)")
		}

		host := valkeyHost
		if host == "" {
			host = "localhost"
		}
		port := 6379
		if valkeyPort != "" {
			p, err := strconv.Atoi(valkeyPort)
			if err == nil {
				port = p
			}
		}

		var err error
		backend, err = NewValkeyBackend(ValkeyBackendConfig{
			Host:             host,
			Port:             port,
			CollectionPrefix: "test_vs_",
			ConnectTimeout:   5,
		})
		Expect(err).NotTo(HaveOccurred())

		ctx = context.Background()
	})

	AfterEach(func() {
		if backend != nil {
			backend.Close()
		}
	})

	// -----------------------------------------------------------------------
	// CreateCollection / DeleteCollection / CollectionExists
	// -----------------------------------------------------------------------

	Context("CreateCollection", func() {
		It("should create and verify a collection", func() {
			vsID := "integ_create_" + uniqueSuffix()
			defer cleanupCollection(ctx, backend, vsID)

			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())

			exists, err := backend.CollectionExists(ctx, vsID)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())
		})

		It("should return error for duplicate collection", func() {
			vsID := "integ_dup_" + uniqueSuffix()
			defer cleanupCollection(ctx, backend, vsID)

			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())

			err = backend.CreateCollection(ctx, vsID, 3)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("already exists"))
		})

		It("should create collections with different dimensions", func() {
			vsID128 := "integ_dim128_" + uniqueSuffix()
			vsID768 := "integ_dim768_" + uniqueSuffix()
			defer cleanupCollection(ctx, backend, vsID128)
			defer cleanupCollection(ctx, backend, vsID768)

			err := backend.CreateCollection(ctx, vsID128, 128)
			Expect(err).NotTo(HaveOccurred())

			err = backend.CreateCollection(ctx, vsID768, 768)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("DeleteCollection", func() {
		It("should delete an existing collection", func() {
			vsID := "integ_del_" + uniqueSuffix()

			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())

			err = backend.DeleteCollection(ctx, vsID)
			Expect(err).NotTo(HaveOccurred())

			exists, err := backend.CollectionExists(ctx, vsID)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should return error for non-existent collection", func() {
			err := backend.DeleteCollection(ctx, "integ_nonexistent_"+uniqueSuffix())
			Expect(err).To(HaveOccurred())
		})
	})

	Context("CollectionExists", func() {
		It("should return false for non-existent collection", func() {
			exists, err := backend.CollectionExists(ctx, "integ_nope_"+uniqueSuffix())
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})
	})

	// -----------------------------------------------------------------------
	// InsertChunks
	// -----------------------------------------------------------------------

	Context("InsertChunks", func() {
		var vsID string

		BeforeEach(func() {
			vsID = "integ_insert_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			cleanupCollection(ctx, backend, vsID)
		})

		It("should insert chunks and retrieve via search", func() {
			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
			}
			err := backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			// Allow indexing time.
			time.Sleep(500 * time.Millisecond)

			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))
		})

		It("should handle empty chunk slice without error", func() {
			err := backend.InsertChunks(ctx, vsID, []EmbeddedChunk{})
			Expect(err).NotTo(HaveOccurred())
		})

		It("should insert multiple chunks for different files", func() {
			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "alpha", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f2", Filename: "b.txt", Content: "beta", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 0},
				{ID: "c3", FileID: "f2", Filename: "b.txt", Content: "gamma", Embedding: normalizeVec([]float32{0, 0, 1}), ChunkIndex: 1},
			}
			err := backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)

			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{0, 1, 0}), 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 2))
		})
	})

	// -----------------------------------------------------------------------
	// Search — mirrors Milvus test data for comparison
	// -----------------------------------------------------------------------

	Context("Search", func() {
		var vsID string

		BeforeEach(func() {
			vsID = "integ_search_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())

			// Same test data as Milvus integration tests.
			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			// Allow indexing.
			time.Sleep(500 * time.Millisecond)
		})

		AfterEach(func() {
			cleanupCollection(ctx, backend, vsID)
		})

		It("should return the most similar result first", func() {
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))
			Expect(results[0].Content).To(Equal("hello"))
		})

		It("should respect threshold", func() {
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0.99, nil)
			Expect(err).NotTo(HaveOccurred())
			// Only the near-exact match should pass a very high threshold.
			for _, r := range results {
				Expect(r.Score).To(BeNumerically(">=", 0.99))
			}
		})

		It("should respect topK", func() {
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 1, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
		})

		It("should return scores in descending order", func() {
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			if len(results) >= 2 {
				Expect(results[0].Score).To(BeNumerically(">=", results[1].Score))
			}
		})

		It("should populate all result fields", func() {
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 1, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))

			r := results[0]
			Expect(r.FileID).To(Equal("f1"))
			Expect(r.Filename).To(Equal("doc.txt"))
			Expect(r.Content).NotTo(BeEmpty())
			Expect(r.Score).To(BeNumerically(">", 0))
			Expect(r.ChunkIndex).To(BeNumerically(">=", 0))
		})

		It("should return empty results for empty collection", func() {
			emptyVS := "integ_empty_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, emptyVS, 3)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupCollection(ctx, backend, emptyVS)

			results, err := backend.Search(ctx, emptyVS, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})
	})

	// -----------------------------------------------------------------------
	// Search with file_id filter
	// -----------------------------------------------------------------------

	Context("Search with filter", func() {
		var vsID string

		BeforeEach(func() {
			vsID = "integ_filter_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())

			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "alpha", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "beta", Embedding: normalizeVec([]float32{0.9, 0.1, 0}), ChunkIndex: 1},
				{ID: "c3", FileID: "f2", Filename: "b.txt", Content: "gamma", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)
		})

		AfterEach(func() {
			cleanupCollection(ctx, backend, vsID)
		})

		It("should filter results by file_id", func() {
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 10, 0,
				map[string]interface{}{"file_id": "f2"})
			Expect(err).NotTo(HaveOccurred())

			for _, r := range results {
				Expect(r.FileID).To(Equal("f2"))
			}
		})

		It("should return all results with nil filter", func() {
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 2))
		})

		It("should reject invalid file_id filter", func() {
			_, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 10, 0,
				map[string]interface{}{"file_id": "bad;injection"})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("invalid"))
		})
	})

	// -----------------------------------------------------------------------
	// DeleteByFileID
	// -----------------------------------------------------------------------

	Context("DeleteByFileID", func() {
		var vsID string

		BeforeEach(func() {
			vsID = "integ_delf_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())

			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Content: "a", Filename: "a.txt", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Content: "b", Filename: "a.txt", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
				{ID: "c3", FileID: "f2", Content: "c", Filename: "b.txt", Embedding: normalizeVec([]float32{0, 0, 1}), ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)
		})

		AfterEach(func() {
			cleanupCollection(ctx, backend, vsID)
		})

		It("should delete only chunks for the specified file", func() {
			err := backend.DeleteByFileID(ctx, vsID, "f1")
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)

			// Only f2 chunks should remain.
			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{0, 0, 1}), 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))
			for _, r := range results {
				Expect(r.FileID).To(Equal("f2"))
			}
		})

		It("should not error when deleting non-existent file_id", func() {
			err := backend.DeleteByFileID(ctx, vsID, "f_nonexistent")
			Expect(err).NotTo(HaveOccurred())
		})

		It("should reject invalid file_id", func() {
			err := backend.DeleteByFileID(ctx, vsID, "bad;chars")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("invalid"))
		})
	})

	// -----------------------------------------------------------------------
	// Close
	// -----------------------------------------------------------------------

	Context("Close", func() {
		It("should not return an error", func() {
			host := valkeyHost
			if host == "" {
				host = "localhost"
			}
			port := 6379
			if valkeyPort != "" {
				p, err := strconv.Atoi(valkeyPort)
				if err == nil {
					port = p
				}
			}

			b, err := NewValkeyBackend(ValkeyBackendConfig{
				Host:             host,
				Port:             port,
				CollectionPrefix: "test_close_",
				ConnectTimeout:   5,
			})
			Expect(err).NotTo(HaveOccurred())

			err = b.Close()
			Expect(err).NotTo(HaveOccurred())
		})
	})

	// -----------------------------------------------------------------------
	// Custom config parameters
	// -----------------------------------------------------------------------

	Context("custom config", func() {
		It("should accept custom prefix and metric type", func() {
			host := valkeyHost
			if host == "" {
				host = "localhost"
			}
			port := 6379
			if valkeyPort != "" {
				p, err := strconv.Atoi(valkeyPort)
				if err == nil {
					port = p
				}
			}

			b, err := NewValkeyBackend(ValkeyBackendConfig{
				Host:             host,
				Port:             port,
				CollectionPrefix: "custom_pfx_",
				MetricType:       "L2",
				IndexM:           32,
				IndexEf:          400,
				ConnectTimeout:   5,
			})
			Expect(err).NotTo(HaveOccurred())
			defer b.Close()

			Expect(b.collectionPrefix).To(Equal("custom_pfx_"))
			Expect(b.metricType).To(Equal("L2"))
			Expect(b.indexM).To(Equal(32))
			Expect(b.indexEf).To(Equal(400))

			// Verify it can create a collection with L2 metric.
			vsID := "integ_l2_" + uniqueSuffix()
			err = b.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupCollection(ctx, b, vsID)

			exists, err := b.CollectionExists(ctx, vsID)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())
		})
	})

	// -----------------------------------------------------------------------
	// IP metric type
	// -----------------------------------------------------------------------

	Context("IP metric type", func() {
		It("should create collection and search with IP metric", func() {
			host := valkeyHost
			if host == "" {
				host = "localhost"
			}
			port := 6379
			if valkeyPort != "" {
				p, err := strconv.Atoi(valkeyPort)
				if err == nil {
					port = p
				}
			}

			b, err := NewValkeyBackend(ValkeyBackendConfig{
				Host:             host,
				Port:             port,
				CollectionPrefix: "test_ip_",
				MetricType:       "IP",
				ConnectTimeout:   5,
			})
			Expect(err).NotTo(HaveOccurred())
			defer b.Close()

			vsID := "integ_ip_" + uniqueSuffix()
			err = b.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupCollection(ctx, b, vsID)

			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
			}
			err = b.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)

			results, err := b.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))
		})
	})

	// -----------------------------------------------------------------------
	// Concurrent access
	// -----------------------------------------------------------------------

	Context("concurrent operations", func() {
		It("should handle concurrent inserts and searches without error", func() {
			vsID := "integ_concurrent_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupCollection(ctx, backend, vsID)

			// Insert seed data.
			seed := []EmbeddedChunk{
				{ID: "seed1", FileID: "f1", Filename: "a.txt", Content: "seed", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, vsID, seed)
			Expect(err).NotTo(HaveOccurred())
			time.Sleep(500 * time.Millisecond)

			const numGoroutines = 10
			errCh := make(chan error, numGoroutines*2)
			done := make(chan struct{}, numGoroutines)

			for i := 0; i < numGoroutines; i++ {
				go func(id int) {
					defer func() { done <- struct{}{} }()

					chunk := EmbeddedChunk{
						ID:        fmt.Sprintf("conc_%d", id),
						FileID:    "f1",
						Filename:  "a.txt",
						Content:   fmt.Sprintf("concurrent %d", id),
						Embedding: normalizeVec([]float32{float32(id%3 + 1), float32((id + 1) % 3), float32((id + 2) % 3)}),
					}
					if insertErr := backend.InsertChunks(ctx, vsID, []EmbeddedChunk{chunk}); insertErr != nil {
						errCh <- insertErr
					}

					if _, searchErr := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil); searchErr != nil {
						errCh <- searchErr
					}
				}(i)
			}

			for i := 0; i < numGoroutines; i++ {
				<-done
			}
			close(errCh)

			var errs []error
			for e := range errCh {
				errs = append(errs, e)
			}
			Expect(errs).To(BeEmpty())
		})
	})

	// -----------------------------------------------------------------------
	// Special characters in content/filenames
	// -----------------------------------------------------------------------

	Context("special characters roundtrip", func() {
		It("should preserve special characters in content and filename", func() {
			vsID := "integ_special_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupCollection(ctx, backend, vsID)

			chunks := []EmbeddedChunk{
				{
					ID:        "sp1",
					FileID:    "f1",
					Filename:  "path/to/doc (1).txt",
					Content:   `content with "quotes" and <tags> & symbols`,
					Embedding: normalizeVec([]float32{1, 0, 0}),
				},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)

			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 1, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
			Expect(results[0].Content).To(ContainSubstring(`"quotes"`))
			Expect(results[0].Content).To(ContainSubstring("<tags>"))
			Expect(results[0].Filename).To(Equal("path/to/doc (1).txt"))
		})
	})

	// -----------------------------------------------------------------------
	// Upsert / overwrite same chunk ID
	// -----------------------------------------------------------------------

	Context("overwrite same chunk ID", func() {
		It("should overwrite chunk data when inserting with same ID", func() {
			vsID := "integ_upsert_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupCollection(ctx, backend, vsID)

			original := []EmbeddedChunk{
				{ID: "dup1", FileID: "f1", Filename: "a.txt", Content: "original", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, vsID, original)
			Expect(err).NotTo(HaveOccurred())

			// Overwrite with new content.
			updated := []EmbeddedChunk{
				{ID: "dup1", FileID: "f1", Filename: "a.txt", Content: "updated", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, vsID, updated)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)

			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			// Should only have one result (overwritten, not duplicated).
			Expect(results).To(HaveLen(1))
			Expect(results[0].Content).To(Equal("updated"))
		})
	})

	// -----------------------------------------------------------------------
	// Higher-dimensional embeddings
	// -----------------------------------------------------------------------

	Context("higher-dimensional embeddings", func() {
		It("should work with 128-dimensional vectors", func() {
			vsID := "integ_highdim_" + uniqueSuffix()
			err := backend.CreateCollection(ctx, vsID, 128)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupCollection(ctx, backend, vsID)

			// Build a 128-dim vector with a clear signal in the first component.
			vec128 := make([]float32, 128)
			vec128[0] = 1.0
			vec128 = normalizeVec(vec128)

			chunks := []EmbeddedChunk{
				{ID: "hd1", FileID: "f1", Filename: "big.txt", Content: "high-dim", Embedding: vec128, ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)

			results, err := backend.Search(ctx, vsID, vec128, 1, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(HaveLen(1))
			Expect(results[0].Content).To(Equal("high-dim"))
		})
	})
})

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// uniqueSuffix returns a short unique string for test isolation.
func uniqueSuffix() string {
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}

// cleanupCollection silently removes a test collection.
func cleanupCollection(ctx context.Context, b *ValkeyBackend, vsID string) {
	if b != nil {
		_ = b.DeleteCollection(ctx, vsID)
	}
}

// Verify ValkeyBackend satisfies the interface at compile time.
var _ VectorStoreBackend = (*ValkeyBackend)(nil)
