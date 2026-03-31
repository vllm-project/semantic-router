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
	"fmt"
	"os"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func uniqueSuffix() string { return strconv.FormatInt(time.Now().UnixNano(), 36) }

func cleanupCollection(ctx context.Context, b *ValkeyBackend, vsID string) {
	if b != nil {
		_ = b.DeleteCollection(ctx, vsID)
	}
}

func newIntegBackend() (*ValkeyBackend, context.Context) {
	if os.Getenv("SKIP_VALKEY_TESTS") != "false" {
		Skip("Skipping Valkey tests (set SKIP_VALKEY_TESTS=false to enable)")
	}
	host := os.Getenv("VALKEY_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if p := os.Getenv("VALKEY_PORT"); p != "" {
		if v, err := strconv.Atoi(p); err == nil {
			port = v
		}
	}
	b, err := NewValkeyBackend(ValkeyBackendConfig{
		Host: host, Port: port, CollectionPrefix: "test_vs_", ConnectTimeout: 5,
	})
	Expect(err).NotTo(HaveOccurred())
	return b, context.Background()
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

var _ = Describe("ValkeyBackend integ collection ops", func() {
	var (
		backend *ValkeyBackend
		ctx     context.Context
	)
	BeforeEach(func() { backend, ctx = newIntegBackend() })
	AfterEach(func() {
		if backend != nil {
			_ = backend.Close()
		}
	})

	It("create and verify", func() {
		vsID := "integ_create_" + uniqueSuffix()
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		exists, err := backend.CollectionExists(ctx, vsID)
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeTrue())
	})
	It("duplicate collection error", func() {
		vsID := "integ_dup_" + uniqueSuffix()
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		Expect(backend.CreateCollection(ctx, vsID, 3)).To(HaveOccurred())
	})
	It("different dimensions", func() {
		a, b := "integ_dim128_"+uniqueSuffix(), "integ_dim768_"+uniqueSuffix()
		defer cleanupCollection(ctx, backend, a)
		defer cleanupCollection(ctx, backend, b)
		Expect(backend.CreateCollection(ctx, a, 128)).NotTo(HaveOccurred())
		Expect(backend.CreateCollection(ctx, b, 768)).NotTo(HaveOccurred())
	})
	It("delete collection", func() {
		vsID := "integ_del_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		Expect(backend.DeleteCollection(ctx, vsID)).NotTo(HaveOccurred())
		exists, err := backend.CollectionExists(ctx, vsID)
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeFalse())
	})
	It("delete non-existent errors", func() {
		Expect(backend.DeleteCollection(ctx, "integ_nonexistent_"+uniqueSuffix())).To(HaveOccurred())
	})
	It("exists returns false for missing", func() {
		exists, err := backend.CollectionExists(ctx, "integ_nope_"+uniqueSuffix())
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeFalse())
	})
	It("close", func() {
		b, _ := newIntegBackend()
		Expect(b.Close()).NotTo(HaveOccurred())
	})
})

var _ = Describe("ValkeyBackend integ insert and delete", func() {
	var (
		backend *ValkeyBackend
		ctx     context.Context
	)
	BeforeEach(func() { backend, ctx = newIntegBackend() })
	AfterEach(func() {
		if backend != nil {
			_ = backend.Close()
		}
	})

	It("insert and search", func() {
		vsID := "integ_insert_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		chunks := []EmbeddedChunk{
			{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0})},
			{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
		}
		Expect(backend.InsertChunks(ctx, vsID, chunks)).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(results)).To(BeNumerically(">=", 1))
	})
	It("empty insert", func() {
		vsID := "integ_empty_insert_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.InsertChunks(ctx, vsID, []EmbeddedChunk{})).NotTo(HaveOccurred())
	})
	It("delete by file_id", func() {
		vsID := "integ_delf_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		chunks := []EmbeddedChunk{
			{ID: "c1", FileID: "f1", Content: "a", Filename: "a.txt", Embedding: normalizeVec([]float32{1, 0, 0})},
			{ID: "c2", FileID: "f1", Content: "b", Filename: "a.txt", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
			{ID: "c3", FileID: "f2", Content: "c", Filename: "b.txt", Embedding: normalizeVec([]float32{0, 0, 1})},
		}
		Expect(backend.InsertChunks(ctx, vsID, chunks)).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		Expect(backend.DeleteByFileID(ctx, vsID, "f1")).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		results, err := backend.Search(ctx, vsID, normalizeVec([]float32{0, 0, 1}), 10, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		for _, r := range results {
			Expect(r.FileID).To(Equal("f2"))
		}
	})
	It("delete non-existent file_id ok", func() {
		vsID := "integ_delf_noop_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.DeleteByFileID(ctx, vsID, "f_nonexistent")).NotTo(HaveOccurred())
	})
	It("reject invalid file_id", func() {
		vsID := "integ_delf_bad_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.DeleteByFileID(ctx, vsID, "bad;chars")).To(HaveOccurred())
	})
	// Verify that DeleteByFileID pages through results correctly by inserting
	// more chunks than the page size (1000).  This exercises the second-page
	// path where the loop fetches a partial page and then an empty page.
	It("paginated delete removes all chunks across multiple pages", func() {
		const totalChunks = 1100 // > pageSize (1000) to force a second page
		vsID := "integ_paged_del_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)

		chunks := make([]EmbeddedChunk, totalChunks)
		for i := 0; i < totalChunks; i++ {
			chunks[i] = EmbeddedChunk{
				ID:       fmt.Sprintf("pc%d", i),
				FileID:   "bigfile",
				Filename: "big.txt",
				Content:  fmt.Sprintf("chunk %d", i),
				// All vectors point in the same direction so insertion is fast.
				Embedding: normalizeVec([]float32{1, 0, 0}),
			}
		}
		Expect(backend.InsertChunks(ctx, vsID, chunks)).NotTo(HaveOccurred())
		time.Sleep(2 * time.Second) // allow index to catch up

		Expect(backend.DeleteByFileID(ctx, vsID, "bigfile")).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)

		results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), totalChunks, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(results).To(BeEmpty())
	})
})

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

var _ = Describe("ValkeyBackend integ Search", func() {
	var (
		backend *ValkeyBackend
		ctx     context.Context
		vsID    string
	)
	BeforeEach(func() {
		backend, ctx = newIntegBackend()
		vsID = "integ_search_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		chunks := []EmbeddedChunk{
			{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0})},
			{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
		}
		Expect(backend.InsertChunks(ctx, vsID, chunks)).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
	})
	AfterEach(func() {
		cleanupCollection(ctx, backend, vsID)
		if backend != nil {
			_ = backend.Close()
		}
	})

	It("most similar first", func() {
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(r[0].Content).To(Equal("hello"))
	})
	It("threshold", func() {
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0.99, nil)
		Expect(err).NotTo(HaveOccurred())
		for _, x := range r {
			Expect(x.Score).To(BeNumerically(">=", 0.99))
		}
	})
	It("topK", func() {
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 1, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(HaveLen(1))
	})
	It("descending order", func() {
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		if len(r) >= 2 {
			Expect(r[0].Score).To(BeNumerically(">=", r[1].Score))
		}
	})
	It("all fields populated including ChunkIndex", func() {
		// Search for the chunk closest to {0,1,0} — that is c2 with ChunkIndex=1.
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{0, 1, 0}), 1, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(HaveLen(1))
		Expect(r[0].FileID).To(Equal("f1"))
		Expect(r[0].Filename).To(Equal("doc.txt"))
		Expect(r[0].Content).NotTo(BeEmpty())
		Expect(r[0].Score).To(BeNumerically(">", 0))
		Expect(r[0].ChunkIndex).To(Equal(1)) // verifies chunk_index is in the RETURN list
	})
	It("empty collection", func() {
		emptyVS := "integ_empty_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, emptyVS, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, emptyVS)
		r, err := backend.Search(ctx, emptyVS, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(BeEmpty())
	})
})

var _ = Describe("ValkeyBackend integ Search filter", func() {
	var (
		backend *ValkeyBackend
		ctx     context.Context
		vsID    string
	)
	BeforeEach(func() {
		backend, ctx = newIntegBackend()
		vsID = "integ_filter_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		chunks := []EmbeddedChunk{
			{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "alpha", Embedding: normalizeVec([]float32{1, 0, 0})},
			{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "beta", Embedding: normalizeVec([]float32{0.9, 0.1, 0}), ChunkIndex: 1},
			{ID: "c3", FileID: "f2", Filename: "b.txt", Content: "gamma", Embedding: normalizeVec([]float32{0, 1, 0})},
		}
		Expect(backend.InsertChunks(ctx, vsID, chunks)).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
	})
	AfterEach(func() {
		cleanupCollection(ctx, backend, vsID)
		if backend != nil {
			_ = backend.Close()
		}
	})

	It("filter by file_id", func() {
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 10, 0, map[string]interface{}{"file_id": "f2"})
		Expect(err).NotTo(HaveOccurred())
		for _, x := range r {
			Expect(x.FileID).To(Equal("f2"))
		}
	})
	It("nil filter returns all", func() {
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 10, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(r)).To(BeNumerically(">=", 2))
	})
	It("reject invalid file_id", func() {
		_, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 10, 0, map[string]interface{}{"file_id": "bad;injection"})
		Expect(err).To(HaveOccurred())
	})
})

// ---------------------------------------------------------------------------
// Advanced
// ---------------------------------------------------------------------------

var _ = Describe("ValkeyBackend integ config variants", func() {
	var ctx context.Context
	BeforeEach(func() {
		if os.Getenv("SKIP_VALKEY_TESTS") != "false" {
			Skip("Skipping Valkey tests (set SKIP_VALKEY_TESTS=false to enable)")
		}
		ctx = context.Background()
	})

	It("custom prefix and L2 metric", func() {
		host := os.Getenv("VALKEY_HOST")
		if host == "" {
			host = "localhost"
		}
		port := 6379
		if p := os.Getenv("VALKEY_PORT"); p != "" {
			if v, err := strconv.Atoi(p); err == nil {
				port = v
			}
		}
		b, err := NewValkeyBackend(ValkeyBackendConfig{
			Host: host, Port: port, CollectionPrefix: "custom_pfx_",
			MetricType: "L2", IndexM: 32, IndexEf: 400, ConnectTimeout: 5,
		})
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = b.Close() }()
		Expect(b.metricType).To(Equal("L2"))
		vsID := "integ_l2_" + uniqueSuffix()
		Expect(b.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, b, vsID)
		exists, err := b.CollectionExists(ctx, vsID)
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeTrue())
	})

	It("IP metric", func() {
		host := os.Getenv("VALKEY_HOST")
		if host == "" {
			host = "localhost"
		}
		port := 6379
		if p := os.Getenv("VALKEY_PORT"); p != "" {
			if v, err := strconv.Atoi(p); err == nil {
				port = v
			}
		}
		b, err := NewValkeyBackend(ValkeyBackendConfig{
			Host: host, Port: port, CollectionPrefix: "test_ip_",
			MetricType: "IP", ConnectTimeout: 5,
		})
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = b.Close() }()
		vsID := "integ_ip_" + uniqueSuffix()
		Expect(b.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, b, vsID)
		Expect(b.InsertChunks(ctx, vsID, []EmbeddedChunk{
			{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0})},
		})).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		r, err := b.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(r)).To(BeNumerically(">=", 1))
	})
})

var _ = Describe("ValkeyBackend integ edge cases", func() {
	var (
		backend *ValkeyBackend
		ctx     context.Context
	)
	BeforeEach(func() { backend, ctx = newIntegBackend() })
	AfterEach(func() {
		if backend != nil {
			_ = backend.Close()
		}
	})

	It("concurrent operations", func() {
		vsID := "integ_concurrent_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.InsertChunks(ctx, vsID, []EmbeddedChunk{
			{ID: "seed1", FileID: "f1", Filename: "a.txt", Content: "seed", Embedding: normalizeVec([]float32{1, 0, 0})},
		})).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		const n = 10
		errCh := make(chan error, n*2)
		done := make(chan struct{}, n)
		for i := 0; i < n; i++ {
			go func(id int) {
				defer func() { done <- struct{}{} }()
				chunk := EmbeddedChunk{
					ID: fmt.Sprintf("conc_%d", id), FileID: "f1", Filename: "a.txt",
					Content:   fmt.Sprintf("concurrent %d", id),
					Embedding: normalizeVec([]float32{float32(id%3 + 1), float32((id + 1) % 3), float32((id + 2) % 3)}),
				}
				if e := backend.InsertChunks(ctx, vsID, []EmbeddedChunk{chunk}); e != nil {
					errCh <- e
				}
				if _, e := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil); e != nil {
					errCh <- e
				}
			}(i)
		}
		for i := 0; i < n; i++ {
			<-done
		}
		close(errCh)
		var errs []error
		for e := range errCh {
			errs = append(errs, e)
		}
		Expect(errs).To(BeEmpty())
	})

	It("special characters roundtrip", func() {
		vsID := "integ_special_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.InsertChunks(ctx, vsID, []EmbeddedChunk{{
			ID: "sp1", FileID: "f1", Filename: "path/to/doc (1).txt",
			Content: `content with "quotes" and <tags> & symbols`, Embedding: normalizeVec([]float32{1, 0, 0}),
		}})).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 1, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(HaveLen(1))
		Expect(r[0].Content).To(ContainSubstring(`"quotes"`))
		Expect(r[0].Filename).To(Equal("path/to/doc (1).txt"))
	})

	It("upsert overwrites", func() {
		vsID := "integ_upsert_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 3)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		Expect(backend.InsertChunks(ctx, vsID, []EmbeddedChunk{
			{ID: "dup1", FileID: "f1", Filename: "a.txt", Content: "original", Embedding: normalizeVec([]float32{1, 0, 0})},
		})).NotTo(HaveOccurred())
		Expect(backend.InsertChunks(ctx, vsID, []EmbeddedChunk{
			{ID: "dup1", FileID: "f1", Filename: "a.txt", Content: "updated", Embedding: normalizeVec([]float32{1, 0, 0})},
		})).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		r, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(HaveLen(1))
		Expect(r[0].Content).To(Equal("updated"))
	})

	It("128-dimensional vectors", func() {
		vsID := "integ_highdim_" + uniqueSuffix()
		Expect(backend.CreateCollection(ctx, vsID, 128)).NotTo(HaveOccurred())
		defer cleanupCollection(ctx, backend, vsID)
		vec := make([]float32, 128)
		vec[0] = 1.0
		vec = normalizeVec(vec)
		Expect(backend.InsertChunks(ctx, vsID, []EmbeddedChunk{
			{ID: "hd1", FileID: "f1", Filename: "big.txt", Content: "high-dim", Embedding: vec},
		})).NotTo(HaveOccurred())
		time.Sleep(500 * time.Millisecond)
		r, err := backend.Search(ctx, vsID, vec, 1, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(HaveLen(1))
		Expect(r[0].Content).To(Equal("high-dim"))
	})
})
