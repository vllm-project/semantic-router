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
	"strings"
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// insertSpyBackend wraps a backend and records every InsertChunks call so tests
// can assert that ingestion is actually batched (call count and per-call size).
type insertSpyBackend struct {
	VectorStoreBackend
	mu          sync.Mutex
	insertCalls int
	insertSizes []int
}

func newInsertSpyBackend(inner VectorStoreBackend) *insertSpyBackend {
	return &insertSpyBackend{VectorStoreBackend: inner}
}

func (s *insertSpyBackend) InsertChunks(ctx context.Context, vectorStoreID string, chunks []EmbeddedChunk) error {
	s.mu.Lock()
	s.insertCalls++
	s.insertSizes = append(s.insertSizes, len(chunks))
	s.mu.Unlock()
	return s.VectorStoreBackend.InsertChunks(ctx, vectorStoreID, chunks)
}

func (s *insertSpyBackend) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.insertCalls
}

func (s *insertSpyBackend) maxBatch() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := 0
	for _, n := range s.insertSizes {
		if n > m {
			m = n
		}
	}
	return m
}

// indexFailEmbedder embeds normally but returns an error once it reaches a
// specific chunk index, letting a test fail ingestion partway through a
// multi-batch file.
type indexFailEmbedder struct {
	dim      int
	failAt   int // chunk index at which Embed returns an error (-1 = never)
	mu       sync.Mutex
	seen     int
	seenList []string
}

func (e *indexFailEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.mu.Lock()
	idx := e.seen
	e.seen++
	e.seenList = append(e.seenList, text)
	e.mu.Unlock()

	if e.failAt >= 0 && idx == e.failAt {
		return nil, fmt.Errorf("synthetic embed failure at chunk %d", idx)
	}
	emb := make([]float32, e.dim)
	for i := range emb {
		emb[i] = 0.1
	}
	return emb, nil
}

func (e *indexFailEmbedder) Dimension() int { return e.dim }

// batchFixture builds a pipeline with a spy backend and a configurable batch
// size and embedder, so batching behavior can be observed directly.
type batchFixture struct {
	backend  *insertSpyBackend
	mem      *MemoryBackend
	store    *FileStore
	mgr      *Manager
	pipeline *IngestionPipeline
	tempDir  string
	ctx      context.Context
}

func newBatchFixture(embedder Embedder, batchSize int) *batchFixture {
	GinkgoHelper()

	tempDir, err := os.MkdirTemp("", "pipeline-batch-test-*")
	Expect(err).NotTo(HaveOccurred())

	mem := NewMemoryBackend(MemoryBackendConfig{})
	spy := newInsertSpyBackend(mem)
	store, err := NewFileStore(tempDir, NewMemoryMetadataRegistry())
	Expect(err).NotTo(HaveOccurred())

	mgr := NewManager(spy, NewMemoryMetadataRegistry(), 3, BackendTypeMemory)
	pipeline := NewIngestionPipeline(spy, store, mgr, embedder, PipelineConfig{
		Workers:   1,
		QueueSize: 10,
		BatchSize: batchSize,
	})
	pipeline.Start()

	return &batchFixture{
		backend:  spy,
		mem:      mem,
		store:    store,
		mgr:      mgr,
		pipeline: pipeline,
		tempDir:  tempDir,
		ctx:      context.Background(),
	}
}

func (f *batchFixture) cleanup() {
	_ = f.pipeline.Stop(context.Background())
	_ = os.RemoveAll(f.tempDir)
}

func (f *batchFixture) createStore(name string) *VectorStore {
	GinkgoHelper()
	vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: name})
	Expect(err).NotTo(HaveOccurred())
	return vs
}

// attach saves an n-chunk document and attaches it, returning the store and the
// vector-store-file ID so each test can assert on the resulting status.
func (f *batchFixture) attach(storeName string, numChunks int) (*VectorStore, string) {
	GinkgoHelper()
	vs := f.createStore(storeName)
	record, err := f.store.Save("doc.txt", []byte(nParagraphContent(numChunks)), "assistants")
	Expect(err).NotTo(HaveOccurred())
	vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
	Expect(err).NotTo(HaveOccurred())
	return vs, vsf.ID
}

// nParagraphContent returns text that ChunkText (auto strategy) splits into
// exactly n chunks: n distinct, heading-free paragraphs separated by blank
// lines, each well under DefaultMaxChunkSize so it maps to a single chunk.
func nParagraphContent(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteString("\n\n")
		}
		fmt.Fprintf(&b, "paragraph number %d with some filler words", i)
	}
	return b.String()
}

func countChunks(f *batchFixture, vsID string) int {
	GinkgoHelper()
	results, err := f.mem.Search(f.ctx, vsID, []float32{0.1, 0.1, 0.1}, 100000, 0, nil)
	Expect(err).NotTo(HaveOccurred())
	return len(results)
}

var _ = Describe("IngestionPipeline bounded batching", func() {
	var f *batchFixture

	AfterEach(func() {
		if f != nil {
			f.cleanup()
			f = nil
		}
	})

	DescribeTable("stores every chunk across batch boundaries",
		func(numChunks, batchSize int) {
			f = newBatchFixture(&mockEmbedder{dim: 3}, batchSize)
			vs, vsfID := f.attach("batch-correctness", numChunks)
			expectPipelineStatus(f.pipeline, vsfID, "completed")

			// All chunks landed in the backend.
			Expect(countChunks(f, vs.ID)).To(Equal(numChunks))

			// Insert happened in bounded batches: expected number of calls and no
			// call larger than the configured batch size.
			expectedCalls := (numChunks + batchSize - 1) / batchSize
			Expect(f.backend.calls()).To(Equal(expectedCalls))
			Expect(f.backend.maxBatch()).To(BeNumerically("<=", batchSize))
		},
		Entry("single chunk, batch 4", 1, 4),
		Entry("fewer than one batch", 3, 4),
		Entry("exact multiple of batch", 8, 4),
		Entry("one over a batch boundary", 5, 4),
		Entry("batch size of one", 4, 1),
		Entry("many chunks, small batch", 13, 5),
	)

	It("defaults to batch size 64 when unset", func() {
		const numChunks = 65 // 65 chunks -> 2 inserts at default batch 64
		f = newBatchFixture(&mockEmbedder{dim: 3}, 0)
		vs, vsfID := f.attach("batch-default", numChunks)
		expectPipelineStatus(f.pipeline, vsfID, "completed")

		Expect(countChunks(f, vs.ID)).To(Equal(numChunks))
		Expect(f.backend.calls()).To(Equal(2))
		Expect(f.backend.maxBatch()).To(BeNumerically("<=", 64))
	})

	It("removes already-inserted chunks when a later batch fails", func() {
		const numChunks = 10
		const batchSize = 3
		// Fail at chunk index 7 -> batches [0,3) and [3,6) inserted, then batch
		// [6,9) fails while embedding chunk 7.
		f = newBatchFixture(&indexFailEmbedder{dim: 3, failAt: 7}, batchSize)
		vs, vsfID := f.attach("batch-failcleanup", numChunks)
		expectPipelineStatus(f.pipeline, vsfID, "failed")

		status, err := f.pipeline.GetFileStatus(vsfID)
		Expect(err).NotTo(HaveOccurred())
		Expect(status.LastError).NotTo(BeNil())
		Expect(status.LastError.Code).To(Equal("embedding_error"))

		// Partial chunks from the first two batches must have been cleaned up:
		// a failed file leaves nothing searchable.
		Expect(countChunks(f, vs.ID)).To(Equal(0))

		updated, err := f.mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Failed).To(Equal(1))
		Expect(updated.FileCounts.Completed).To(Equal(0))
		Expect(updated.FileCounts.InProgress).To(Equal(0))
	})

	It("does not attempt cleanup when the first batch fails", func() {
		const numChunks = 6
		const batchSize = 3
		// Fail at chunk index 0 -> no batch ever inserted; nothing to clean up.
		f = newBatchFixture(&indexFailEmbedder{dim: 3, failAt: 0}, batchSize)
		vs, vsfID := f.attach("batch-firstfail", numChunks)
		expectPipelineStatus(f.pipeline, vsfID, "failed")

		Expect(f.backend.calls()).To(Equal(0))
		Expect(countChunks(f, vs.ID)).To(Equal(0))
	})
})

var _ = Describe("IngestionPipeline batching config default", func() {
	It("applies the default batch size through NewIngestionPipeline", func() {
		tempDir, err := os.MkdirTemp("", "pipeline-cfg-*")
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = os.RemoveAll(tempDir) }()

		mem := NewMemoryBackend(MemoryBackendConfig{})
		store, err := NewFileStore(tempDir, NewMemoryMetadataRegistry())
		Expect(err).NotTo(HaveOccurred())
		mgr := NewManager(mem, NewMemoryMetadataRegistry(), 3, BackendTypeMemory)

		p := NewIngestionPipeline(mem, store, mgr, &mockEmbedder{dim: 3}, PipelineConfig{})
		Expect(p.batchSize).To(Equal(defaultBatchSize))

		p2 := NewIngestionPipeline(mem, store, mgr, &mockEmbedder{dim: 3}, PipelineConfig{BatchSize: 7})
		Expect(p2.batchSize).To(Equal(7))
	})
})
