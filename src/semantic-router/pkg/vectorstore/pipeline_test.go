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
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// mockEmbedder provides a simple mock for testing.
type mockEmbedder struct {
	dim    int
	err    error
	called int
}

func (m *mockEmbedder) Embed(_ string) ([]float32, error) {
	m.called++
	if m.err != nil {
		return nil, m.err
	}
	emb := make([]float32, m.dim)
	for i := range emb {
		emb[i] = 0.1
	}
	return emb, nil
}

func (m *mockEmbedder) Dimension() int {
	return m.dim
}

type ingestionPipelineTestFixture struct {
	backend  *MemoryBackend
	store    *FileStore
	mgr      *Manager
	embedder *mockEmbedder
	pipeline *IngestionPipeline
	tempDir  string
	ctx      context.Context
}

func newIngestionPipelineTestFixture() *ingestionPipelineTestFixture {
	GinkgoHelper()

	tempDir, err := os.MkdirTemp("", "pipeline-test-*")
	Expect(err).NotTo(HaveOccurred())

	backend := NewMemoryBackend(MemoryBackendConfig{})
	store, err := NewFileStore(tempDir, NewMemoryMetadataRegistry())
	Expect(err).NotTo(HaveOccurred())

	mgr := NewManager(backend, NewMemoryMetadataRegistry(), 3, BackendTypeMemory)
	embedder := &mockEmbedder{dim: 3}
	pipeline := NewIngestionPipeline(backend, store, mgr, embedder, PipelineConfig{
		Workers:   1,
		QueueSize: 10,
	})
	pipeline.Start()

	return &ingestionPipelineTestFixture{
		backend:  backend,
		store:    store,
		mgr:      mgr,
		embedder: embedder,
		pipeline: pipeline,
		tempDir:  tempDir,
		ctx:      context.Background(),
	}
}

func (f *ingestionPipelineTestFixture) cleanup() {
	GinkgoHelper()

	f.pipeline.Stop()
	Expect(os.RemoveAll(f.tempDir)).To(Succeed())
}

func (f *ingestionPipelineTestFixture) createStore(name string) *VectorStore {
	GinkgoHelper()

	vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: name})
	Expect(err).NotTo(HaveOccurred())
	return vs
}

func (f *ingestionPipelineTestFixture) saveFile(filename, content string) *FileRecord {
	GinkgoHelper()

	record, err := f.store.Save(filename, []byte(content), "assistants")
	Expect(err).NotTo(HaveOccurred())
	return record
}

func expectPipelineStatus(pipeline *IngestionPipeline, fileID, expected string) {
	GinkgoHelper()

	Eventually(func() string {
		status, err := pipeline.GetFileStatus(fileID)
		if err != nil {
			return ""
		}
		return status.Status
	}, 5*time.Second, 50*time.Millisecond).Should(Equal(expected))
}

var _ = Describe("IngestionPipeline attach processing", func() {
	var f *ingestionPipelineTestFixture

	BeforeEach(func() {
		f = newIngestionPipelineTestFixture()
	})

	AfterEach(func() {
		f.cleanup()
	})

	It("should process a text file end-to-end", func() {
		vs := f.createStore("pipeline-test")
		record := f.saveFile("test.txt", "Hello world.\n\nSecond paragraph.")

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(vsf.Status).To(Equal("in_progress"))

		expectPipelineStatus(f.pipeline, vsf.ID, "completed")

		results, err := f.backend.Search(f.ctx, vs.ID, []float32{0.1, 0.1, 0.1}, 10, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(results)).To(BeNumerically(">=", 1))

		updated, err := f.mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Completed).To(Equal(1))
		Expect(updated.FileCounts.InProgress).To(Equal(0))
	})

	It("should handle markdown files", func() {
		vs := f.createStore("md-test")
		record := f.saveFile("doc.md", "# Title\n\nFirst section.\n\n## Subtitle\n\nSecond section.")

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		expectPipelineStatus(f.pipeline, vsf.ID, "completed")
	})

	It("should use static chunking strategy", func() {
		vs := f.createStore("static-test")
		record := f.saveFile("alpha.txt", "abcdefghijklmnopqrstuvwxyz")
		strategy := &ChunkingStrategy{
			Type:   "static",
			Static: &StaticChunkConfig{MaxChunkSizeTokens: 10, ChunkOverlapTokens: 0},
		}

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, strategy)
		Expect(err).NotTo(HaveOccurred())
		expectPipelineStatus(f.pipeline, vsf.ID, "completed")
		Expect(f.embedder.called).To(BeNumerically(">=", 3))
	})
})

var _ = Describe("IngestionPipeline attach failures", func() {
	var f *ingestionPipelineTestFixture

	BeforeEach(func() {
		f = newIngestionPipelineTestFixture()
	})

	AfterEach(func() {
		f.cleanup()
	})

	It("should return error for non-existent file", func() {
		vs := f.createStore("err-test")

		_, err := f.pipeline.AttachFile(vs.ID, "file_nonexistent", nil)
		Expect(err).To(HaveOccurred())
	})

	It("should return error for non-existent vector store", func() {
		record := f.saveFile("test.txt", "data")

		_, err := f.pipeline.AttachFile("vs_nonexistent", record.ID, nil)
		Expect(err).To(HaveOccurred())
	})

	It("should mark failed on embedding error", func() {
		vs := f.createStore("embed-fail")
		record := f.saveFile("test.txt", "content")
		f.embedder.err = fmt.Errorf("embedding model unavailable")

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		expectPipelineStatus(f.pipeline, vsf.ID, "failed")

		status, err := f.pipeline.GetFileStatus(vsf.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(status.LastError).NotTo(BeNil())
		Expect(status.LastError.Code).To(Equal("embedding_error"))
	})

	It("should mark failed on empty file", func() {
		vs := f.createStore("empty-test")
		record := f.saveFile("empty.txt", "   ")

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		expectPipelineStatus(f.pipeline, vsf.ID, "failed")

		status, err := f.pipeline.GetFileStatus(vsf.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(status.LastError.Code).To(Equal("empty_content"))
	})
})

var _ = Describe("IngestionPipeline status, detach, and lifecycle", func() {
	var f *ingestionPipelineTestFixture

	BeforeEach(func() {
		f = newIngestionPipelineTestFixture()
	})

	AfterEach(func() {
		f.cleanup()
	})

	It("should return files for a specific vector store", func() {
		vs := f.createStore("list-test")
		record := f.saveFile("a.txt", "content a")

		_, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() int {
			return len(f.pipeline.ListFileStatuses(vs.ID))
		}, 5*time.Second, 50*time.Millisecond).Should(Equal(1))
	})

	It("should return defensive status copies", func() {
		vs := f.createStore("status-copy")
		record := f.saveFile("copy.txt", "content")

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		status, err := f.pipeline.GetFileStatus(vsf.ID)
		Expect(err).NotTo(HaveOccurred())
		status.Status = "mutated"
		status.ChunkingStrategy = &ChunkingStrategy{Type: "mutated"}

		listed := f.pipeline.ListFileStatuses(vs.ID)
		Expect(listed).To(HaveLen(1))
		listed[0].Status = "mutated-again"

		reloaded, err := f.pipeline.GetFileStatus(vsf.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(reloaded.Status).NotTo(ContainSubstring("mutated"))
		Expect(reloaded.ChunkingStrategy).To(BeNil())
	})

	It("should remove chunks when detaching a file", func() {
		vs := f.createStore("detach-test")
		record := f.saveFile("test.txt", "Some content here")

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		expectPipelineStatus(f.pipeline, vsf.ID, "completed")

		err = f.pipeline.DetachFile(f.ctx, vs.ID, vsf.ID)
		Expect(err).NotTo(HaveOccurred())

		_, err = f.pipeline.GetFileStatus(vsf.ID)
		Expect(err).To(HaveOccurred())

		results, err := f.backend.Search(f.ctx, vs.ID, []float32{0.1, 0.1, 0.1}, 10, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(results).To(BeEmpty())
	})

	It("should return empty statuses and errors for unknown records", func() {
		files := f.pipeline.ListFileStatuses("vs_unknown")
		Expect(files).To(BeEmpty())

		err := f.pipeline.DetachFile(f.ctx, "vs_x", "vsf_nonexistent")
		Expect(err).To(HaveOccurred())
	})

	It("should start and stop idempotently", func() {
		f.pipeline.Start()
		f.pipeline.Stop()
		f.pipeline.Stop()
	})
})
