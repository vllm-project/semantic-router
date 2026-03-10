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

var _ = Describe("IngestionPipeline", func() {
	var (
		backend  *MemoryBackend
		store    *FileStore
		mgr      *Manager
		embedder *mockEmbedder
		pipeline *IngestionPipeline
		tempDir  string
		ctx      context.Context
	)

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "pipeline-test-*")
		Expect(err).NotTo(HaveOccurred())

		backend = NewMemoryBackend(MemoryBackendConfig{})
		store, err = NewFileStore(tempDir)
		Expect(err).NotTo(HaveOccurred())

		mgr = NewManager(backend, 3, BackendTypeMemory)
		embedder = &mockEmbedder{dim: 3}
		pipeline = NewIngestionPipeline(backend, store, mgr, embedder, PipelineConfig{
			Workers:   1,
			QueueSize: 10,
		})
		pipeline.Start()
		ctx = context.Background()
	})

	AfterEach(func() {
		pipeline.Stop()
		os.RemoveAll(tempDir)
	})

	Context("AttachFile", func() {
		It("should process a text file end-to-end", func() {
			// Create vector store.
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "pipeline-test"})
			Expect(err).NotTo(HaveOccurred())

			// Upload file.
			record, err := store.Save("test.txt", []byte("Hello world.\n\nSecond paragraph."), "assistants")
			Expect(err).NotTo(HaveOccurred())

			// Attach file.
			vsf, err := pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(vsf.Status).To(Equal("in_progress"))

			// Wait for processing.
			Eventually(func() string {
				status, _ := pipeline.GetFileStatus(vsf.ID)
				return status.Status
			}, 5*time.Second, 50*time.Millisecond).Should(Equal("completed"))

			// Verify chunks were inserted.
			results, err := backend.Search(ctx, vs.ID, []float32{0.1, 0.1, 0.1}, 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))

			// Verify file counts.
			updated, err := mgr.GetStore(vs.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(updated.FileCounts.Completed).To(Equal(1))
			Expect(updated.FileCounts.InProgress).To(Equal(0))
		})

		It("should handle markdown files", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "md-test"})
			Expect(err).NotTo(HaveOccurred())

			content := "# Title\n\nFirst section.\n\n## Subtitle\n\nSecond section."
			record, err := store.Save("doc.md", []byte(content), "assistants")
			Expect(err).NotTo(HaveOccurred())

			vsf, err := pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				status, _ := pipeline.GetFileStatus(vsf.ID)
				return status.Status
			}, 5*time.Second, 50*time.Millisecond).Should(Equal("completed"))
		})

		It("should use static chunking strategy", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "static-test"})
			Expect(err).NotTo(HaveOccurred())

			content := "abcdefghijklmnopqrstuvwxyz"
			record, err := store.Save("alpha.txt", []byte(content), "assistants")
			Expect(err).NotTo(HaveOccurred())

			strategy := &ChunkingStrategy{
				Type:   "static",
				Static: &StaticChunkConfig{MaxChunkSizeTokens: 10, ChunkOverlapTokens: 0},
			}
			vsf, err := pipeline.AttachFile(vs.ID, record.ID, strategy)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				status, _ := pipeline.GetFileStatus(vsf.ID)
				return status.Status
			}, 5*time.Second, 50*time.Millisecond).Should(Equal("completed"))

			// Should produce 3 chunks: [a-j], [k-t], [u-z]
			Expect(embedder.called).To(BeNumerically(">=", 3))
		})

		It("should return error for non-existent file", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "err-test"})
			Expect(err).NotTo(HaveOccurred())

			_, err = pipeline.AttachFile(vs.ID, "file_nonexistent", nil)
			Expect(err).To(HaveOccurred())
		})

		It("should return error for non-existent vector store", func() {
			record, err := store.Save("test.txt", []byte("data"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			_, err = pipeline.AttachFile("vs_nonexistent", record.ID, nil)
			Expect(err).To(HaveOccurred())
		})

		It("should mark failed on embedding error", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "embed-fail"})
			Expect(err).NotTo(HaveOccurred())

			record, err := store.Save("test.txt", []byte("content"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			// Make embedder fail.
			embedder.err = fmt.Errorf("embedding model unavailable")

			vsf, err := pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				status, _ := pipeline.GetFileStatus(vsf.ID)
				return status.Status
			}, 5*time.Second, 50*time.Millisecond).Should(Equal("failed"))

			status, _ := pipeline.GetFileStatus(vsf.ID)
			Expect(status.LastError).NotTo(BeNil())
			Expect(status.LastError.Code).To(Equal("embedding_error"))
		})

		It("should mark failed on empty file", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "empty-test"})
			Expect(err).NotTo(HaveOccurred())

			record, err := store.Save("empty.txt", []byte("   "), "assistants")
			Expect(err).NotTo(HaveOccurred())

			vsf, err := pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				status, _ := pipeline.GetFileStatus(vsf.ID)
				return status.Status
			}, 5*time.Second, 50*time.Millisecond).Should(Equal("failed"))

			status, _ := pipeline.GetFileStatus(vsf.ID)
			Expect(status.LastError.Code).To(Equal("empty_content"))
		})
	})

	Context("ListFileStatuses", func() {
		It("should return files for a specific vector store", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "list-test"})
			Expect(err).NotTo(HaveOccurred())

			record, err := store.Save("a.txt", []byte("content a"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			_, err = pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() int {
				return len(pipeline.ListFileStatuses(vs.ID))
			}, 5*time.Second, 50*time.Millisecond).Should(Equal(1))
		})

		It("should return empty for unknown store", func() {
			files := pipeline.ListFileStatuses("vs_unknown")
			Expect(files).To(BeEmpty())
		})
	})

	Context("DetachFile", func() {
		It("should remove chunks and update counts", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "detach-test"})
			Expect(err).NotTo(HaveOccurred())

			record, err := store.Save("test.txt", []byte("Some content here"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			vsf, err := pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(err).NotTo(HaveOccurred())

			// Wait for completion.
			Eventually(func() string {
				status, _ := pipeline.GetFileStatus(vsf.ID)
				return status.Status
			}, 5*time.Second, 50*time.Millisecond).Should(Equal("completed"))

			err = pipeline.DetachFile(ctx, vs.ID, vsf.ID)
			Expect(err).NotTo(HaveOccurred())

			// Verify file status removed.
			_, err = pipeline.GetFileStatus(vsf.ID)
			Expect(err).To(HaveOccurred())

			// Verify chunks removed.
			results, err := backend.Search(ctx, vs.ID, []float32{0.1, 0.1, 0.1}, 10, 0, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})

		It("should return error for non-existent vsf", func() {
			err := pipeline.DetachFile(ctx, "vs_x", "vsf_nonexistent")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("Start/Stop", func() {
		It("should be idempotent", func() {
			pipeline.Start() // already started in BeforeEach
			pipeline.Stop()
			pipeline.Stop() // double stop should not panic
		})
	})
})
