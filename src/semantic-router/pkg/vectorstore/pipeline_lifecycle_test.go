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
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

type blockingEmbedder struct {
	dim     int
	once    sync.Once
	started chan struct{}
	release chan struct{}
}

func newBlockingEmbedder(dim int) *blockingEmbedder {
	return &blockingEmbedder{
		dim:     dim,
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (e *blockingEmbedder) Embed(_ string) ([]float32, error) {
	e.once.Do(func() {
		close(e.started)
	})
	<-e.release

	emb := make([]float32, e.dim)
	for i := range emb {
		emb[i] = 0.1
	}
	return emb, nil
}

func (e *blockingEmbedder) Dimension() int {
	return e.dim
}

type pipelineLifecycleFixture struct {
	backend  *MemoryBackend
	store    *FileStore
	mgr      *Manager
	pipeline *IngestionPipeline
	tempDir  string
	ctx      context.Context
}

func newPipelineLifecycleFixture(embedder Embedder) *pipelineLifecycleFixture {
	tempDir, err := os.MkdirTemp("", "pipeline-lifecycle-test-*")
	Expect(err).NotTo(HaveOccurred())

	backend := NewMemoryBackend(MemoryBackendConfig{})
	store, err := NewFileStore(tempDir, NewMemoryMetadataRegistry())
	Expect(err).NotTo(HaveOccurred())

	mgr := NewManager(backend, NewMemoryMetadataRegistry(), 3, BackendTypeMemory)
	pipeline := NewIngestionPipeline(
		backend,
		store,
		mgr,
		embedder,
		PipelineConfig{Workers: 1, QueueSize: 10},
	)
	pipeline.Start()

	return &pipelineLifecycleFixture{
		backend:  backend,
		store:    store,
		mgr:      mgr,
		pipeline: pipeline,
		tempDir:  tempDir,
		ctx:      context.Background(),
	}
}

func (f *pipelineLifecycleFixture) cleanup() {
	f.pipeline.Stop()
	_ = os.RemoveAll(f.tempDir)
}

var _ = Describe("IngestionPipeline lifecycle", func() {
	var f *pipelineLifecycleFixture

	BeforeEach(func() {
		f = newPipelineLifecycleFixture(&mockEmbedder{dim: 3})
	})

	AfterEach(func() {
		f.cleanup()
	})

	It("rejects new attachments while stopped without creating stuck statuses", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "stopped"})
		Expect(err).NotTo(HaveOccurred())

		record, err := f.store.Save("stopped.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		f.pipeline.Stop()
		_, err = f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("ingestion pipeline is not running"))
		Expect(f.pipeline.ListFileStatuses(vs.ID)).To(BeEmpty())

		updated, err := f.mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Total).To(Equal(0))
	})

	It("restarts workers after a stop", func() {
		f.pipeline.Stop()
		f.pipeline.Start()

		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "restarted"})
		Expect(err).NotTo(HaveOccurred())

		record, err := f.store.Save("restarted.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() string {
			status, err := f.pipeline.GetFileStatus(vsf.ID)
			if err != nil {
				return ""
			}
			return status.Status
		}, 5*time.Second, 50*time.Millisecond).Should(Equal("completed"))
	})
})

var _ = Describe("IngestionPipeline queued shutdown", func() {
	var (
		embedder *blockingEmbedder
		f        *pipelineLifecycleFixture
	)

	BeforeEach(func() {
		embedder = newBlockingEmbedder(3)
		f = newPipelineLifecycleFixture(embedder)
	})

	AfterEach(func() {
		f.cleanup()
	})

	It("fails queued attachments during stop", func() {
		released := false
		defer func() {
			if !released {
				close(embedder.release)
			}
		}()

		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "queued-stop"})
		Expect(err).NotTo(HaveOccurred())

		firstRecord, err := f.store.Save("first.txt", []byte("first content"), "assistants")
		Expect(err).NotTo(HaveOccurred())
		secondRecord, err := f.store.Save("second.txt", []byte("second content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		first, err := f.pipeline.AttachFile(vs.ID, firstRecord.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		second, err := f.pipeline.AttachFile(vs.ID, secondRecord.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		stopped := make(chan struct{})
		go func() {
			defer close(stopped)
			f.pipeline.Stop()
		}()

		Eventually(func() string {
			status, statusErr := f.pipeline.GetFileStatus(second.ID)
			if statusErr != nil {
				return ""
			}
			return status.Status
		}, 5*time.Second, 50*time.Millisecond).Should(Equal("failed"))

		secondStatus, err := f.pipeline.GetFileStatus(second.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(secondStatus.LastError).NotTo(BeNil())
		Expect(secondStatus.LastError.Code).To(Equal("pipeline_stopped"))

		close(embedder.release)
		released = true
		Eventually(stopped, 5*time.Second).Should(BeClosed())

		firstStatus, err := f.pipeline.GetFileStatus(first.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(firstStatus.Status).To(Equal("completed"))

		updated, err := f.mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Completed).To(Equal(1))
		Expect(updated.FileCounts.Failed).To(Equal(1))
		Expect(updated.FileCounts.InProgress).To(Equal(0))
		Expect(updated.FileCounts.Total).To(Equal(2))
	})
})
