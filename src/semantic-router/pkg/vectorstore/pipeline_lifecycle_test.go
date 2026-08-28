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
	dim         int
	once        sync.Once
	releaseOnce sync.Once
	started     chan struct{}
	release     chan struct{}
}

func newBlockingEmbedder(dim int) *blockingEmbedder {
	return &blockingEmbedder{
		dim:     dim,
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
}

// Embed intentionally ignores ctx and blocks until released, modelling a
// stage that does not honor cancellation mid-call. This is what the bounded
// Stop(ctx) path must defend against; making stages natively ctx-interruptible
// is a deliberate follow-up tracked under #2474.
func (e *blockingEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
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

func (e *blockingEmbedder) releaseAll() {
	e.releaseOnce.Do(func() { close(e.release) })
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
	return newPipelineLifecycleFixtureWithRegistry(embedder, NewMemoryMetadataRegistry())
}

func newPipelineLifecycleFixtureWithRegistry(embedder Embedder, registry StoreRegistry) *pipelineLifecycleFixture {
	tempDir, err := os.MkdirTemp("", "pipeline-lifecycle-test-*")
	Expect(err).NotTo(HaveOccurred())

	backend := NewMemoryBackend(MemoryBackendConfig{})
	store, err := NewFileStore(tempDir, NewMemoryMetadataRegistry())
	Expect(err).NotTo(HaveOccurred())

	mgr := NewManager(backend, registry, 3, BackendTypeMemory)
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

type stallingStoreRegistry struct {
	StoreRegistry
	mu      sync.RWMutex
	stalled bool
	started chan struct{}
	once    sync.Once
}

func newStallingStoreRegistry() *stallingStoreRegistry {
	return &stallingStoreRegistry{
		StoreRegistry: NewMemoryMetadataRegistry(),
		started:       make(chan struct{}),
	}
}

func (r *stallingStoreRegistry) SaveStore(ctx context.Context, vs *VectorStore) error {
	r.mu.RLock()
	stalled := r.stalled
	r.mu.RUnlock()
	if stalled {
		r.once.Do(func() { close(r.started) })
		<-ctx.Done()
		return ctx.Err()
	}
	return r.StoreRegistry.SaveStore(ctx, vs)
}

func (r *stallingStoreRegistry) stall() {
	r.mu.Lock()
	r.stalled = true
	r.mu.Unlock()
}

func (r *stallingStoreRegistry) resume() {
	r.mu.Lock()
	r.stalled = false
	r.mu.Unlock()
}

func (f *pipelineLifecycleFixture) cleanup() {
	_ = f.pipeline.Stop(context.Background())
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

		_ = f.pipeline.Stop(context.Background())
		_, err = f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("ingestion pipeline is not running"))
		Expect(f.pipeline.ListFileStatuses(vs.ID)).To(BeEmpty())

		updated, err := f.mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Total).To(Equal(0))
	})

	It("restarts workers after a stop", func() {
		_ = f.pipeline.Stop(context.Background())
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
				embedder.releaseAll()
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
			_ = f.pipeline.Stop(context.Background())
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

		embedder.releaseAll()
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

var _ = Describe("IngestionPipeline bounded Stop", func() {
	var (
		embedder *blockingEmbedder
		f        *pipelineLifecycleFixture
	)

	BeforeEach(func() {
		embedder = newBlockingEmbedder(3)
		f = newPipelineLifecycleFixture(embedder)
	})

	AfterEach(func() {
		// Release the wedged embedder so the worker can unwind, then clean up.
		embedder.releaseAll()
		_ = f.pipeline.Stop(context.Background())
		_ = os.RemoveAll(f.tempDir)
	})

	It("returns within the deadline when a job is wedged inside a stage", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "wedged-stop"})
		Expect(err).NotTo(HaveOccurred())

		record, err := f.store.Save("wedged.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		_, err = f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		// Wait until the worker is blocked inside Embed, so Stop cannot drain
		// gracefully and must fall back to its bounded deadline path.
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()

		start := time.Now()
		stopErr := make(chan error, 1)
		go func() { stopErr <- f.pipeline.Stop(ctx) }()

		var err2 error
		Eventually(stopErr, 5*time.Second, 20*time.Millisecond).Should(Receive(&err2))
		Expect(err2).To(MatchError(context.DeadlineExceeded))
		// Stop must return promptly once the deadline elapses, not hang on the
		// wedged worker.
		Expect(time.Since(start)).To(BeNumerically("<", 3*time.Second))
	})

	It("is idempotent and safe to call after a bounded Stop", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "double-stop"})
		Expect(err).NotTo(HaveOccurred())

		record, err := f.store.Save("double.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		_, err = f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		Expect(f.pipeline.Stop(ctx)).To(MatchError(context.DeadlineExceeded))

		// A second Stop must report that the generation is still stopping instead
		// of falsely claiming a clean shutdown. Once the wedged stage is released,
		// a retry can observe the worker join and return nil.
		ctx2, cancel2 := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel2()
		Expect(f.pipeline.Stop(ctx2)).To(MatchError(context.DeadlineExceeded))
		embedder.releaseAll()
		Eventually(func() error {
			ctx3, cancel3 := context.WithTimeout(context.Background(), time.Second)
			defer cancel3()
			return f.pipeline.Stop(ctx3)
		}, 5*time.Second, 50*time.Millisecond).Should(BeNil())
	})
})

var _ = Describe("IngestionPipeline bounded queued cleanup", func() {
	var (
		embedder *blockingEmbedder
		registry *stallingStoreRegistry
		f        *pipelineLifecycleFixture
	)

	BeforeEach(func() {
		embedder = newBlockingEmbedder(3)
		registry = newStallingStoreRegistry()
		f = newPipelineLifecycleFixtureWithRegistry(embedder, registry)
	})

	AfterEach(func() {
		registry.resume()
		embedder.releaseAll()
		_ = f.pipeline.Stop(context.Background())
		_ = os.RemoveAll(f.tempDir)
	})

	It("shares the Stop deadline across queued cleanup", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "stalled-registry"})
		Expect(err).NotTo(HaveOccurred())

		first, err := f.store.Save("active.txt", []byte("active content"), "assistants")
		Expect(err).NotTo(HaveOccurred())
		_, err = f.pipeline.AttachFile(vs.ID, first.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		var queued []*VectorStoreFile
		for i := 0; i < 3; i++ {
			record, saveErr := f.store.Save("queued.txt", []byte("queued content"), "assistants")
			Expect(saveErr).NotTo(HaveOccurred())
			status, attachErr := f.pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(attachErr).NotTo(HaveOccurred())
			queued = append(queued, status)
		}

		registry.stall()
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		start := time.Now()
		err = f.pipeline.Stop(ctx)

		Expect(err).To(MatchError(context.DeadlineExceeded))
		Expect(time.Since(start)).To(BeNumerically("<", 3*time.Second))
		Expect(registry.started).To(BeClosed())
		status, statusErr := f.pipeline.GetFileStatus(queued[0].ID)
		Expect(statusErr).NotTo(HaveOccurred())
		Expect(status.Status).To(Equal("failed"))
	})
})

var _ = Describe("IngestionPipeline restart after timed-out Stop", func() {
	var (
		embedder *blockingEmbedder
		f        *pipelineLifecycleFixture
	)

	BeforeEach(func() {
		embedder = newBlockingEmbedder(3)
		f = newPipelineLifecycleFixture(embedder)
	})

	AfterEach(func() {
		embedder.releaseAll()
		_ = f.pipeline.Stop(context.Background())
		_ = os.RemoveAll(f.tempDir)
	})

	It("does not reuse a timed-out generation during restart", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "restart-generation"})
		Expect(err).NotTo(HaveOccurred())
		record, err := f.store.Save("restart.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())
		_, err = f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		Expect(f.pipeline.Stop(ctx)).To(MatchError(context.DeadlineExceeded))

		started := make(chan struct{})
		go func() {
			f.pipeline.Start()
			close(started)
		}()
		Consistently(started, 150*time.Millisecond, 20*time.Millisecond).ShouldNot(BeClosed())
		embedder.releaseAll()
		Eventually(started, 5*time.Second).Should(BeClosed())

		second, err := f.store.Save("restart-second.txt", []byte("new content"), "assistants")
		Expect(err).NotTo(HaveOccurred())
		vsf, err := f.pipeline.AttachFile(vs.ID, second.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() string {
			status, statusErr := f.pipeline.GetFileStatus(vsf.ID)
			if statusErr != nil {
				return ""
			}
			return status.Status
		}, 5*time.Second, 50*time.Millisecond).Should(Equal("completed"))
		Expect(f.pipeline.Stop(context.Background())).To(BeNil())
	})
})
