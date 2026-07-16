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

// releaseAll unblocks any goroutine parked in Embed. It is safe to call more
// than once so tests and their AfterEach hooks can both release without a
// double-close panic.
func (e *blockingEmbedder) releaseAll() {
	e.releaseOnce.Do(func() {
		close(e.release)
	})
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
		defer embedder.releaseAll()

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

	It("preserves the stopping state so a second Stop does not falsely report success", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "double-stop"})
		Expect(err).NotTo(HaveOccurred())

		record, err := f.store.Save("double.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		_, err = f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		// First Stop times out while the worker is wedged inside Embed.
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		Expect(f.pipeline.Stop(ctx)).To(MatchError(context.DeadlineExceeded))

		// A second bounded Stop must NOT report a clean shutdown: the previous
		// generation's worker is still live. It must again honor its deadline and
		// return DeadlineExceeded rather than nil (the false-success bug).
		ctx2, cancel2 := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel2()
		Expect(f.pipeline.Stop(ctx2)).To(MatchError(context.DeadlineExceeded))

		// Once the wedged stage is released, a Stop with a real deadline observes
		// the worker join and returns nil.
		embedder.releaseAll()
		Eventually(func() error {
			ctxN, cancelN := context.WithTimeout(context.Background(), time.Second)
			defer cancelN()
			return f.pipeline.Stop(ctxN)
		}, 5*time.Second, 50*time.Millisecond).Should(BeNil())
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

	// Regression for the generation-reuse race: a bounded Stop that times out
	// leaves the old generation's worker live. An immediate Start must create a
	// fresh generation (its own WaitGroup, queue, and root context) rather than
	// reusing resources the old worker still holds, so the old worker can neither
	// corrupt the new WaitGroup nor steal a new-generation job with its cancelled
	// root context.
	It("starts a fresh generation and processes new jobs after a timed-out stop", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "restart-after-timeout"})
		Expect(err).NotTo(HaveOccurred())

		wedged, err := f.store.Save("wedged.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		wedgedFile, err := f.pipeline.AttachFile(vs.ID, wedged.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		// Park the only worker inside Embed so Stop cannot drain gracefully.
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		Expect(f.pipeline.Stop(ctx)).To(MatchError(context.DeadlineExceeded))

		// Immediately restart. Start must block until the still-stopping
		// generation joins; release the wedged embedder so that join can happen.
		startReturned := make(chan struct{})
		go func() {
			defer close(startReturned)
			f.pipeline.Start()
		}()

		// Start is blocked on the old generation's join until we release it.
		Consistently(startReturned, 200*time.Millisecond, 20*time.Millisecond).ShouldNot(BeClosed())
		embedder.releaseAll()
		Eventually(startReturned, 5*time.Second).Should(BeClosed())

		// The new generation must process a fresh job to completion. Because the
		// released embedder now returns immediately, this exercises the new queue
		// and new root context.
		fresh, err := f.store.Save("fresh.txt", []byte("fresh content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		freshFile, err := f.pipeline.AttachFile(vs.ID, fresh.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() string {
			status, statusErr := f.pipeline.GetFileStatus(freshFile.ID)
			if statusErr != nil {
				return ""
			}
			return status.Status
		}, 5*time.Second, 50*time.Millisecond).Should(Equal("completed"))

		// The wedged file from the old generation completes once its worker is
		// released (it was mid-Embed when the deadline elapsed); either way it
		// must not remain in_progress forever.
		Eventually(func() string {
			status, statusErr := f.pipeline.GetFileStatus(wedgedFile.ID)
			if statusErr != nil {
				return ""
			}
			return status.Status
		}, 5*time.Second, 50*time.Millisecond).Should(Or(Equal("completed"), Equal("failed")))
	})
})

var _ = Describe("IngestionPipeline attach during restart after timed-out Stop", func() {
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

	// Regression: after a timed-out Stop the pipeline is stateStopping with a
	// wedged worker. A concurrent Start must not hold lifecycleMu while it waits
	// for the old generation to join, otherwise a concurrent AttachFile can
	// reserve a count and then block forever behind Start, stranding the count
	// with no compensation. This asserts attach makes progress (does not hang)
	// and file counts stay consistent once everything settles.
	It("does not strand counts when attach races a restart", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "attach-restart-race"})
		Expect(err).NotTo(HaveOccurred())

		wedged, err := f.store.Save("wedged.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())
		_, err = f.pipeline.AttachFile(vs.ID, wedged.ID, nil)
		Expect(err).NotTo(HaveOccurred())
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		// Time out the Stop with the worker wedged inside Embed.
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		Expect(f.pipeline.Stop(ctx)).To(MatchError(context.DeadlineExceeded))

		// Kick off a restart (will block until the wedged gen joins) and a
		// concurrent attach. Neither must hang beyond release.
		startDone := make(chan struct{})
		go func() { defer close(startDone); f.pipeline.Start() }()

		attachDone := make(chan error, 1)
		go func() {
			rec, saveErr := f.store.Save("racer.txt", []byte("content"), "assistants")
			if saveErr != nil {
				attachDone <- saveErr
				return
			}
			_, aErr := f.pipeline.AttachFile(vs.ID, rec.ID, nil)
			attachDone <- aErr
		}()

		// Release the wedged worker so the old generation can join and the
		// restart can complete.
		embedder.releaseAll()

		Eventually(startDone, 5*time.Second).Should(BeClosed())
		var attachErr error
		Eventually(attachDone, 5*time.Second).Should(Receive(&attachErr))
		// The attach either enqueued onto the fresh generation (nil) or was
		// rejected because it observed the stopping/stopped window
		// ("not running"). Both are valid; the invariant is it must not hang.
		if attachErr != nil {
			Expect(attachErr.Error()).To(ContainSubstring("not running"))
		}

		// Once the dust settles, counts must be internally consistent:
		// completed + failed + in_progress == total, with no negative values and
		// nothing permanently stuck in_progress.
		Eventually(func() bool {
			updated, gErr := f.mgr.GetStore(vs.ID)
			if gErr != nil {
				return false
			}
			fc := updated.FileCounts
			if fc.InProgress != 0 {
				return false
			}
			return fc.Completed+fc.Failed+fc.InProgress == fc.Total &&
				fc.Completed >= 0 && fc.Failed >= 0 && fc.Total >= 0
		}, 5*time.Second, 50*time.Millisecond).Should(BeTrue())
	})
})
