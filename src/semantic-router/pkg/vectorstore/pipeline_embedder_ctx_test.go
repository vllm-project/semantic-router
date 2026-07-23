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
	"runtime"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// ctxAwareEmbedder blocks inside Embed until either it is explicitly released
// or the context it is given is cancelled. This is the behavior that threading
// a context into the Embedder interface makes possible: unlike blockingEmbedder
// (which ignores ctx to model a non-cooperative stage that can only be freed by
// an explicit release), this embedder unwinds on its own once the pipeline's
// lifecycle context is cancelled.
type ctxAwareEmbedder struct {
	dim     int
	once    sync.Once
	started chan struct{}
	release chan struct{}
}

func newCtxAwareEmbedder(dim int) *ctxAwareEmbedder {
	return &ctxAwareEmbedder{
		dim:     dim,
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (e *ctxAwareEmbedder) Embed(ctx context.Context, _ string) ([]float32, error) {
	e.once.Do(func() { close(e.started) })
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-e.release:
		emb := make([]float32, e.dim)
		for i := range emb {
			emb[i] = 0.1
		}
		return emb, nil
	}
}

func (e *ctxAwareEmbedder) Dimension() int { return e.dim }

var _ Embedder = (*ctxAwareEmbedder)(nil)

var _ = Describe("IngestionPipeline embedder context propagation", func() {
	var (
		embedder *ctxAwareEmbedder
		f        *pipelineLifecycleFixture
		baseline int
	)

	BeforeEach(func() {
		embedder = newCtxAwareEmbedder(3)
		// Capture the goroutine baseline BEFORE the fixture starts the worker,
		// so the worker is not already counted in the baseline. Otherwise
		// `NumGoroutine() <= baseline+1` is trivially satisfied while the worker
		// is still unwinding, and the leak assertion stops being a real barrier.
		baseline = runtime.NumGoroutine()
		f = newPipelineLifecycleFixture(embedder)
	})

	AfterEach(func() {
		// Safety net only: the test itself should unwind the embedder via context
		// cancellation, not via an explicit release.
		select {
		case <-embedder.release:
		default:
			close(embedder.release)
		}
		_ = os.RemoveAll(f.tempDir)
	})

	It("unwinds an in-flight embed via lifecycle cancellation without an explicit release", func() {
		vs, err := f.mgr.CreateStore(f.ctx, CreateStoreRequest{Name: "ctx-embed"})
		Expect(err).NotTo(HaveOccurred())

		record, err := f.store.Save("ctx-embed.txt", []byte("content"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		vsf, err := f.pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		// Wait until the worker is parked inside Embed.
		Eventually(embedder.started, 5*time.Second).Should(BeClosed())

		// A bounded Stop cannot drain while the embed is in flight, so it returns
		// its deadline error and cancels the lifecycle root context.
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		Expect(f.pipeline.Stop(ctx)).To(MatchError(context.DeadlineExceeded))

		// The interrupted job is recorded as failed (eventually — failJob runs as
		// the worker unwinds, concurrently with this read), and is classified as
		// a cancellation rather than an embedding/backend fault.
		Eventually(func(g Gomega) {
			status, statusErr := f.pipeline.GetFileStatus(vsf.ID)
			g.Expect(statusErr).NotTo(HaveOccurred())
			g.Expect(status.Status).To(Equal("failed"))
			g.Expect(status.LastError).NotTo(BeNil())
			g.Expect(status.LastError.Code).To(Equal("cancelled"))
		}, 5*time.Second, 50*time.Millisecond).Should(Succeed())

		// The key A2 behavior: because the embedder now receives the (now
		// cancelled) lifecycle context, the parked embed returns on its own —
		// no explicit release is needed — so the worker unwinds and no
		// goroutines are leaked. A ctx-ignoring embedder would stay parked here.
		// The baseline was captured before the worker started, so this is a real
		// leak barrier, not a tautology.
		Eventually(runtime.NumGoroutine, 5*time.Second, 50*time.Millisecond).
			Should(BeNumerically("<=", baseline+1))
	})
})
