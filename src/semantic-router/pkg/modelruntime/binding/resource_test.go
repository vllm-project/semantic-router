package binding

import (
	"context"
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
)

type testModel struct{ closes atomic.Int32 }

func (m *testModel) Close() error { m.closes.Add(1); return nil }

func testIdentity() ResourceIdentity {
	return ResourceIdentity{Artifact: "maintained-model", Revision: "revision-a", Provider: "test", Device: "cpu", Precision: "fp32"}
}

func TestPoolSharesCompatibleResourcesAndClosesLastOwner(t *testing.T) {
	pool := NewPool()
	model := &testModel{}
	var loads atomic.Int32
	loader := func(context.Context) (io.Closer, error) { loads.Add(1); return model, nil }
	first, err := pool.Acquire(context.Background(), testIdentity(), "one", nil, loader)
	if err != nil {
		t.Fatal(err)
	}
	second, err := pool.Acquire(context.Background(), testIdentity(), "one", nil, loader)
	if err != nil {
		t.Fatal(err)
	}
	if loads.Load() != 1 {
		t.Fatalf("loads=%d", loads.Load())
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if model.closes.Load() != 0 {
		t.Fatal("closed a resource still owned by another binding")
	}
	if err := second.Use(context.Background(), func(got io.Closer) error {
		if got != model {
			t.Fatal("shared owner has the wrong resource")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
	_ = second.Close()
	if model.closes.Load() != 1 {
		t.Fatalf("closes=%d", model.closes.Load())
	}
	if err := first.Use(context.Background(), func(io.Closer) error { t.Fatal("closed binding executed"); return nil }); !errors.Is(err, ErrClosed) {
		t.Fatalf("got %v", err)
	}
}

func TestResourceCloseWaitsForCanceledNativeCall(t *testing.T) {
	model := &testModel{}
	resource, err := NewPool().Acquire(context.Background(), testIdentity(), "", nil, func(context.Context) (io.Closer, error) { return model, nil })
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	started, finish, returned, closed := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		defer close(returned)
		_ = resource.Use(ctx, func(io.Closer) error { close(started); <-finish; return nil })
	}()
	<-started
	cancel()
	go func() { _ = resource.Close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("unloaded while non-preemptible native call was active")
	case <-time.After(20 * time.Millisecond):
	}
	if model.closes.Load() != 0 {
		t.Fatal("model closed early")
	}
	close(finish)
	<-returned
	<-closed
	if model.closes.Load() != 1 {
		t.Fatal("model was not released after execution completed")
	}
}

func TestSharedAliasesCannotBypassAdmission(t *testing.T) {
	pool := NewPool()
	load := func(context.Context) (io.Closer, error) { return &testModel{}, nil }
	first, err := pool.Acquire(context.Background(), testIdentity(), "1/0/shed", admission.NewSemaphore(1, 0, 0, admission.OverflowShed), load)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := pool.Acquire(context.Background(), testIdentity(), "1/0/shed", admission.Noop{}, load)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	started, finish, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		_ = first.Use(context.Background(), func(io.Closer) error { close(started); <-finish; return nil })
	}()
	<-started
	err = second.Use(context.Background(), func(io.Closer) error { t.Error("alias bypassed resource capacity"); return nil })
	if !errors.Is(err, admission.ErrQueueFull) {
		t.Errorf("got %v", err)
	}
	close(finish)
	<-done
	if _, err := pool.Acquire(context.Background(), testIdentity(), "2/0/shed", nil, load); !errors.Is(err, ErrCapability) {
		t.Fatalf("conflicting budget: %v", err)
	}
}

func TestFailedCandidateDoesNotReleaseServingResource(t *testing.T) {
	pool := NewPool()
	old := &testModel{}
	first, err := pool.Acquire(context.Background(), testIdentity(), "", nil, func(context.Context) (io.Closer, error) { return old, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	candidate := testIdentity()
	candidate.Revision = "revision-b"
	partial := &testModel{}
	want := errors.New("warmup failed")
	if _, err := pool.Acquire(context.Background(), candidate, "", nil, func(context.Context) (io.Closer, error) { return partial, want }); !errors.Is(err, want) {
		t.Fatal(err)
	}
	if partial.closes.Load() != 1 || old.closes.Load() != 0 {
		t.Fatal("candidate rollback affected serving model or leaked partial load")
	}
	if err := first.Use(context.Background(), func(io.Closer) error { return nil }); err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentLoadsAndCanceledWaiter(t *testing.T) {
	pool := NewPool()
	model := &testModel{}
	started, finish := make(chan struct{}), make(chan struct{})
	var loads atomic.Int32
	load := func(context.Context) (io.Closer, error) { loads.Add(1); close(started); <-finish; return model, nil }
	var owner *Resource
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		var err error
		owner, err = pool.Acquire(context.Background(), testIdentity(), "", nil, load)
		if err != nil {
			t.Error(err)
		}
	}()
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := pool.Acquire(ctx, testIdentity(), "", nil, load); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("canceled waiter got %v", err)
	}
	close(finish)
	wg.Wait()
	if loads.Load() != 1 || model.closes.Load() != 0 {
		t.Fatal("waiter disturbed resource ownership")
	}
	_ = owner.Close()
	if model.closes.Load() != 1 {
		t.Fatal("leaked reference from canceled waiter")
	}
}

func TestPhysicalIdentityIncludesExecutionConfiguration(t *testing.T) {
	base := testIdentity()
	want, _ := base.Key()
	variants := []ResourceIdentity{base, base, base, base}
	variants[0].Device = "migraphx:1"
	variants[1].Precision = "fp16"
	variants[2].Revision = "revision-b"
	variants[3].Execution = "different-effective-weights"
	for _, variant := range variants {
		got, _ := variant.Key()
		if got == want {
			t.Fatal("incompatible resource identities shared")
		}
	}
}

type failingCleanupModel struct{ failure error }

func (m failingCleanupModel) Close() error { return m.failure }

func TestFailedResourceLoadRetainsCleanupError(t *testing.T) {
	pool := NewPool()
	loadFailure, closeFailure := errors.New("load failed"), errors.New("cleanup failed")
	_, err := pool.Acquire(context.Background(), testIdentity(), "", nil, func(context.Context) (io.Closer, error) {
		return failingCleanupModel{failure: closeFailure}, loadFailure
	})
	if !errors.Is(err, loadFailure) || !errors.Is(err, closeFailure) {
		t.Fatalf("lost load or cleanup failure: %v", err)
	}
	model := &testModel{}
	next, err := pool.Acquire(context.Background(), testIdentity(), "", nil, func(context.Context) (io.Closer, error) { return model, nil })
	if err != nil {
		t.Fatalf("failed entry was retained: %v", err)
	}
	if closeErr := next.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if model.closes.Load() != 1 {
		t.Fatal("recovered resource leaked")
	}
}
