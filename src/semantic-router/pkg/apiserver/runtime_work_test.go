//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRetainedAPIWorkKeepsShutdownOwnershipAfterCancellation(t *testing.T) {
	owner := &countedAPIResource{}
	resources := newServerOwnedResources([]io.Closer{owner})
	ctx, cancel := context.WithCancel(context.WithValue(context.Background(), serverOwnedResourcesContextKey{}, resources))
	defer cancel()
	started, finish := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(finish) }) }
	defer unblock()
	var releases atomic.Int32
	result := make(chan error, 1)
	go func() {
		_, err := runRetainedAPIWork(ctx, func() { releases.Add(1) }, func(context.Context) (int, error) { close(started); <-finish; return 1, nil })
		result <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("worker did not start")
	}
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel error=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("caller did not return on cancellation")
	}
	resources.beginDrain()
	drained := make(chan error, 1)
	go func() { drained <- resources.drainAndClose() }()
	select {
	case <-drained:
		t.Fatal("shutdown unloaded an active runtime worker")
	case <-time.After(20 * time.Millisecond):
	}
	if releases.Load() != 0 || owner.closes.Load() != 0 {
		t.Fatal("cancellation released native ownership")
	}
	unblock()
	select {
	case err := <-drained:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("completed worker did not drain")
	}
	if releases.Load() != 1 || owner.closes.Load() != 1 {
		t.Fatalf("release=%d close=%d", releases.Load(), owner.closes.Load())
	}
}

func TestRetainedAPIWorkReleasesOnPanicAndShutdownRejection(t *testing.T) {
	for _, stopping := range []bool{false, true} {
		resources := newServerOwnedResources(nil)
		if stopping {
			resources.beginDrain()
		}
		ctx := context.WithValue(context.Background(), serverOwnedResourcesContextKey{}, resources)
		releases, invoked := 0, false
		_, err := runRetainedAPIWork(ctx, func() { releases++ }, func(context.Context) (int, error) { invoked = true; panic("provider failure") })
		if !errors.Is(err, errAPIWorkerUnavailable) || releases != 1 || invoked == stopping {
			t.Fatalf("stopping=%v err=%v releases=%d invoked=%v", stopping, err, releases, invoked)
		}
		if err := resources.drainAndClose(); err != nil {
			t.Fatal(err)
		}
	}
}
