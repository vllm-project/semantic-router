package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
)

func TestManagementDeadlineStillRetiresLeasedRouterWithinOriginalBudget(t *testing.T) {
	runtime := extproc.NewRouterService(nil)
	var acquire extproc.AcquireFunc
	if err := runtime.Swap(&extproc.OpenAIRouter{}, func(lease extproc.AcquireFunc) { acquire = lease }); err != nil {
		t.Fatal(err)
	}
	release, ok := acquire()
	if !ok {
		t.Fatal("could not lease live generation")
	}
	defer release()
	accepted, finishRequest := make(chan struct{}), make(chan struct{})
	management := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		defer release()
		close(accepted)
		<-finishRequest
		w.WriteHeader(http.StatusNoContent)
	}))
	defer management.Close()
	defer closeTestSignal(finishRequest)
	requestDone := startManagementRequest(management)
	waitForTestSignal(t, accepted, "management request not accepted")
	retirementRequested, runtimeClosed, hookClosed := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var managementAttempts atomic.Int32
	hooks := []func(context.Context) error{func(context.Context) error { close(hookClosed); return nil }}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- shutdownRouterComponents(ctx,
			func(ctx context.Context) error { managementAttempts.Add(1); return management.Config.Shutdown(ctx) },
			func(ctx context.Context) error {
				close(retirementRequested)
				err := runtime.Shutdown(ctx)
				if err == nil {
					close(runtimeClosed)
				}
				return err
			}, &hooks, func(context.Context) error { return nil })
	}()
	select {
	case <-retirementRequested:
	case <-done:
		t.Fatal("management timeout skipped runtime retirement")
	case <-time.After(2 * time.Second):
		t.Fatal("shutdown exceeded its original budget")
	}
	// Actual RouterService leases must prevent closing an in-flight consumer.
	select {
	case <-runtimeClosed:
		t.Fatal("runtime closed under active lease")
	default:
	}
	select {
	case <-hookClosed:
		t.Fatal("unleased hooks closed under active management request")
	default:
	}
	close(finishRequest)
	if err := waitForTestError(t, done, "runtime did not retire after lease release"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("initial management deadline was lost: %v", err)
	}
	if err := waitForTestError(t, requestDone, "request did not finish"); err != nil {
		t.Fatal(err)
	}
	waitForTestSignal(t, runtimeClosed, "runtime remained open after drain")
	waitForTestSignal(t, hookClosed, "confirmed management drain did not release hooks")
	if managementAttempts.Load() != 2 {
		t.Fatalf("management completion was not confirmed: %d", managementAttempts.Load())
	}
	if lateRelease, admitted := acquire(); admitted {
		lateRelease()
		t.Fatal("retired generation accepted another lease")
	}
}
