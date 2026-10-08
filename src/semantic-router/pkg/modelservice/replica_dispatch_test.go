package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestReplicaDispatchRotatesIdleWorkersAcrossGenerations(t *testing.T) {
	first, runtimes := testReplicaPool(t, runtimetest.Model{ID: "one"}, runtimetest.Model{ID: "two"}, runtimetest.Model{ID: "three"}, runtimetest.Model{ID: "four"})
	declaration := first.members["primary"].pool.declaration
	for i := range 12 {
		// A configuration publication must not reset the idle-worker tie break.
		lease, err := first.manager.AcquireDeployments(map[string]config.ModelDeployment{"primary": declaration})
		if err != nil {
			t.Fatal(err)
		}
		body := json.RawMessage(fmt.Sprintf(`{"state":"request %d","questions":{"task":{"type":"noul","instructions":"Relevant?"}}}`, i))
		_, err = lease.SystemOne(t.Context(), "primary", body)
		if closeErr := lease.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, runtime := range runtimes {
		if got := runtime.Calls("decisions"); got != 3 {
			t.Fatalf("idle worker was skipped or preferred: %d exchanges, want 3", got)
		}
	}
}

func TestReplicaCanceledExchangesReleaseAdmissionWithoutReportingSuccess(t *testing.T) {
	for _, outcome := range []replicaOutcome{replicaCanceled, replicaTimeout} {
		t.Run(string(outcome), func(t *testing.T) {
			lease, runtimes := testReplicaPool(t, runtimetest.Model{ID: "worker"})
			pool := lease.members["primary"].pool
			worker := pool.workers[0]
			runtimes[0].SetDelay(400 * time.Millisecond)
			load := &worker.member.served.load
			load.mu.Lock()
			load.failures = 2
			load.mu.Unlock()
			count := func(result replicaOutcome) float64 {
				return testutil.ToFloat64(replicaRequestsTotal.WithLabelValues("primary", worker.id, string(result)))
			}
			beforeOK, beforeOutcome := count(replicaOK), count(outcome)
			ctx, cancel := context.WithTimeout(t.Context(), 150*time.Millisecond)
			defer cancel()
			completed := make(chan error, 1)
			go func() {
				_, err := lease.SystemOne(ctx, "primary", json.RawMessage(`{"state":"wait","questions":{"task":{"type":"noul","instructions":"Relevant?"}}}`))
				completed <- err
			}()
			deadline := time.Now().Add(time.Second)
			for {
				if pool.status().Replicas[0].Inflight == 1 {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("request did not reach the worker")
				}
				time.Sleep(time.Millisecond)
			}
			if outcome == replicaCanceled {
				cancel()
			}
			if err := <-completed; err == nil || !errors.Is(ctx.Err(), context.Canceled) && !errors.Is(ctx.Err(), context.DeadlineExceeded) {
				t.Fatalf("canceled request unexpectedly completed: %v", err)
			}
			if count(outcome) != beforeOutcome+1 || count(replicaOK) != beforeOK {
				t.Fatal("canceled exchange counted as successful or lost")
			}
			load.mu.Lock()
			defer load.mu.Unlock()
			if load.inflight != 0 || load.work != 0 || load.failures != 2 || !load.backoff.IsZero() {
				t.Fatalf("cancellation leaked admission or changed health: %+v", load)
			}
		})
	}
}
