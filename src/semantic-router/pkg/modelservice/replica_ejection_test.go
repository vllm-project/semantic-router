package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// fakeReplica is one worker of an ejection test pool and the /health probes
// it received.
type fakeReplica struct {
	runtime *runtimetest.Runtime
	probes  atomic.Int64
	worker  replicaWorker
}

// ejectionPool serves the named fake workers as the replicas of one
// deployment that declares detection, once every one of them is ready.
func ejectionPool(t *testing.T, detection *config.DeploymentOutlierDetection, names ...string) (*Lease, *replicaPool, map[string]*fakeReplica) {
	t.Helper()
	return replicaDeployment(t, config.ModelDeployment{OutlierDetection: detection}, names...)
}

// replicaDeployment serves the named fake workers as the replicas of
// declaration, once every one of them is ready.
func replicaDeployment(t *testing.T, declaration config.ModelDeployment, names ...string) (*Lease, *replicaPool, map[string]*fakeReplica) {
	t.Helper()
	declaration.Provider, declaration.Artifact = config.ModelRuntimeProvider, poolArtifact
	replicas := make(map[string]*fakeReplica, len(names))
	for _, name := range names {
		replica := &fakeReplica{runtime: runtimetest.New(runtimetest.Model{ID: name, Repo: poolArtifact, ModelSHA256: strings.Repeat("a", 64)})}
		replicas[name] = replica
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/health":
				replica.probes.Add(1)
			case "/v1/systemone":
				r.URL.Path = "/v1/decisions"
			}
			replica.runtime.Handler().ServeHTTP(w, r)
		}))
		t.Cleanup(server.Close)
		declaration.Replicas = append(declaration.Replicas, config.ModelReplica{Endpoint: server.URL, Model: name})
	}
	manager := NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"primary": declaration})
	if err != nil {
		t.Fatal(err)
	}
	pool := lease.members["primary"].pool
	for _, w := range pool.workers {
		replicas[w.member.served.name].worker = w
	}
	waitFor(t, func() bool { return pool.status().ReadyReplicas == len(names) }, "replicas did not become ready")
	return lease, pool, replicas
}

func waitFor(t *testing.T, done func() bool, failure string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !done() {
		if time.Now().After(deadline) {
			t.Fatal(failure)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func intPtr(v int) *int { return &v }

func (r *fakeReplica) answer(p *replicaPool) { p.recordTimeouts(r.worker, replicaOK, time.Now()) }

// timeOut records an exchange dispatched a second ago that ran out of its deadline.
func (r *fakeReplica) timeOut(p *replicaPool) {
	p.recordTimeouts(r.worker, replicaTimeout, time.Now().Add(-time.Second))
}

func (r *fakeReplica) ejected() bool { return r.cooldown() > 0 }

// cooldown is how long the replica stays out of dispatch for its timeouts.
func (r *fakeReplica) cooldown() time.Duration {
	load := &r.worker.member.served.load
	load.mu.Lock()
	defer load.mu.Unlock()
	if load.ejected.IsZero() {
		return 0
	}
	return time.Until(load.ejected)
}

// endCooldown moves the replica's ejection into the past.
func (r *fakeReplica) endCooldown() {
	load := &r.worker.member.served.load
	load.mu.Lock()
	defer load.mu.Unlock()
	if !load.ejected.IsZero() {
		load.ejected = time.Now().Add(-time.Millisecond)
	}
}

func (r *fakeReplica) state(p *replicaPool) ReplicaStatus {
	for _, replica := range p.status().Replicas {
		if replica.ID == r.worker.id {
			return replica
		}
	}
	return ReplicaStatus{}
}

func (r *fakeReplica) ejections(p *replicaPool) float64 {
	return testutil.ToFloat64(replicaEjectionsTotal.WithLabelValues(p.name, r.worker.id))
}

func (r *fakeReplica) readmissions(p *replicaPool, reason string) float64 {
	return testutil.ToFloat64(replicaReadmissionsTotal.WithLabelValues(p.name, r.worker.id, reason))
}

func TestReplicaEjectionCountsConsecutiveTimeouts(t *testing.T) {
	_, pool, replicas := ejectionPool(t, nil, "healthy", "slow")
	healthy, slow := replicas["healthy"], replicas["slow"]
	healthy.answer(pool)
	slow.timeOut(pool)
	slow.timeOut(pool)
	// Cancellations, refusals and failures are no answer: they neither count nor reset.
	for _, outcome := range []replicaOutcome{replicaCanceled, replicaRejected, replicaFailed} {
		pool.recordTimeouts(slow.worker, outcome, time.Now())
	}
	if slow.ejected() {
		t.Fatal("ejected before three consecutive timeouts")
	}
	slow.answer(pool)
	slow.timeOut(pool)
	slow.timeOut(pool)
	if slow.ejected() {
		t.Fatal("a successful answer did not start the count again")
	}
	ejections, probes := slow.ejections(pool), slow.probes.Load()
	healthy.answer(pool)
	slow.timeOut(pool)
	if left := slow.cooldown(); left <= 29*time.Second || left > 30*time.Second {
		t.Fatalf("ejected for %v, want the 30s default", left)
	}
	if slow.ejections(pool) != ejections+1 {
		t.Fatal("ejection not counted")
	}
	status, replica := pool.status(), slow.state(pool)
	if replica.Ready || replica.State != "ejected" || replica.Reason != ejectedReason || status.State != "degraded" || status.ReadyReplicas != 1 {
		t.Fatalf("inventory hides the ejection: %+v", status)
	}
	for range 4 {
		w, release, err := pool.retain(1)
		if err != nil {
			t.Fatal(err)
		}
		release(replicaRejected)
		if w.id == slow.worker.id {
			t.Fatal("an ejected replica was dispatched")
		}
	}
	waitFor(t, func() bool { return slow.probes.Load() > probes }, "the ejected replica was not probed")
}

func TestReplicaEjectionCooldownGrowsUntilAnAnswer(t *testing.T) {
	_, pool, replicas := ejectionPool(t, &config.DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(2), BaseEjectionMs: 1000, MaxEjectionMs: 3000}, "healthy", "slow")
	healthy, slow := replicas["healthy"], replicas["slow"]
	slow.timeOut(pool)
	expired, answered := slow.readmissions(pool, readmittedCooldown), slow.readmissions(pool, readmittedAnswered)
	for i, want := range []time.Duration{time.Second, 2 * time.Second, 3 * time.Second, 3 * time.Second} {
		// The count outlives an ejection: once its cool-down ends, the next
		// timeout ejects the replica again, for longer.
		healthy.answer(pool)
		slow.timeOut(pool)
		if left := slow.cooldown(); left <= want-500*time.Millisecond || left > want {
			t.Fatalf("ejection %d lasts %v, want %v", i+1, left, want)
		}
		slow.endCooldown()
		if state := slow.state(pool).State; state != "ready" {
			t.Fatalf("ejection %d: %s after its cool-down", i+1, state)
		}
	}
	if got := slow.readmissions(pool, readmittedCooldown) - expired; got != 4 {
		t.Fatalf("%v readmissions after a cool-down, want 4", got)
	}
	// A successful answer readmits the replica at once and starts both the
	// count and the cool-down's growth again.
	slow.timeOut(pool)
	slow.answer(pool)
	if slow.ejected() || slow.readmissions(pool, readmittedAnswered) != answered+1 {
		t.Fatal("an answer did not readmit the replica")
	}
	healthy.answer(pool)
	slow.timeOut(pool)
	if slow.ejected() {
		t.Fatal("ejected below the threshold after an answer")
	}
	slow.timeOut(pool)
	if left := slow.cooldown(); left <= 500*time.Millisecond || left > time.Second {
		t.Fatalf("ejected for %v after an answer, want the base again", left)
	}
}

func TestReplicaEjectionKeepsAReplicaThatServes(t *testing.T) {
	t.Run("no other replica answered since the timeouts began", func(t *testing.T) {
		_, pool, replicas := ejectionPool(t, nil, "healthy", "slow")
		healthy, slow := replicas["healthy"], replicas["slow"]
		healthy.answer(pool)
		for range 5 {
			pool.recordTimeouts(slow.worker, replicaTimeout, time.Now())
		}
		if slow.ejected() {
			t.Fatal("a model slow on every replica lost one")
		}
		healthy.answer(pool)
		slow.timeOut(pool)
		if !slow.ejected() {
			t.Fatal("not ejected once another replica answered")
		}
	})
	t.Run("the other replica is out of dispatch", func(t *testing.T) {
		_, pool, replicas := ejectionPool(t, nil, "healthy", "slow")
		healthy, slow := replicas["healthy"], replicas["slow"]
		healthy.answer(pool)
		load := &healthy.worker.member.served.load
		load.mu.Lock()
		load.backoff = time.Now().Add(time.Minute)
		load.mu.Unlock()
		for range 3 {
			slow.timeOut(pool)
		}
		if slow.ejected() {
			t.Fatal("ejected the last replica in dispatch")
		}
	})
	t.Run("one worker", func(t *testing.T) {
		_, pool, replicas := ejectionPool(t, nil, "only")
		for range 5 {
			replicas["only"].timeOut(pool)
		}
		if replicas["only"].ejected() {
			t.Fatal("ejected a deployment's only worker")
		}
	})
	t.Run("every replica ejected", func(t *testing.T) {
		_, pool, replicas := ejectionPool(t, nil, "one", "two")
		for _, replica := range replicas {
			load := &replica.worker.member.served.load
			load.mu.Lock()
			load.ejected = time.Now().Add(time.Minute)
			load.mu.Unlock()
		}
		if status := pool.status(); status.ReadyReplicas != 2 || status.State != "ready" {
			t.Fatalf("ejection emptied the pool: %+v", status)
		}
		_, release, err := pool.retain(1)
		if err != nil {
			t.Fatalf("ejection emptied the pool: %v", err)
		}
		release(replicaOK)
	})
}

func TestReplicaEjectionHoldsAtMaxEjectionPercent(t *testing.T) {
	for _, test := range []struct {
		detection *config.DeploymentOutlierDetection
		ejected   int
	}{
		{&config.DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(1), MaxEjectionPercent: 25}, 1},
		{&config.DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(1)}, 2},
		{&config.DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(1), MaxEjectionPercent: 100}, 3},
		{&config.DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(0)}, 0},
	} {
		t.Run(fmt.Sprintf("%d", test.ejected), func(t *testing.T) {
			_, pool, replicas := ejectionPool(t, test.detection, "healthy", "a", "b", "c")
			replicas["healthy"].answer(pool)
			ejected := 0
			for _, name := range []string{"a", "b", "c"} {
				for range 3 {
					replicas[name].timeOut(pool)
				}
				if replicas[name].ejected() {
					ejected++
				}
			}
			if ejected != test.ejected || replicas["healthy"].ejected() {
				t.Fatalf("%d of 4 replicas ejected, want %d", ejected, test.ejected)
			}
		})
	}
}

// One replica's forward hangs while its /health stays ready (the Nov 10
// incident): every exchange it takes runs out of its deadline, the caller's
// or, sooner, the deployment's timeout_ms, which also backs it off between
// exchanges; calls 25 ms apart reach it again after each backoff.
func TestReplicaPoolStopsDispatchingToAHungReplica(t *testing.T) {
	for _, timeoutMs := range []int{0, 50} {
		t.Run(fmt.Sprintf("timeout_ms %d", timeoutMs), func(t *testing.T) {
			lease, pool, replicas := replicaDeployment(t, config.ModelDeployment{TimeoutMs: timeoutMs}, "healthy", "hung")
			hung := replicas["hung"]
			hung.runtime.SetDelay(300 * time.Millisecond)
			timeouts := replicaRequestsTotal.WithLabelValues(pool.name, hung.worker.id, string(replicaTimeout))
			before := testutil.ToFloat64(timeouts)
			failed := 0
			for i := 0; i < 80 && !hung.ejected(); i++ {
				ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
				body := json.RawMessage(fmt.Sprintf(`{"state":"request %d","questions":{"q":{"type":"noul","instructions":"Relevant?"}}}`, i))
				if _, err := lease.SystemOne(ctx, "primary", body); err != nil {
					failed++
				}
				cancel()
				time.Sleep(25 * time.Millisecond)
			}
			if calls := hung.runtime.Calls("decisions"); calls != 3 || failed != 3 || testutil.ToFloat64(timeouts)-before != 3 {
				t.Fatalf("the hung replica took %d exchanges, %d failed and %v timed out, want 3", calls, failed, testutil.ToFloat64(timeouts)-before)
			}
			waitFor(t, func() bool { return hung.state(pool).State == "ejected" }, "the hung replica was not ejected")
			if status := lease.Statuses()[0]; status.State != "degraded" || status.ReadyReplicas != 1 {
				t.Fatalf("pool %+v", status)
			}
		})
	}
}

// An exchange past the deployment's timeout_ms backs the replica off and
// probes it, as a failure does, so with ejection off a hung replica still
// leaves dispatch between its timeouts instead of taking every other call.
func TestReplicaPoolBacksOffAReplicaPastTimeoutMs(t *testing.T) {
	off := &config.DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(0)}
	lease, pool, replicas := replicaDeployment(t, config.ModelDeployment{TimeoutMs: 50, OutlierDetection: off}, "healthy", "hung")
	hung := replicas["hung"]
	hung.runtime.SetDelay(300 * time.Millisecond)
	probes := hung.probes.Load()
	for i := 0; i < 4 && hung.runtime.Calls("decisions") == 0; i++ {
		ctx, cancel := context.WithTimeout(t.Context(), time.Second)
		body := json.RawMessage(fmt.Sprintf(`{"state":"request %d","questions":{"q":{"type":"noul","instructions":"Relevant?"}}}`, i))
		_, _ = lease.SystemOne(ctx, "primary", body)
		cancel()
	}
	if replica := hung.state(pool); replica.State != "backoff" {
		t.Fatalf("after an exchange past timeout_ms the hung replica is %+v, want backoff", replica)
	}
	waitFor(t, func() bool { return hung.probes.Load() > probes }, "the replica was not probed")
}

// A connection the dialer gave up on stays a failure, with its backoff and
// probe: only an exchange past its deadline counts toward ejection.
func TestReplicaTransportErrorKeepsADialTimeoutAFailure(t *testing.T) {
	client := &http.Client{Transport: &http.Transport{DialContext: (&net.Dialer{Timeout: time.Nanosecond}).DialContext}}
	response, err := client.Get("http://127.0.0.1:1")
	if err == nil {
		response.Body.Close()
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("not a dial timeout: %v", err)
	}
	if outcome := replicaTransportError(context.Background(), err); outcome != replicaFailed {
		t.Fatalf("a dial timeout is %s", outcome)
	}
}
