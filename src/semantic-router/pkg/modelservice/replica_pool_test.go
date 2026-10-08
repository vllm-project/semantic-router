package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

const poolArtifact = "vllm-sr/Decision-2.0-Kai-0.6B"

func testReplicaPool(t *testing.T, models ...runtimetest.Model) (*Lease, []*runtimetest.Runtime) {
	t.Helper()
	declaration := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: poolArtifact}
	var runtimes []*runtimetest.Runtime
	for _, model := range models {
		if model.Repo == "" {
			model.Repo = poolArtifact
		}
		if model.ModelSHA256 == "" {
			model.ModelSHA256 = strings.Repeat("a", 64)
		}
		runtime := runtimetest.New(model)
		runtimes = append(runtimes, runtime)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/v1/systemone" {
				r.URL.Path = "/v1/decisions"
			}
			runtime.Handler().ServeHTTP(w, r)
		}))
		t.Cleanup(server.Close)
		declaration.Replicas = append(declaration.Replicas, config.ModelReplica{Endpoint: server.URL, ServedName: model.ID})
	}
	manager := NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"primary": declaration})
	if err != nil {
		t.Fatal(err)
	}
	// Observe every worker before testing compatibility, not just the first one.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		all := true
		for _, g := range lease.groups {
			for _, served := range g.models {
				all = all && served.ready.Load()
			}
		}
		if all {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	return lease, runtimes
}

func TestReplicaPoolFusesBeforeChoosingWorker(t *testing.T) {
	lease, runtimes := testReplicaPool(t, runtimetest.Model{ID: "worker-one", Joint: true}, runtimetest.Model{ID: "worker-two", Joint: true})
	var responses [2]Response
	bundle, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"primary"}, ask: func(ctx context.Context) (err error) {
			responses[0], err = lease.Decide(ctx, "primary", Request{State: "same state", Questions: []Question{choiceQuestion("domain")}})
			return
		}},
		asker{deployments: []string{"primary"}, delay: 20 * time.Millisecond, ask: func(ctx context.Context) (err error) {
			responses[1], err = lease.Decide(ctx, "primary", Request{State: "same state", Questions: []Question{choiceQuestion("preference")}})
			return
		}},
	)
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
	calls := append(runtimes[0].Decisions(), runtimes[1].Decisions()...)
	if len(calls) != 1 || len(calls[0].Questions) != 2 || bundle.Flushes() != 1 {
		t.Fatalf("stage split before replica dispatch: calls=%+v flushes=%d", calls, bundle.Flushes())
	}
	for i, id := range []string{"domain", "preference"} {
		if responses[i].Model != "primary" || math.Abs(responses[i].Answers[id].Probabilities["a"]-0.65) > 1e-9 {
			t.Fatalf("fusion semantics or logical identity lost: %+v", responses[i])
		}
	}
	status := lease.Statuses()[0]
	if status.ReadyReplicas != 2 || len(status.Replicas) != 2 || strings.Contains(status.Replicas[0].ID, "127.") {
		t.Fatalf("pool inventory=%+v", status)
	}
}

func TestReplicaPoolBalancesOutstandingWorkAndBoundsAdmission(t *testing.T) {
	lease, _ := testReplicaPool(t, runtimetest.Model{ID: "a"}, runtimetest.Model{ID: "b"})
	pool := lease.members["primary"].pool
	first, releaseFirst, err := pool.retain(100)
	if err != nil {
		t.Fatal(err)
	}
	defer releaseFirst(false)
	second, releaseSecond, err := pool.retain(10)
	if err != nil {
		t.Fatal(err)
	}
	defer releaseSecond(false)
	if first.id == second.id {
		t.Fatal("outstanding work did not choose the idle replica")
	}
	releases := []func(bool){releaseFirst, releaseSecond}
	for i := 2; i < 2*replicaAdmissionLimit; i++ {
		_, release, err := pool.retain(1)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	if _, _, err := pool.retain(1); !errors.Is(err, ErrOverloaded) {
		t.Fatalf("unbounded admission: %v", err)
	}
	for _, release := range releases {
		release(false)
	}
	for _, replica := range pool.status().Replicas {
		if replica.Inflight != 0 || replica.EstimatedWork != 0 {
			t.Fatalf("admission not released: %+v", replica)
		}
	}
}

func TestReplicaPoolRejectsArtifactAndCapabilityMismatch(t *testing.T) {
	for _, mismatch := range []runtimetest.Model{{ID: "wrong", Repo: "another/model"}, {ID: "wrong", MaxInputTokens: 4096}, {ID: "wrong", ModelSHA256: strings.Repeat("b", 64)}} {
		lease, runtimes := testReplicaPool(t, runtimetest.Model{ID: "good"}, mismatch)
		// A compatible baseline is selected deterministically by preparation here.
		pool := lease.members["primary"].pool
		for _, w := range pool.workers {
			if w.member.served.name == "good" {
				card, err := pool.workerCard(w)
				if err != nil {
					t.Fatal(err)
				}
				pool.baseline = &card
			}
		}
		if _, err := lease.Decide(context.Background(), "primary", sampleRequest("x")); err != nil {
			t.Fatal(err)
		}
		if runtimes[1].Calls("decisions") != 0 || len(runtimes[1].Decisions()) != 0 {
			t.Fatal("heterogeneous worker received inference")
		}
		if status := lease.Statuses()[0]; status.ReadyReplicas != 1 || status.State != "degraded" {
			t.Fatalf("mismatch hidden: %+v", status)
		}
	}
}

func TestReplicaPoolDrainsAcceptedExchangeAcrossLeaseClose(t *testing.T) {
	started, finish := make(chan struct{}), make(chan struct{})
	runtime := runtimetest.New(runtimetest.Model{ID: "worker", Repo: poolArtifact, ModelSHA256: strings.Repeat("a", 64)})
	var once sync.Once
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/decisions" {
			once.Do(func() { close(started) })
			<-finish
		}
		runtime.Handler().ServeHTTP(w, r)
	}))
	defer server.Close()
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Artifact: poolArtifact, Replicas: []config.ModelReplica{{Endpoint: server.URL, ServedName: "worker"}}}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if _, err := lease.Card(ctx, "primary"); err != nil {
		t.Fatal(err)
	}
	completed := make(chan error, 1)
	go func() { _, err := lease.Decide(ctx, "primary", sampleRequest("drain me")); completed <- err }()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if err := lease.Close(); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	remaining := len(manager.groups)
	manager.mu.Unlock()
	if remaining != 1 {
		t.Fatalf("accepted worker stopped during reload: %d", remaining)
	}
	close(finish)
	if err := <-completed; err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	remaining = len(manager.groups)
	manager.mu.Unlock()
	if remaining != 0 {
		t.Fatalf("drained worker leaked: %d", remaining)
	}
}

func TestReplicaPoolSystemOnePreservesNativePayloadAndLogicalIdentity(t *testing.T) {
	lease, _ := testReplicaPool(t, runtimetest.Model{ID: "internal-worker"})
	result, err := lease.SystemOne(context.Background(), "primary", json.RawMessage(`{"state":"hello","questions":{"q":{"type":"noul","instructions":"Relevant?"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	var response struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
	}
	if err = json.Unmarshal(result.Body, &response); err != nil {
		t.Fatal(err)
	}
	if response.Model != "primary" || len(response.Answers) != 1 {
		t.Fatalf("native result = %s", result.Body)
	}
}

func TestReplicaPoolBackoffAndNilBodyAreExplicit(t *testing.T) {
	lease, _ := testReplicaPool(t, runtimetest.Model{ID: "worker"})
	pool := lease.members["primary"].pool
	_, release, err := pool.retain(1)
	if err != nil {
		t.Fatal(err)
	}
	release(true)
	status := pool.status()
	if status.Ready || status.ReadyReplicas != 0 || status.Replicas[0].Ready || status.Replicas[0].State != "backoff" {
		t.Fatalf("backoff availability inconsistent: %+v", status)
	}
	if _, _, err = pool.retain(1); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("cooling worker admitted: %v", err)
	}
	response, err := pool.Do(httptest.NewRequest(http.MethodGet, "http://model-runtime/v1/models", nil))
	if response != nil {
		_ = response.Body.Close()
	}
	if !errors.Is(err, ErrRejected) {
		t.Fatalf("nil body got %v", err)
	}
}

func TestReplicaPoolSinglePlacementShorthandKeepsCacheAndInventory(t *testing.T) {
	runtime := runtimetest.New(runtimetest.Model{ID: "primary"})
	server := httptest.NewServer(runtime.Handler())
	defer server.Close()
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	implicit := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: server.URL}
	first, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"primary": implicit})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err = first.Card(ctx, "primary"); err != nil {
		t.Fatal(err)
	}
	explicit := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Replicas: []config.ModelReplica{{Endpoint: server.URL}}}
	second, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"primary": explicit})
	if err != nil {
		t.Fatal(err)
	}
	if first.members["primary"].group != second.members["primary"].group {
		t.Fatal("equivalent placement shorthand restarted worker")
	}
	for _, lease := range []*Lease{first, second} {
		if _, err = lease.Decide(ctx, "primary", sampleRequest("identical cacheable request")); err != nil {
			t.Fatal(err)
		}
		status := lease.Statuses()[0]
		if status.DesiredReplicas != 1 || status.ReadyReplicas != 1 || len(status.Replicas) != 1 || !status.Replicas[0].Ready {
			t.Fatalf("single worker observations missing: %+v", status)
		}
	}
	if len(runtime.Decisions()) != 1 {
		t.Fatalf("single placement changed cache semantics: %d calls", len(runtime.Decisions()))
	}
	if err = first.Close(); err != nil {
		t.Fatal(err)
	}
	if !second.Statuses()[0].Ready {
		t.Fatal("closing old lease disabled retained worker")
	}
}

func TestReplicaPoolOverlappingGenerationsShareAdmissionAndDrainState(t *testing.T) {
	first, _ := testReplicaPool(t, runtimetest.Model{ID: "one"}, runtimetest.Model{ID: "two"})
	original := first.members["primary"].pool
	second, err := first.manager.AcquireDeployments(map[string]config.ModelDeployment{"primary": original.declaration})
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	current := second.members["primary"].pool
	oldWorker, releaseOld, err := original.retain(100)
	if err != nil {
		t.Fatal(err)
	}
	defer releaseOld(false)
	newWorker, releaseNew, err := current.retain(1)
	if err != nil {
		t.Fatal(err)
	}
	defer releaseNew(false)
	if oldWorker.id == newWorker.id {
		t.Fatal("new generation ignored prior generation's outstanding work")
	}
	if err = first.Close(); err != nil {
		t.Fatal(err)
	}
	status := second.Statuses()[0]
	if status.ReadyReplicas != 2 {
		t.Fatalf("closing old generation stopped active workers: %+v", status)
	}
	if status.Replicas[0].Inflight+status.Replicas[1].Inflight != 2 {
		t.Fatalf("outstanding work not shared: %+v", status.Replicas)
	}
}

func TestReplicaTransportRejectsNullObjectsWithoutPanic(t *testing.T) {
	for _, body := range []string{`null`, `[]`, `{"tasks":[{"decisions":null}]}`, `{"tasks":[{"classify":[]}]}`} {
		if _, err := rewriteExchangeModel([]byte(body), "worker", strings.Contains(body, "tasks")); !errors.Is(err, ErrRejected) {
			t.Fatalf("invalid transport body %s got %v", body, err)
		}
	}
}

func TestReplicaTransportNativeModelCannotBeBypassedWithUnknownTasks(t *testing.T) {
	body, err := rewriteExchangeModel([]byte(`{"model":"unauthorized","state":"x","questions":{},"tasks":[{"decisions":{"model":"other"}}]}`), "authorized", false)
	if err != nil {
		t.Fatal(err)
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(body, &object) != nil || string(object["model"]) != `"authorized"` {
		t.Fatalf("native target was not pinned: %s", body)
	}
}
