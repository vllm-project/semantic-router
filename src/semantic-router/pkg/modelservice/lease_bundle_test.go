package modelservice

import (
	"context"
	"errors"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func classifyHead(id string) runtimetest.Model {
	return runtimetest.Model{ID: id, Heads: []runtimetest.Head{{Name: "default", Kind: "sequence", Labels: []string{"math", "law", "other"}}}}
}

// attachedLease leases deployments served by fake runtime processes; each
// runtime serves the deployments listed for it.
func attachedLease(t *testing.T, runtimes map[*runtimetest.Runtime][]string) *Lease {
	t.Helper()
	deployments := map[string]config.ModelDeployment{}
	cfg := &config.RouterConfig{}
	for runtime, names := range runtimes {
		server := httptest.NewServer(runtime.Handler())
		t.Cleanup(server.Close)
		for _, name := range names {
			deployments[name] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: server.URL}
			cfg.DecisionRules = append(cfg.DecisionRules, config.DecisionSignalRule{Name: "r_" + name, Deployment: name, Question: config.DecisionQuestion{Type: "noul", Instructions: "x"}})
		}
	}
	cfg.ModelDeployments = deployments
	manager := NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.Acquire(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for name := range deployments {
		waitReady(t, lease, name)
	}
	return lease
}

func classifyText(text string) ClassifyRequest {
	return ClassifyRequest{Inputs: []ClassifyInput{{Text: text}}}
}

func TestBundleSendsOneCallPerLogicalDeploymentWhenEveryParticipantParks(t *testing.T) {
	encoders := runtimetest.New(classifyHead("domain"), classifyHead("guard"), classifyHead("feedback"))
	decisions := runtimetest.New(runtimetest.Model{ID: "kai"})
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{encoders: {"domain", "guard", "feedback"}, decisions: {"kai"}})
	ctx, bundle := WithBundle(context.Background(), 50*time.Millisecond)
	var wg sync.WaitGroup
	errs := make(chan error, 4)
	for _, deployment := range []string{"domain", "guard", "feedback"} {
		leave := bundle.Join()
		wg.Add(1)
		go func(deployment string) {
			defer wg.Done()
			defer leave()
			response, err := lease.Classify(ctx, deployment, classifyText("is this math"))
			if err == nil && response.Results[0].Label != "math" {
				err = errors.New("wrong label " + response.Results[0].Label)
			}
			errs <- err
		}(deployment)
	}
	leave := bundle.Join()
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer leave()
		_, err := lease.Decide(ctx, "kai", sampleRequest("is this hard"))
		errs <- err
	}()
	started := time.Now()
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if elapsed := time.Since(started); elapsed > 40*time.Millisecond {
		t.Fatalf("an idle bundle flushes at once, not at its window: %v", elapsed)
	}
	if calls, tasks := encoders.Bundles(); calls != 3 || tasks != 3 || encoders.Calls("classify") != 0 {
		t.Fatalf("independent encoder deployments: %d bundles with %d tasks, %d direct calls", calls, tasks, encoders.Calls("classify"))
	}
	if calls, tasks := decisions.Bundles(); calls != 1 || tasks != 1 {
		t.Fatalf("decision process: %d bundles with %d tasks", calls, tasks)
	}
	if bundle.Flushes() != 1 {
		t.Fatalf("one stage, one flush: %d", bundle.Flushes())
	}
}

func TestBundleWaitsForARunningParticipantAtMostItsWindow(t *testing.T) {
	encoders := runtimetest.New(classifyHead("domain"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{encoders: {"domain"}})
	ctx, bundle := WithBundle(context.Background(), 5*time.Millisecond)
	slow := bundle.Join()
	defer slow()
	leave := bundle.Join()
	started := time.Now()
	if _, err := lease.Classify(ctx, "domain", classifyText("law")); err != nil {
		t.Fatal(err)
	}
	leave()
	if elapsed := time.Since(started); elapsed < 5*time.Millisecond || elapsed > 200*time.Millisecond {
		t.Fatalf("a participant that never parks delays the flush by the window only: %v", elapsed)
	}
}

func TestFanCountsTheCallerAsBlockedWhileItsWorkParks(t *testing.T) {
	encoders := runtimetest.New(classifyHead("domain"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{encoders: {"domain"}})
	ctx, bundle := WithBundle(context.Background(), time.Second)
	leave := bundle.Join()
	texts := []string{"math one", "law two", "other three"}
	labels := make([]string, len(texts))
	started := time.Now()
	Fan(ctx, len(texts), func(i int) {
		response, err := lease.Classify(ctx, "domain", classifyText(texts[i]))
		if err == nil {
			labels[i] = response.Results[0].Label
		}
	})
	leave()
	if labels[0] != "math" || labels[1] != "law" || labels[2] != "other" {
		t.Fatalf("labels = %v", labels)
	}
	if calls, tasks := encoders.Bundles(); calls != 1 || tasks != 1 || time.Since(started) > 500*time.Millisecond {
		t.Fatalf("a fan-out of three calls to one head is one bundle of one fused task: %d calls, %d tasks", calls, tasks)
	}
}

func TestResultCacheServesRepeatedClassificationsAndDecisions(t *testing.T) {
	encoders := runtimetest.New(classifyHead("domain"))
	decisions := runtimetest.New(runtimetest.Model{ID: "kai"})
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{encoders: {"domain"}, decisions: {"kai"}})
	for range 3 {
		if _, err := lease.Classify(context.Background(), "domain", classifyText("math")); err != nil {
			t.Fatal(err)
		}
		if _, err := lease.Decide(context.Background(), "kai", sampleRequest("hard")); err != nil {
			t.Fatal(err)
		}
	}
	if encoders.Calls("classify") != 1 || decisions.Calls("decisions") != 1 {
		t.Fatalf("repeats are cache hits: %d classify, %d decisions", encoders.Calls("classify"), decisions.Calls("decisions"))
	}
	if _, err := lease.Classify(context.Background(), "domain", ClassifyRequest{Inputs: []ClassifyInput{{Text: "math"}}, Overflow: "truncate"}); err != nil {
		t.Fatal(err)
	}
	if encoders.Calls("classify") != 2 {
		t.Fatal("options are part of the cache key")
	}
}

func TestCallsCarryTheirDeadline(t *testing.T) {
	encoders := runtimetest.New(classifyHead("domain"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{encoders: {"domain"}})
	encoders.SetDelay(200 * time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := lease.Classify(ctx, "domain", classifyText("late")); ErrorReason(err) != "timeout" {
		t.Fatalf("a late answer is a timeout, got %v", err)
	}
	bundled, bundle := WithBundle(ctx, time.Millisecond)
	leave := bundle.Join()
	defer leave()
	if _, err := lease.Classify(bundled, "domain", classifyText("later")); ErrorReason(err) != "timeout" {
		t.Fatalf("a bundled late answer is a timeout, got %v", err)
	}
}
