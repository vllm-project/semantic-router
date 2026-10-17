package classification

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// TestSignalStageSendsOneBundleWithoutWaitingForHeuristics runs three
// model-backed signals and a slow heuristic one: the model calls reach the
// runtime as one bundle as soon as they have all parked.
func TestSignalStageSendsOneBundleWithoutWaitingForHeuristics(t *testing.T) {
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"signals": {Heads: []runtimetest.Head{
			{Name: "domain", Kind: "sequence", Labels: []string{"math", "law"}},
			{Name: "guard", Kind: "sequence", Labels: []string{"benign", "jailbreak"}},
			{Name: "feedback", Kind: "sequence", Labels: []string{"satisfied", "wrong_answer"}},
		}},
	})
	ctx := context.Background()
	call := make(map[string]func(context.Context) error)
	for _, deployment := range []string{"domain", "guard", "feedback"} {
		spec := config.ResolvedModelBinding{
			Recipe: config.DefaultRecipeName, Name: deployment + "_consumer",
			Binding:    config.ModelBinding{Deployment: "signals", Head: deployment, Contract: config.RemoteClassifierContractLabelDistribution},
			Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://fake", Input: config.ModelInputBudget{Overflow: "reject"}},
		}
		handle, err := runtime.Sequence(ctx, spec)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = handle.Close() })
		call[deployment] = func(stage context.Context) error {
			_, err := handle.Call(stage, string(config.DefaultRecipeName), "is this about math")
			return err
		}
	}
	before, _ := fake.Bundles()
	stage, bundle := modelservice.WithBundle(ctx, 50*time.Millisecond)
	var mu sync.Mutex
	finished := map[string]time.Duration{}
	started := time.Now()
	record := func(name string, err error) {
		if err != nil {
			t.Error(err)
		}
		mu.Lock()
		finished[name] = time.Since(started)
		mu.Unlock()
	}
	dispatchers := []signalDispatch{
		{config.SignalTypeDomain, "Domain", func(ctx context.Context) { record("domain", call["domain"](ctx)) }},
		{config.SignalTypeJailbreak, "Jailbreak", func(ctx context.Context) { record("guard", call["guard"](ctx)) }},
		{config.SignalTypeUserFeedback, "User feedback", func(ctx context.Context) { record("feedback", call["feedback"](ctx)) }},
		{config.SignalTypeKeyword, "Keyword", func(context.Context) { time.Sleep(200 * time.Millisecond) }},
	}
	used := map[string]bool{"domain:a": true, "jailbreak:a": true, "user_feedback:a": true, "keyword:a": true}
	ready := map[string]bool{config.SignalTypeDomain: true, config.SignalTypeJailbreak: true, config.SignalTypeUserFeedback: true, config.SignalTypeKeyword: true}
	var wg sync.WaitGroup
	runSignalDispatchers(stage, dispatchers, used, ready, bundle, func(string) []string { return nil }, &wg)
	wg.Wait()
	after, tasks := fake.Bundles()
	if after-before != 1 || tasks != 3 || bundle.Flushes() != 1 {
		t.Fatalf("one stage must send one bundle of three tasks: %d bundles, %d tasks, %d flushes", after-before, tasks, bundle.Flushes())
	}
	for name, elapsed := range finished {
		if elapsed > 150*time.Millisecond {
			t.Fatalf("%s waited for the heuristic signal: %v", name, elapsed)
		}
	}
}
