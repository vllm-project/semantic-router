package serving_test

import (
	"context"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// TestServingOverALeaseBundlesEverySignalOfAStage prepares three bindings on
// one runtime process and checks that one request stage sends one bundle.
func TestServingOverALeaseBundlesEverySignalOfAStage(t *testing.T) {
	fake := runtimetest.New(
		runtimetest.Model{ID: "domain", Heads: []runtimetest.Head{{Name: "default", Kind: "sequence", Labels: []string{"math", "law", "other"}}}},
		runtimetest.Model{ID: "pii", Heads: []runtimetest.Head{{Name: "default", Kind: "token", Labels: []string{"O", "B-PERSON", "I-PERSON", "B-EMAIL", "I-EMAIL"}}}},
		runtimetest.Model{ID: "guard", Heads: []runtimetest.Head{{Name: "default", Kind: "sequence", Labels: []string{"benign", "jailbreak"}}}},
	)
	server := httptest.NewServer(fake.Handler())
	defer server.Close()
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{}
	cfg.ModelBindings = map[string]config.ModelBinding{}
	consumers := map[string]string{"domain": "domain_classifier", "pii": "pii_classifier", "guard": "prompt_guard"}
	contracts := map[string]string{"domain": config.RemoteClassifierContractLabelDistribution, "pii": config.RemoteClassifierContractTokenSpans, "guard": config.RemoteClassifierContractLabelDistribution}
	for deployment, consumer := range consumers {
		cfg.ModelDeployments[deployment] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: server.URL}
		cfg.ModelBindings[consumer] = config.ModelBinding{Deployment: deployment, Contract: contracts[deployment]}
	}
	cfg.Decisions = []config.Decision{{Name: "all", Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{
		{Type: config.SignalTypeDomain, Name: "math"}, {Type: config.SignalTypePII, Name: "any"}, {Type: config.SignalTypeJailbreak, Name: "jb"},
	}}}}
	manager := modelservice.NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	lease, err := manager.Acquire(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(lease.Deployments()) != 3 {
		t.Fatalf("every active consumer's deployment is leased: %v", lease.Deployments())
	}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	runtime := serving.New(lease, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	lookup := func(name string) config.ResolvedModelBinding {
		spec, ok := plan.Lookup(config.DefaultRecipeName, name)
		if !ok {
			t.Fatalf("binding %s is missing", name)
		}
		return spec
	}
	domain, err := runtime.Sequence(ctx, lookup("domain_classifier"))
	if err != nil {
		t.Fatal(err)
	}
	guard, err := runtime.Sequence(ctx, lookup("prompt_guard"))
	if err != nil {
		t.Fatal(err)
	}
	pii, err := runtime.Tokens(ctx, lookup("pii_classifier"))
	if err != nil {
		t.Fatal(err)
	}
	before, _ := fake.Bundles()
	stage, bundle := modelservice.WithBundle(ctx, 50*time.Millisecond)
	text := "my person asks about math"
	var (
		wg       sync.WaitGroup
		labels   tasks.LabelDistribution
		jailbook tasks.LabelDistribution
		spans    tasks.TokenClassificationResult
		errs     [3]error
	)
	run := func(i int, call func()) {
		leave := bundle.Join()
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer leave()
			call()
		}()
	}
	run(0, func() { labels, errs[0] = domain.Call(stage, "default", text) })
	run(1, func() { jailbook, errs[1] = guard.Call(stage, "default", text) })
	run(2, func() { spans, errs[2] = pii.Call(stage, "default", text) })
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if labels.Probabilities[0] < 0.5 || jailbook.Probabilities[1] > 0.5 || len(spans.Entities) != 1 || text[spans.Entities[0].Start:spans.Entities[0].End] != "person" {
		t.Fatalf("labels %v, guard %v, spans %+v", labels.Probabilities, jailbook.Probabilities, spans.Entities)
	}
	if after, tasks := fake.Bundles(); after-before != 1 || tasks < 3 {
		t.Fatalf("one stage sends one bundle: %d bundles, %d tasks", after-before, tasks)
	}
}
