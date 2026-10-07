package serving_test

import (
	"context"
	"errors"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// vela2Lease serves a Vela 2.0-style model (deployment "vela") and a System
// One decision model (deployment "kai") from one fake runtime process.
func vela2Lease(t *testing.T) (*modelservice.Lease, *runtimetest.Runtime, string) {
	t.Helper()
	fake := runtimetest.New(
		runtimetest.Model{ID: "vela", Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON", "EMAIL_ADDRESS"}, BroadHead: true}},
		runtimetest.Model{ID: "kai"},
	)
	server := httptest.NewServer(fake.Handler())
	t.Cleanup(server.Close)
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{
		"vela": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
		"kai":  {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	return lease, fake, server.URL
}

func spanBinding(consumer, deployment, endpoint string) config.ResolvedModelBinding {
	return config.ResolvedModelBinding{
		Recipe: config.DefaultRecipeName, Name: consumer,
		Binding:    config.ModelBinding{Deployment: deployment, Contract: config.RemoteClassifierContractTokenSpans},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: endpoint}.WithDefaults(),
	}
}

func TestPIIBindingAsksTheReadyMadeQuestionOfADecisionModel(t *testing.T) {
	lease, fake, endpoint := vela2Lease(t)
	runtime := serving.New(lease, nil)
	spec := spanBinding("pii_classifier", "vela", endpoint)
	labels, err := runtime.Labels(context.Background(), spec)
	if err != nil || labels != nil {
		t.Fatalf("a ready-made span question serves no label list: %v %v", labels, err)
	}
	handle, err := runtime.Tokens(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	if capability := handle.Capability(); capability.Preset != "pii" || len(capability.Labels) != 0 {
		t.Fatalf("capability = %+v", capability)
	}
	text := "Grüße an person und email_address"
	result, err := handle.Call(context.Background(), string(config.DefaultRecipeName), text)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Entities) != 2 || result.Entities[0].EntityType != "PERSON" || text[result.Entities[0].Start:result.Entities[0].End] != "person" ||
		result.Entities[1].EntityType != "EMAIL_ADDRESS" || !result.HasScores() {
		t.Fatalf("spans become byte-offset entities: %+v", result.Entities)
	}
	if fake.Calls("classify") != 0 {
		t.Fatal("a decision model is never asked for a classify head")
	}

	// In a request stage, the PII question and the decision signals asked of
	// the same deployment about the same text travel in one task.
	signals := modelservice.Request{State: text, Questions: []modelservice.Question{{ID: "urgent", Type: "noul", Instructions: "Urgent?"}}}
	ctx, bundle := modelservice.WithBundle(context.Background(), time.Second)
	var wg sync.WaitGroup
	var decided modelservice.Response
	var spans tasks.TokenClassificationResult
	var decideErr, spanErr error
	for _, call := range []func(){
		func() { decided, decideErr = lease.Decide(ctx, "vela", signals) },
		func() { spans, spanErr = handle.Call(ctx, string(config.DefaultRecipeName), text) },
	} {
		leave := bundle.Join()
		wg.Add(1)
		go func(call func()) {
			defer wg.Done()
			defer leave()
			call()
		}(call)
	}
	wg.Wait()
	if decideErr != nil || spanErr != nil || decided.Answers["urgent"].Type != "noul" || len(spans.Entities) != 2 {
		t.Fatalf("fused answers: %+v %v / %+v %v", decided, decideErr, spans, spanErr)
	}
	if calls, tasksSent := fake.Bundles(); calls != 1 || tasksSent != 1 {
		t.Fatalf("one stage, one task for the deployment's questions: %d bundles, %d tasks", calls, tasksSent)
	}
}

func TestHallucinationBindingAsksTheReadyMadeQuestionAboutTheAnswer(t *testing.T) {
	lease, _, endpoint := vela2Lease(t)
	handle, err := serving.New(lease, nil).Grounded(context.Background(), spanBinding("hallucination_detector", "vela", endpoint), 0.4)
	if err != nil {
		t.Fatal(err)
	}
	result, err := handle.Call(context.Background(), string(config.DefaultRecipeName), tasks.GroundedTextRequest{Context: "Ann met Bob in Paris", Question: "Who did Ann meet?", Answer: "Ann met Tom"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Entities) != 1 || result.Entities[0].Text != "Tom" || result.Entities[0].EntityType != "unsupported" || result.Summary == nil {
		t.Fatalf("unsupported spans refer to the answer: %+v", result)
	}
	clean, err := handle.Call(context.Background(), string(config.DefaultRecipeName), tasks.GroundedTextRequest{Context: "Ann met Bob", Answer: "Ann met Bob"})
	if err != nil || len(clean.Entities) != 0 || clean.Summary != nil {
		t.Fatalf("a supported answer has no spans and no aggregate: %+v %v", clean, err)
	}
}

func TestSpanBindingsFailPreparationTheModelCannotServe(t *testing.T) {
	lease, _, endpoint := vela2Lease(t)
	runtime := serving.New(lease, nil)
	cases := map[string]config.ResolvedModelBinding{
		"a model without the preset": spanBinding("pii_classifier", "kai", endpoint),
		"the broad head": func() config.ResolvedModelBinding {
			spec := spanBinding("pii_classifier", "vela", endpoint)
			spec.Binding.Head = config.DecisionSpanHeadBroad
			return spec
		}(),
		"a label mapping": func() config.ResolvedModelBinding {
			spec := spanBinding("pii_classifier", "vela", endpoint)
			spec.Binding.MappingPath = "pii_mapping.json"
			return spec
		}(),
		"an input budget": func() config.ResolvedModelBinding {
			spec := spanBinding("pii_classifier", "vela", endpoint)
			spec.Deployment.Input = config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}
			return spec
		}(),
	}
	for name, spec := range cases {
		if _, err := runtime.Tokens(context.Background(), spec); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("%s: expected a capability error, got %v", name, err)
		}
	}
	router := spanBinding("pii_classifier", "vela", endpoint)
	router.Binding.Head = config.DecisionSpanHeadRouter
	if _, err := runtime.Tokens(context.Background(), router); err != nil {
		t.Fatalf("head router names the head that answers: %v", err)
	}
	if _, err := runtime.Sequence(context.Background(), spanBinding("classifier.topics", "vela", endpoint)); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("a custom classifier has no question to ask, got %v", err)
	}
}
