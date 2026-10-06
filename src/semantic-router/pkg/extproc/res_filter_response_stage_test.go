package extproc

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func responseStageBinding(name, deployment, contract string) config.ResolvedModelBinding {
	return config.ResolvedModelBinding{
		Recipe: config.DefaultRecipeName, Name: name,
		Binding:    config.ModelBinding{Deployment: deployment, Contract: contract},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://fake", Input: config.ModelInputBudget{Overflow: "reject"}},
	}
}

// TestResponseStageSendsOneBundle runs the response guard and the
// hallucination detector on one answer: both model calls reach the runtime as
// one /v1/bundle, and every check publishes afterwards, in order.
func TestResponseStageSendsOneBundle(t *testing.T) {
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"guard": servingtest.Sequence("benign", "jailbreak"),
		"halu":  servingtest.Grounded(),
	})
	ctx := context.Background()
	recipe := string(config.DefaultRecipeName)
	guard, err := runtime.Sequence(ctx, responseStageBinding("prompt_guard", "guard", config.RemoteClassifierContractLabelDistribution))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = guard.Close() })
	halu, err := runtime.Grounded(ctx, responseStageBinding("hallucination_detector", "halu", config.RemoteClassifierContractTokenSpans), 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = halu.Close() })

	answer := "The tower is 450 meters tall."
	var (
		published         []string
		guardErr, haluErr error
		spans             tasks.TokenClassificationResult
	)
	before, beforeTasks := fake.Bundles()
	runResponseStage(ctx,
		responseStageCheck{
			run:     func(call context.Context) { _, guardErr = guard.Call(call, recipe, answer) },
			publish: func() { published = append(published, "guard") },
		},
		responseStageCheck{publish: func() { published = append(published, "unbacked") }},
		responseStageCheck{
			run: func(call context.Context) {
				spans, haluErr = halu.Call(call, recipe, tasks.GroundedTextRequest{Context: "The tower is 330 meters tall.", Question: "How tall is it?", Answer: answer})
			},
			publish: func() { published = append(published, "halu") },
		},
	)
	after, afterTasks := fake.Bundles()

	if guardErr != nil || haluErr != nil {
		t.Fatalf("guard: %v, halu: %v", guardErr, haluErr)
	}
	if after-before != 1 || afterTasks-beforeTasks != 2 {
		t.Fatalf("the response stage must send one bundle of two tasks: %d bundles, %d tasks", after-before, afterTasks-beforeTasks)
	}
	if len(spans.Entities) == 0 {
		t.Fatal("the grounded answer lost its unsupported span")
	}
	if want := []string{"guard", "unbacked", "halu"}; !reflect.DeepEqual(published, want) {
		t.Fatalf("published %v, want %v", published, want)
	}
}

func TestResponseStagePublishesChecksWithoutModelCalls(t *testing.T) {
	var published int
	runResponseStage(context.Background(),
		responseStageCheck{publish: func() { published++ }},
		responseStageCheck{},
	)
	if published != 1 {
		t.Fatalf("published %d checks, want 1", published)
	}
}
