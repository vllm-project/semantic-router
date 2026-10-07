package classification

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// TestAStageOfManyPIIPiecesFitsTheRuntimeLimits fans PII out the way a long
// prompt without a window or a long include_history conversation does: one
// call per piece, more than the runtime takes in one bundle. Every piece gets
// its own spans, and the stage reaches the runtime as one bundle of one task.
func TestAStageOfManyPIIPiecesFitsTheRuntimeLimits(t *testing.T) {
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"pii": {Heads: []runtimetest.Head{{Name: "default", Kind: "token", Labels: []string{"O", "B-EMAIL", "I-EMAIL"}}}},
	})
	ctx := context.Background()
	handle, err := runtime.Tokens(ctx, config.ResolvedModelBinding{
		Recipe: config.DefaultRecipeName, Name: "pii_classifier",
		Binding:    config.ModelBinding{Deployment: "pii", Contract: config.RemoteClassifierContractTokenSpans},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://fake", Input: config.ModelInputBudget{Overflow: "reject"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handle.Close() })
	before, beforeTasks := fake.Bundles()
	stage, bundle := modelservice.WithBundle(ctx, 50*time.Millisecond)
	leave := bundle.Join()
	pieces := 4 * modelservice.DefaultBundleTasks
	entities, errs := make([]int, pieces), make([]error, pieces)
	modelservice.Fan(stage, pieces, func(i int) {
		result, err := handle.Call(stage, string(config.DefaultRecipeName), fmt.Sprintf("message %d asks to email Bob", i))
		entities[i], errs[i] = len(result.Entities), err
	})
	leave()
	for i := range errs {
		if errs[i] != nil || entities[i] != 1 {
			t.Fatalf("piece %d: %d entities, %v", i, entities[i], errs[i])
		}
	}
	after, tasks := fake.Bundles()
	if after-before != 1 || tasks-beforeTasks != 1 {
		t.Fatalf("%d pieces went out as %d bundles of %d tasks, want one bundle of one fused task", pieces, after-before, tasks-beforeTasks)
	}
}
