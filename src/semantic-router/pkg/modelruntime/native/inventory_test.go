package native

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestPreparedInventoryFollowsNativeWarmupAndRetirement(t *testing.T) {
	artifact := nativeHeadlessFullFixture(t, 1)
	spec := config.ResolvedModelBinding{
		Recipe: "active", Name: "domain_classifier",
		Binding: config.ModelBinding{Deployment: "fixture", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelDistribution},
		Deployment: config.ModelDeployment{
			Artifact: artifact, Provider: "candle", Device: "cpu", Precision: "native",
			Input: config.ModelInputBudget{MaxTokens: 512, Overflow: "reject"},
		},
	}
	pool := binding.NewPool()
	current, candidate := New(pool), New(pool)
	handle, err := current.Sequence(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handle.Close() })
	got := current.PreparedBindings()
	if len(got) != 1 || got[0].Artifact != artifact || got[0].Identity.Recipe != "active" || got[0].Capability.Device != "cpu" || got[0].Capability.Provider != "candle" {
		t.Fatalf("native warmup evidence missing: %+v", got)
	}
	invalid := spec
	invalid.Deployment.Input.MaxTokens = 1024
	if failed, err := candidate.Sequence(context.Background(), invalid); err == nil {
		_ = failed.Close()
		t.Fatal("fixture must reject the candidate's unsupported input budget")
	}
	if len(candidate.PreparedBindings()) != 0 || len(current.PreparedBindings()) != 1 {
		t.Fatal("failed preparation changed the published inventory")
	}
	if err := handle.Close(); err != nil {
		t.Fatal(err)
	}
	if len(current.PreparedBindings()) != 0 {
		t.Fatal("retired native task is still visible")
	}
}
