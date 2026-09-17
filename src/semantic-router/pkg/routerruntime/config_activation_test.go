package routerruntime

import (
	"errors"
	"testing"
)

func TestConfigActivationDoesNotAttributeStaleCompletionToNewDocument(t *testing.T) {
	r := NewRegistry(nil)
	old := r.BeginConfigActivation("old", "file")
	current := r.BeginConfigActivation("new", "file")
	r.SetConfigActivationStage(old, "warmup")
	r.FinishConfigActivation(old, "failed", errors.New("old failure"))
	state := r.ConfigActivation()
	if state.Attempt != current || state.DocumentHash != "new" || state.Status != "preparing" || state.FailureDetail != "" {
		t.Fatalf("stale attempt changed current activation: %+v", state)
	}
	r.SetConfigActivationStage(current, "model_prepare")
	r.FinishConfigActivation(current, "failed", errors.New("model unavailable"))
	state = r.ConfigActivation()
	if state.Status != "failed" || state.Stage != "model_prepare" || state.FinishedAt == nil || state.FailureDetail != "model unavailable" {
		t.Fatalf("failure state not retained: %+v", state)
	}
	r.FinishConfigActivation(current, "active", nil)
	if r.ConfigActivation().Status != "failed" {
		t.Fatal("a completed attempt was overwritten")
	}
	*state.FinishedAt = state.StartedAt
	if r.ConfigActivation().FinishedAt.Equal(state.StartedAt) {
		t.Fatal("snapshot shares mutable timestamp with registry")
	}
}
