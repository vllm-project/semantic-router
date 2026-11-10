package routing

import (
	"context"
	"testing"
)

func TestListenerModelsRestrictOnlyWhenSet(t *testing.T) {
	if _, ok := ListenerModelsFrom(WithListenerModels(context.Background(), nil)); ok {
		t.Fatal("an empty allow-list must restrict nothing")
	}
	models := []string{"vllm-sr/auto"}
	ctx := WithListenerModels(context.Background(), models)
	models[0] = "changed"
	got, ok := ListenerModelsFrom(ctx)
	if !ok || !got.Allows("vllm-sr/auto") || got.Allows("changed") || got.Allows("VLLM-SR/AUTO") {
		t.Fatalf("allow-list = %v, %v; want exactly vllm-sr/auto, independent of the caller's slice", got, ok)
	}
}
