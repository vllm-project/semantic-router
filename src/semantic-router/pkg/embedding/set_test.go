package embedding

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestTheDefaultIsTheNamedPrimaryDeployment(t *testing.T) {
	provider, err := NewFuncProvider("test", 4, func(context.Context, string) ([]float32, error) { return make([]float32, 4), nil })
	if err != nil {
		t.Fatal(err)
	}
	set := NewSet(map[string]Provider{"vela-embedding": provider, "omni": provider}, "vela-embedding")
	if set.Primary() != "vela-embedding" || !set.Has("") || !set.Has("omni") {
		t.Fatalf("primary %q", set.Primary())
	}
	if _, err := NewSet(nil, "").Default(); err == nil {
		t.Error("an empty set served a default")
	}
}

func TestGetReportsAModelTheGenerationDidNotPrepare(t *testing.T) {
	_, err := NewSet(nil, "vela-embedding").Get("vela-embedding", 0, 0)
	if !errors.Is(err, ErrModelNotPrepared) || !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("Get() error = %v, want ErrModelNotPrepared and ErrCapability", err)
	}
	if want := `embedding deployment "vela-embedding" was not prepared for this generation`; !strings.Contains(err.Error(), want) {
		t.Fatalf("Get() error = %q, want it to contain %q", err, want)
	}
}
