package native

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestGroundedGPURequiresBudgetBeforeLoadingArtifact(t *testing.T) {
	for _, device := range []string{"rocm:0", "migraphx:0"} {
		spec := config.ResolvedModelBinding{Binding: config.ModelBinding{Adapter: "vela_halu"}, Deployment: config.ModelDeployment{Artifact: "missing-artifact", Provider: "ort", Device: device}}
		_, err := New(nil).ortGrounded(context.Background(), spec)
		if !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("%s: expected missing execution budget before model load, got %v", device, err)
		}
	}
}
