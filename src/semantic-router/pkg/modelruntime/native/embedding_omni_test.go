package native

import (
	"errors"
	"testing"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestOmniFactoryPreservesDynamicCPUAndRequiresExplicitGPUBudget(t *testing.T) {
	factory, err := resolveORTEmbeddingFactory("vela_omni")
	if err != nil {
		t.Fatal(err)
	}
	if factory.prepareOptions == nil {
		t.Fatal("missing adapter execution preparation")
	}
	for _, limit := range []int{0, 128, 512} {
		options, prepareErr := factory.prepareOptions(ort.Options{Provider: "cpu", MaxInputTokens: limit})
		if prepareErr != nil || options.ExecutionMaxInputTokens != 0 || options.MaxInputTokens != limit {
			t.Fatalf("CPU lost dynamic input policy: %+v %v", options, prepareErr)
		}
	}
	for _, provider := range []string{"rocm", "migraphx"} {
		if _, prepareErr := factory.prepareOptions(ort.Options{Provider: provider}); !errors.Is(prepareErr, binding.ErrCapability) {
			t.Fatalf("%s missing budget: %v", provider, prepareErr)
		}
		for _, limit := range []int{512, 1024} {
			options, prepareErr := factory.prepareOptions(ort.Options{Provider: provider, MaxInputTokens: limit})
			if prepareErr != nil || options.ExecutionMaxInputTokens != limit || options.AllowCPUFallback {
				t.Fatalf("%s explicit budget lost or fallback enabled: %+v %v", provider, options, prepareErr)
			}
		}
	}
	legacy, err := resolveORTEmbeddingFactory("mmbert")
	if err != nil || legacy.prepareOptions != nil {
		t.Fatalf("changed another adapter's execution policy: %+v %v", legacy, err)
	}
}
