package serving

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestQuestionScanBudgetIsADeclaredWindowInput(t *testing.T) {
	scanning := modelservice.ModelCard{ID: "vela2", MaxScanTokens: 32768}
	bounded := modelservice.ModelCard{ID: "vela2-2.0.0"}
	declared := func(deployment string, input config.ModelInputBudget) config.ResolvedModelBinding {
		return config.ResolvedModelBinding{
			Name:       "prompt_guard",
			Binding:    config.ModelBinding{Deployment: deployment},
			Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Input: input},
		}
	}
	window := config.ModelInputBudget{MaxTokens: 4096, Overflow: "window"}
	cases := []struct {
		name string
		spec config.ResolvedModelBinding
		card modelservice.ModelCard
		want int
		err  bool
	}{
		{"no input keeps the model's budget", declared("guard", config.ModelInputBudget{}), scanning, 0, false},
		{"a window input is the scan budget", declared("guard", window), scanning, 4096, false},
		{"an implicit deployment keeps the model's budget", declared(config.ImplicitDeploymentPrefix+"prompt_guard", window), scanning, 0, false},
		{"truncate", declared("guard", config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}), scanning, 0, true},
		{"a reject budget", declared("guard", config.ModelInputBudget{MaxTokens: 512, Overflow: "reject"}), scanning, 0, true},
		{"window without a budget", declared("guard", config.ModelInputBudget{Overflow: "window"}), scanning, 0, true},
		{"a model without a scan budget", declared("guard", window), bounded, 0, true},
	}
	for _, tc := range cases {
		scan, err := questionScanBudget(tc.spec, tc.card)
		if tc.err != (err != nil) || (err != nil && !errors.Is(err, binding.ErrCapability)) || scan != tc.want {
			t.Fatalf("%s: scan %d, err %v", tc.name, scan, err)
		}
	}
}
