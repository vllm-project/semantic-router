//go:build !windows

package benchmarks

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestModelProtocolSeparatesExecutionSettings(t *testing.T) {
	card := modelservice.ModelCard{Engine: "torch", EngineArgs: modelservice.EngineArgs{
		MaxNumSeqs: 8, MaxNumBatchedTokens: 1024, BatchWindowMS: 2,
		Dtype: "float32", EnableSharedContext: true,
	}}
	input := config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}
	baseline, err := modelRuntimeProtocol(card, input, "full-layer/full-dimension")
	if err != nil {
		t.Fatal(err)
	}
	mutations := map[string]func(*modelservice.ModelCard){
		"engine":         func(c *modelservice.ModelCard) { c.Engine = "onnx" },
		"batch size":     func(c *modelservice.ModelCard) { c.EngineArgs.MaxNumSeqs++ },
		"batch tokens":   func(c *modelservice.ModelCard) { c.EngineArgs.MaxNumBatchedTokens++ },
		"batch window":   func(c *modelservice.ModelCard) { c.EngineArgs.BatchWindowMS++ },
		"precision":      func(c *modelservice.ModelCard) { c.EngineArgs.Dtype = "bfloat16" },
		"shared context": func(c *modelservice.ModelCard) { c.EngineArgs.EnableSharedContext = false },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			changed := card
			mutate(&changed)
			protocol, protocolErr := modelRuntimeProtocol(changed, input, "full-layer/full-dimension")
			if protocolErr != nil || protocol == baseline {
				t.Fatalf("different execution settings remained comparable: %s, %v", protocol, protocolErr)
			}
		})
	}
	for _, changed := range []config.ModelInputBudget{
		{MaxTokens: 8192, Overflow: "truncate"},
		{MaxTokens: 512, Overflow: "reject"},
	} {
		protocol, protocolErr := modelRuntimeProtocol(card, changed, "full-layer/full-dimension")
		if protocolErr != nil || protocol == baseline {
			t.Fatalf("different input budgets remained comparable: %s, %v", protocol, protocolErr)
		}
	}
	protocol, err := modelRuntimeProtocol(card, input, "layer-6/dimension-256")
	if err != nil || protocol == baseline {
		t.Fatalf("different embedding workloads remained comparable: %s, %v", protocol, err)
	}
	protocol, err = modelRuntimeProtocol(card, input, "full-layer/full-dimension")
	if err != nil || protocol != baseline {
		t.Fatalf("identical execution settings differ: %s, %v", protocol, err)
	}
	card.EngineArgs.BatchWindowMS = math.NaN()
	if _, err := modelRuntimeProtocol(card, input, "full-layer/full-dimension"); err == nil {
		t.Fatal("invalid execution settings produced a baseline identity")
	}
}
