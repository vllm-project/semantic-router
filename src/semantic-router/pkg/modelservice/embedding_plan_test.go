package modelservice

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestImplicitEmbeddingSharesStartupCPUBudget(t *testing.T) {
	t.Setenv(CPUProcessesEnv, "")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	cfg.EmbeddingModels.UseCPU = true
	cfg.Tools.Enabled = true
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"domain": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "cpu"},
	}
	cfg.ModelBindings = map[string]config.ModelBinding{
		"domain_classifier": {Deployment: "domain", Contract: config.RemoteClassifierContractLabelDistribution, Adapter: "modernbert"},
	}
	cfg.Decisions = []config.Decision{{Name: "math", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "math"}}}
	plans := planProcesses(config.ModelRuntimeDeploymentsInUse(cfg), []string{"vllm-srun"}, "", 192, "cpu")
	if len(plans) != 2 {
		t.Fatalf("classifier and embedding must be planned together: %d processes", len(plans))
	}
	var embeddingPlanned bool
	for _, plan := range plans {
		if plan.threads != 96 {
			t.Fatalf("process %q got %d threads instead of half the startup budget", plan.name, plan.threads)
		}
		if _, ok := plan.members["@embedding.mmbert"]; ok {
			embeddingPlanned = true
		}
	}
	if !embeddingPlanned {
		t.Fatal("embedding would still be loaded outside the generation plan")
	}
}
