package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVelaHaluImplicitBindingKeepsTaskPolicy(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	models := consumerModelRuntime(nil)
	models.cfg = &cfg
	spec := models.localSpec("hallucination_detector", cfg.HallucinationMitigation.HallucinationModel.ModelID, "modernbert", config.RemoteClassifierContractTokenSpans, true)
	if spec.Deployment.Provider != "candle" || spec.Deployment.Device != "cpu" || spec.Binding.Adapter != "vela_halu" || spec.Deployment.Input.MaxTokens != 8192 || spec.Deployment.Input.Overflow != "reject" {
		t.Fatalf("incorrect Halu input contract: %+v", spec)
	}
	old := models.localSpec("hallucination_detector", "models/mom-halugate-detector", "modernbert", config.RemoteClassifierContractTokenSpans, true)
	if old.Binding.Adapter != "modernbert" || old.Deployment.Input.MaxTokens != 0 || old.Deployment.Input.Overflow != "truncate" {
		t.Fatalf("legacy explicit model policy changed: %+v", old)
	}
	detector := &HallucinationDetector{config: &cfg.HallucinationMitigation.HallucinationModel}
	if !detector.acceptSpan("09:00", .51, true) || !detector.acceptSpan("猫", .51, true) {
		t.Fatal("default plugin policy swallowed a one-token hallucination")
	}
}
