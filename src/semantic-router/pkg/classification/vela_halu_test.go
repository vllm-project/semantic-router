package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVelaHaluImplicitBindingKeepsTaskPolicy(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.HallucinationMitigation.HallucinationModel.ModelID = config.Vela1SystemModels().HallucinationDetector
	models := consumerModelRuntime(nil)
	models.cfg = &cfg
	spec, err := models.localSpec("hallucination_detector", cfg.HallucinationMitigation.HallucinationModel.ModelID, "modernbert", config.RemoteClassifierContractTokenSpans, true)
	if err != nil {
		t.Fatal(err)
	}
	if !spec.Deployment.IsModelRuntime() || spec.Deployment.Artifact != "vllm-sr/Vela-1.0-Encoder-307M-Halu" || spec.Deployment.Revision == "" ||
		spec.Deployment.Device != "cpu" || spec.Binding.Deployment != "@hallucination_detector" || spec.Binding.Adapter != "vela_halu" ||
		spec.Deployment.Input.MaxTokens != 8192 || spec.Deployment.Input.Overflow != "reject" {
		t.Fatalf("incorrect Halu input contract: %+v", spec)
	}
	if _, err := models.localSpec("hallucination_detector", "models/mom-halugate-detector", "modernbert", config.RemoteClassifierContractTokenSpans, true); err == nil {
		t.Fatal("a legacy model without a model_runtime family must be refused with the migrate hint")
	}
	detector := &HallucinationDetector{config: &cfg.HallucinationMitigation.HallucinationModel}
	if !detector.acceptSpan("09:00", .51, true) || !detector.acceptSpan("猫", .51, true) {
		t.Fatal("default plugin policy swallowed a one-token hallucination")
	}
}

func TestTheDefaultHallucinationDetectorSharesTheVela2Deployment(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	models := consumerModelRuntime(nil)
	models.cfg = &cfg
	spec, err := models.localSpec("hallucination_detector", cfg.HallucinationMitigation.HallucinationModel.ModelID, "modernbert", config.RemoteClassifierContractTokenSpans, true)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Binding.Deployment != config.DefaultDecisionDeployment || spec.Deployment.Artifact != "vllm-sr/Vela-2.0-0.3B" || spec.Deployment.Profile != "max_speed" || spec.Deployment.Device != "cpu" {
		t.Fatalf("the default detector asks the shared Vela 2.0 0.3B deployment: %+v", spec)
	}
}
