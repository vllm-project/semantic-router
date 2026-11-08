package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVelaHaluImplicitBindingKeepsTaskPolicy(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.HallucinationMitigation.HallucinationModel.ModelID = config.Vela1SystemModels().HallucinationDetector
	models := consumerModelRuntime(nil)
	models.cfg = &cfg
	spec, err := models.localSpec("hallucination_detector", "modernbert", config.RemoteClassifierContractTokenSpans)
	if err != nil {
		t.Fatal(err)
	}
	if !spec.Deployment.IsModelRuntime() || spec.Deployment.Artifact != "vllm-sr/Vela-1.0-Encoder-307M-Halu" || spec.Deployment.Revision == "" ||
		spec.Deployment.Device != "cpu" || spec.Binding.Deployment != "Vela-1.0-Encoder-307M-Halu" || spec.Binding.Adapter != "vela_halu" ||
		spec.Deployment.Input.MaxTokens != 8192 || spec.Deployment.Input.Overflow != "reject" {
		t.Fatalf("incorrect Halu input contract: %+v", spec)
	}
	legacy := config.DefaultGlobalConfig()
	legacy.HallucinationMitigation.HallucinationModel.ModelID = "models/mom-halugate-detector"
	models.cfg = &legacy
	if _, err := models.localSpec("hallucination_detector", "modernbert", config.RemoteClassifierContractTokenSpans); err == nil || !strings.Contains(err.Error(), "vllm-sr config migrate") {
		t.Fatalf("a legacy model without a model_runtime family must be refused with the migrate hint: %v", err)
	}
	models.cfg = &cfg
	detector := &HallucinationDetector{config: &cfg.HallucinationMitigation.HallucinationModel}
	if !detector.acceptSpan("09:00", .51, true) || !detector.acceptSpan("猫", .51, true) {
		t.Fatal("default plugin policy swallowed a one-token hallucination")
	}
}

func TestTheDefaultHallucinationDetectorSharesTheDefaultDeployment(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	models := consumerModelRuntime(nil)
	models.cfg = &cfg
	spec, err := models.localSpec("hallucination_detector", "modernbert", config.RemoteClassifierContractTokenSpans)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Binding.Deployment != config.DefaultDecisionDeployment || spec.Deployment.Artifact != "vllm-sr/vela3-fast" || spec.Deployment.HasEngineArgs() || spec.Deployment.Device != "cpu" {
		t.Fatalf("the default detector asks the shared vela3-fast deployment: %+v", spec)
	}
}
