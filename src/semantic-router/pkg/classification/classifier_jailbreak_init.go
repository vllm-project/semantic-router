package classification

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type JailbreakInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}

// deriveArgmax returns the index and score of the highest-probability class
// in a complete distribution. It is the single place argmax/confidence is
// computed for jailbreak classification, so every SequenceClassifierBackend
// only ever returns raw probabilities.
func deriveArgmax(probabilities []float32) (int, float32) {
	bestIdx := -1
	var bestScore float32
	for idx, p := range probabilities {
		if bestIdx == -1 || p > bestScore {
			bestIdx = idx
			bestScore = p
		}
	}
	return bestIdx, bestScore
}

// createJailbreakInference creates the remote jailbreak inference of
// prompt_guard.backend; local prompt guards are prepared task bindings.
func createJailbreakInference(promptGuardCfg *config.PromptGuardConfig, routerCfg *config.RouterConfig, jailbreakMapping *JailbreakMapping, models ...*classifierModelRuntime) (SequenceClassifierBackend, error) {
	if promptGuardCfg.Backend != nil {
		backend := promptGuardCfg.Backend
		contract := config.RemoteClassifierContractLabelDistribution
		if backend.Protocol == config.RemoteClassifierProtocolHTTPChat {
			contract = config.RemoteClassifierContractLabelDecision
		}
		external, err := config.ResolveRemoteClassifierBackend(routerCfg, backend, config.ModelRoleGuardrail, contract)
		if err != nil {
			return nil, err
		}
		switch backend.Protocol {
		case config.RemoteClassifierProtocolHTTPClassify:
			inference, err := newHTTPClassifierInference(external, jailbreakMapping, time.Duration(backend.EffectiveDeadlineMs())*time.Millisecond)
			if err != nil {
				return nil, err
			}
			return bindRemoteJailbreak(models, routerCfg, backend, external, jailbreakMapping, inference)
		case config.RemoteClassifierProtocolHTTPChat:
			inference, err := NewVLLMJailbreakInference(external, promptGuardCfg.Threshold, jailbreakMapping, promptGuardCfg.PositiveLabels)
			if err != nil {
				return nil, err
			}
			inference.timeout = time.Duration(backend.EffectiveDeadlineMs()) * time.Millisecond
			return bindRemoteJailbreak(models, routerCfg, backend, external, jailbreakMapping, inference)
		default:
			return nil, fmt.Errorf("unsupported prompt guard adapter %q", backend.Protocol)
		}
	}

	return nil, fmt.Errorf("prompt guard has no remote backend")
}

// JailbreakDetection represents the result of jailbreak analysis for a piece of content.
type JailbreakDetection struct {
	Content       string               `json:"content"`
	IsJailbreak   bool                 `json:"is_jailbreak"`
	JailbreakType string               `json:"jailbreak_type"`
	Confidence    *float32             `json:"confidence,omitempty"`
	Decision      *tasks.LabelDecision `json:"decision,omitempty"`
	ContentIndex  int                  `json:"content_index"`
}
