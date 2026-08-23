package extproc

import candle_binding "github.com/vllm-project/semantic-router/candle-binding"

// InitModalityClassifier initializes the internal modality signal classifier.
// The classifier contributes routing metadata only; selected logical Models are
// dispatched through BackendInvoker like every other inference request.
func InitModalityClassifier(modelPath string, useCPU bool) error {
	return candle_binding.InitMmBert32KModalityClassifier(modelPath, useCPU)
}
