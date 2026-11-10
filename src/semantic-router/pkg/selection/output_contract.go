package selection

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ApplyDecisionOutput adds the hosted image_generation operation to a request
// that enters a decision whose output contract generates images and does not
// ask for the operation itself, so the operation's capability joins the
// request's: a candidate that cannot carry it is not eligible. A contract
// whose answer is the image alone forces the operation, unless the request
// already chooses its tools; one that also answers in text leaves the call to
// the model. It reports whether the request changed.
func ApplyDecisionOutput(request *llmprotocol.Request, decision *config.Decision) bool {
	if request == nil || decision == nil || !decision.Output.GeneratesImages() || request.ImageGeneration != nil {
		return false
	}
	request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	if decision.Output.ImageOnly() && (request.ToolChoice.Mode == "" || request.ToolChoice.Mode == llmprotocol.ToolChoiceAuto) && request.ToolChoice.Name == "" {
		request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	}
	return true
}
