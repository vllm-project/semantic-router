package extproc

// Modality constants returned by the modality routing classifier.
const (
	ModalityAR        = "AR"        // Text-only response via autoregressive LLM
	ModalityDiffusion = "DIFFUSION" // Image generation via diffusion model
	ModalityBoth      = "BOTH"      // Hybrid response requiring both text and image
)

// ModalityClassificationResult is routing evidence produced by signal
// evaluation. It cannot select or invoke a physical backend.
type ModalityClassificationResult struct {
	Modality   string  // "AR", "DIFFUSION", or "BOTH"
	Confidence float32 // Confidence score (0.0-1.0)
	Method     string  // Detection method used: "classifier", "keyword", "hybrid", or "signal"
}

// setModalityFromSignals records modality evidence for decision evaluation and
// response metadata. Selected logical Models are still transported by Envoy.
func (r *OpenAIRouter) setModalityFromSignals(ctx *RequestContext, matchedModalityRules []string) {
	if len(matchedModalityRules) == 0 {
		return
	}
	ctx.ModalityClassification = &ModalityClassificationResult{
		Modality: matchedModalityRules[0],
		Method:   "signal",
	}
}
