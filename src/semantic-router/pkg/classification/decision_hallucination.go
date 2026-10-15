package classification

import (
	"context"
	"fmt"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// detectJudgment preserves the context/answer boundary and returns a verdict.
// A positive answer never manufactures an excerpt or a token location.
func (d *HallucinationDetector) detectJudgment(ctx context.Context, contextText, question, answer string) (*HallucinationResult, error) {
	if !d.initialized {
		return nil, fmt.Errorf("hallucination detection model not initialized")
	}
	if answer == "" {
		return &HallucinationResult{}, nil
	}
	if contextText == "" {
		return nil, fmt.Errorf("context is required for hallucination detection")
	}
	parts := map[string]string{"context": contextText, "answer": answer}
	if question != "" {
		parts["request"] = question
	}
	result, err := d.judgment.ask(ctx, modelservice.Request{Parts: parts})
	if err != nil {
		return nil, err
	}
	if math.IsNaN(result.Noul) || math.IsInf(result.Noul, 0) || result.Noul < 0 || result.Noul > 1 {
		return nil, fmt.Errorf("hallucination judgment returned invalid probability")
	}
	return &HallucinationResult{HallucinationDetected: result.Noul >= float64(d.hallucinationThreshold()), Confidence: float32(result.Noul), ScoreAvailable: true, ScoreKind: "probability"}, nil
}
