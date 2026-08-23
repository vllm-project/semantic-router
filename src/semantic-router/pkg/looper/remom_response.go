package looper

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func selectReMoMFinalResponse(rounds []RoundResponse, completedFinal bool) (IntermediateResp, error) {
	if len(rounds) == 0 {
		return IntermediateResp{}, fmt.Errorf("remom produced no responses")
	}

	fallbackRounds := rounds
	if completedFinal {
		finalRound := rounds[len(rounds)-1]
		for _, response := range finalRound.Responses {
			if strings.TrimSpace(response.Content) != "" {
				return response, nil
			}
		}
		fallbackRounds = rounds[:len(rounds)-1]
	}

	best, ok := bestPreviousReMoMResponse(fallbackRounds)
	if !ok {
		return IntermediateResp{}, fmt.Errorf("remom produced no assistant content")
	}
	return best, nil
}

func bestPreviousReMoMResponse(rounds []RoundResponse) (IntermediateResp, bool) {
	var best IntermediateResp
	bestLength := -1
	// Prefer the longest publishable output, matching ReMoM's response ordering
	// heuristic. Reverse round order makes equal-length ties favor newer work.
	for roundIndex := len(rounds) - 1; roundIndex >= 0; roundIndex-- {
		for _, response := range rounds[roundIndex].Responses {
			normalized := strings.TrimSpace(response.Content)
			if normalized == "" || len(normalized) <= bestLength {
				continue
			}
			best = response
			bestLength = len(normalized)
		}
	}
	return best, bestLength >= 0
}

// reMoMSynthesisText may use reasoning as internal evidence without changing
// ModelResponse.Content, which remains the only publishable assistant answer.
func reMoMSynthesisText(response *ModelResponse) string {
	if response == nil {
		return ""
	}
	if strings.TrimSpace(response.Content) != "" {
		return response.Content
	}
	return response.ReasoningContent
}

func isUsableReMoMResponse(response *ModelResponse) bool {
	return strings.TrimSpace(reMoMSynthesisText(response)) != ""
}

func usableReMoMResponses(responses []*ModelResponse) []*ModelResponse {
	usable := make([]*ModelResponse, 0, len(responses))
	for _, response := range responses {
		if !isUsableReMoMResponse(response) {
			continue
		}
		usable = append(usable, response)
	}
	return usable
}

// formatReMoMJSONResponse creates a non-streaming JSON response.
func (l *ReMoMLooper) formatReMoMJSONResponse(
	finalResponse IntermediateResp,
	allRoundResponses []RoundResponse,
	modelsUsed []string,
	iterations int,
	usage TokenUsage,
	cfg *config.ReMoMAlgorithmConfig,
) (*Response, error) {
	if strings.TrimSpace(finalResponse.Content) == "" {
		return nil, fmt.Errorf("remom produced no assistant content")
	}
	_ = cfg
	semantic := newTextSemanticResponse("response-remom", finalResponse.Model, finalResponse.Content, usage)
	return newLooperResponse(semantic, false, true, finalResponse.Model, modelsUsed, iterations, "remom", usage, allRoundResponses), nil
}

// formatReMoMStreamingResponse creates an SSE streaming response.
func (l *ReMoMLooper) formatReMoMStreamingResponse(
	finalResponse IntermediateResp,
	allRoundResponses []RoundResponse,
	modelsUsed []string,
	iterations int,
	usage TokenUsage,
	cfg *config.ReMoMAlgorithmConfig,
	includeUsage bool,
) (*Response, error) {
	if strings.TrimSpace(finalResponse.Content) == "" {
		return nil, fmt.Errorf("remom produced no assistant content")
	}
	_ = cfg
	semantic := newTextSemanticResponse("response-remom", finalResponse.Model, finalResponse.Content, usage)
	return newLooperResponse(semantic, true, includeUsage, finalResponse.Model, modelsUsed, iterations, "remom", usage, allRoundResponses), nil
}
