package contextcompression

import "encoding/json"

type HeuristicTokenCounter struct{}

func (HeuristicTokenCounter) CountText(_ string, text string) (int, string) {
	return EstimateTokens(text), "heuristic"
}

func (HeuristicTokenCounter) CountRequest(
	_ string,
	request *RequestIR,
) (int, string) {
	if request == nil {
		return 0, "heuristic"
	}
	var source any = request.Raw
	if request.Semantic != nil {
		source = request.Semantic
	}
	encoded, err := json.Marshal(source)
	if err != nil {
		return 0, "heuristic"
	}
	return EstimateTokens(string(encoded)), "heuristic"
}

type CalibratedTokenCounter struct {
	Estimate func(model string, byteLength int) (tokens int, calibrated bool)
}

func (counter CalibratedTokenCounter) CountText(model string, text string) (int, string) {
	if counter.Estimate == nil {
		return HeuristicTokenCounter{}.CountText(model, text)
	}
	tokens, calibrated := counter.Estimate(model, len([]byte(text)))
	if tokens <= 0 {
		return HeuristicTokenCounter{}.CountText(model, text)
	}
	if calibrated {
		return tokens, "provider_calibrated"
	}
	return tokens, "model_heuristic"
}

func (counter CalibratedTokenCounter) CountRequest(
	model string,
	request *RequestIR,
) (int, string) {
	if request == nil {
		return 0, "model_heuristic"
	}
	var source any = request.Raw
	if request.Semantic != nil {
		source = request.Semantic
	}
	encoded, err := json.Marshal(source)
	if err != nil {
		return HeuristicTokenCounter{}.CountRequest(model, request)
	}
	return counter.CountText(model, string(encoded))
}
