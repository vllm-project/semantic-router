package runtimetest

import (
	"net/http"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

const specialTokens = 2

// items reads a classify input in any of its contract forms.
func items(input api.ClassifyInput) []api.ClassifyItem {
	if text, err := input.AsInputText(); err == nil {
		return []api.ClassifyItem{{Text: &text}}
	}
	if texts, err := input.AsTextList(); err == nil {
		out := make([]api.ClassifyItem, len(texts))
		for i := range texts {
			out[i] = api.ClassifyItem{Text: &texts[i]}
		}
		return out
	}
	objects, _ := input.AsClassifyItemList()
	return objects
}

func field(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func (r *Runtime) classify(body api.ClassifyRequest) (int, api.ClassifyResponse, *api.ErrorBody) {
	model, status, errBody := r.model(body.Model, "classify")
	if status != http.StatusOK {
		return status, api.ClassifyResponse{}, errBody
	}
	head := model.Heads[0]
	if body.Head != nil {
		index := slices.IndexFunc(model.Heads, func(h Head) bool { return h.Name == *body.Head })
		if index < 0 {
			return http.StatusBadRequest, api.ClassifyResponse{}, &api.ErrorBody{Code: "invalid_request", Message: "unknown head"}
		}
		head = model.Heads[index]
	}
	options := api.ClassifyOptions{}
	if body.Options != nil {
		options = *body.Options
	}
	inputs := items(body.Input)
	response := api.ClassifyResponse{Model: model.ID, Head: head.Name, Kind: api.ClassifyResponseKind(head.Kind), Labels: head.Labels}
	for index, input := range inputs {
		result := classifyOne(model, head, input, options)
		result.Index = index
		response.Results = append(response.Results, result)
		if result.Input != nil {
			response.Usage.InputTokens += result.Input.Tokens
		}
	}
	return http.StatusOK, response, nil
}

func classifyOne(model Model, head Head, input api.ClassifyItem, options api.ClassifyOptions) api.ClassifyResult {
	text := field(input.Text)
	if answer := field(input.Answer); answer != "" {
		text = answer
	}
	all := words(text)
	tokens := len(all) + specialTokens
	limit := model.MaxInputTokens
	if options.MaxTokens != nil {
		limit = min(limit, *options.MaxTokens)
	}
	overflow := "reject"
	if options.Overflow != nil {
		overflow = string(*options.Overflow)
	}
	usage := &api.InputUsage{Tokens: tokens, ProcessedTokens: tokens}
	if tokens > limit {
		if overflow != "truncate" {
			failure := api.ItemError("max_length_exceeded")
			return api.ClassifyResult{Error: &failure, Input: usage}
		}
		all = all[:limit-specialTokens]
		usage.ProcessedTokens, usage.Truncated = limit, true
	}
	if overflow != "window" || options.Window == nil {
		result := readout(head, input, all)
		result.Input = usage
		return result
	}
	size := options.Window.Tokens - specialTokens
	overlap := 0
	if options.Window.Overlap != nil {
		overlap = *options.Window.Overlap
	}
	var windows []api.ClassifyWindow
	var spans []api.Span
	reduced := map[string][]float64{}
	for start := 0; ; start += size - overlap {
		end := min(start+size, len(all))
		part := readout(head, input, all[start:end])
		windows = append(windows, api.ClassifyWindow{Start: start, End: end, Probabilities: part.Probabilities, Scores: part.Scores})
		if part.Spans != nil {
			for _, span := range *part.Spans {
				if !slices.ContainsFunc(spans, func(s api.Span) bool { return s.Start == span.Start }) {
					spans = append(spans, span)
				}
			}
		}
		if part.Scores != nil {
			reduced["scores"] = maxEach(reduced["scores"], *part.Scores)
		}
		if part.Probabilities != nil {
			reduced["probabilities"] = maxEach(reduced["probabilities"], *part.Probabilities)
		}
		if end >= len(all) {
			break
		}
	}
	count := len(windows)
	usage.Windows = &count
	result := api.ClassifyResult{Input: usage}
	if head.Kind == "token" {
		// Like the runtime's token heads: one merged span set and a window
		// count, no per-window values.
		slices.SortFunc(spans, func(a, b api.Span) int { return a.Start - b.Start })
		result.Spans = &spans
	} else {
		result.Windows = &windows
	}
	if values, ok := reduced["scores"]; ok {
		result.Scores = &values
	}
	if values, ok := reduced["probabilities"]; ok {
		result.Probabilities = &values
	}
	return result
}

// readout reads one window of words with a head.
func readout(head Head, input api.ClassifyItem, part []word) api.ClassifyResult {
	content := make([]string, len(part))
	for i, w := range part {
		content[i] = strings.ToLower(strings.Trim(w.text, ".,;:!?"))
	}
	switch head.Kind {
	case "sequence":
		probabilities := make([]float64, len(head.Labels))
		chosen := 0
		for i, label := range head.Labels {
			if slices.Contains(content, strings.ToLower(label)) {
				chosen = i
				break
			}
		}
		for i := range probabilities {
			probabilities[i] = 0.1 / float64(max(len(head.Labels)-1, 1))
		}
		probabilities[chosen] = 0.9
		label := head.Labels[chosen]
		return api.ClassifyResult{Label: &label, Probabilities: &probabilities}
	case "scores":
		scores := make([]float64, len(head.Labels))
		for i, label := range head.Labels {
			scores[i] = 0.1
			if slices.Contains(content, strings.ToLower(label)) {
				scores[i] = 0.9
			}
		}
		return api.ClassifyResult{Scores: &scores}
	default:
		spans := []api.Span{}
		grounded := strings.Fields(strings.ToLower(field(input.Context)))
		for i, w := range part {
			if field(input.Answer) != "" {
				if !slices.Contains(grounded, content[i]) {
					spans = append(spans, api.Span{Label: "hallucinated", Start: w.start, End: w.end, Text: w.text, Probability: 0.9})
				}
				continue
			}
			for _, label := range head.Labels {
				entity := strings.TrimPrefix(label, "B-")
				if entity != label && strings.EqualFold(entity, content[i]) {
					spans = append(spans, api.Span{Label: entity, Start: w.start, End: w.end, Text: w.text, Probability: 0.95})
				}
			}
		}
		return api.ClassifyResult{Spans: &spans}
	}
}

func maxEach(into, values []float64) []float64 {
	if into == nil {
		return slices.Clone(values)
	}
	for i := range into {
		into[i] = max(into[i], values[i])
	}
	return into
}
