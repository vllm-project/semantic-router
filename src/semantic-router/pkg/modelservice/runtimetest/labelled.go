package runtimetest

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// Labelled makes a decision model also answer Set and Span questions and the
// pii and halu presets, the way Vela 2.0 does. A text names a label when one
// of its words equals the label's key, case-insensitively: a Set label it
// names gets 0.9 (else 0.1), and every naming word becomes a span of that
// label at 0.95. The pii preset asks for PIILabels; the halu preset marks
// the answer words its context does not contain as "unsupported". A
// question's threshold applies; the default is 0.5. With BroadHead, span
// answers name the head that answered (router for the presets).
type Labelled struct {
	PIILabels []string
	BroadHead bool
	// ScanTokens is the scan budget (its card's max_scan_tokens): a state part
	// with more words fails every question that reads it whole (no overflow:
	// truncate) with scan_budget_exceeded. Zero means four inputs
	// (4 * MaxInputTokens); a request's max_tokens overrides it, down to one input.
	ScanTokens int
}

// scanTokens is the model's scan budget.
func (l Labelled) scanTokens(model Model) int {
	if l.ScanTokens > 0 {
		return l.ScanTokens
	}
	return 4 * model.MaxInputTokens
}

// unscanned fails the questions that read the state whole when a part of it
// has more words than the request's scan budget.
func unscanned(model Model, body api.DecisionRequest, response *api.DecisionResponse) {
	budget := model.Labelled.scanTokens(model)
	if body.Options != nil && body.Options.MaxTokens != nil {
		budget = max(*body.Options.MaxTokens, model.MaxInputTokens)
	}
	over := false
	for _, text := range stateParts(body.State) {
		over = over || len(words(text)) > budget
	}
	if !over {
		return
	}
	for id, question := range body.Questions {
		if question.Overflow != nil && *question.Overflow == api.QuestionOverflowTruncate {
			continue
		}
		response.Answers[id] = api.Answer{Type: question.Type, Error: itemError("scan_budget_exceeded")}
	}
}

// SystemOneTypes are the question types every decision model answers.
var SystemOneTypes = []string{"choice", "noul", "score"}

func questionTypes(model Model) []string {
	if model.Labelled != nil {
		return append(append([]string(nil), SystemOneTypes...), "set", "span")
	}
	return SystemOneTypes
}

func presets(model Model) []string {
	if model.Labelled != nil {
		return []string{"halu", "pii"}
	}
	return nil
}

// decideLabelled answers every question of a Vela 2.0-style request.
func (r *Runtime) decideLabelled(model Model, body api.DecisionRequest) api.DecisionResponse {
	response := api.DecisionResponse{Model: model.ID, Answers: map[string]api.Answer{}, Usage: api.Usage{InputTokens: 10}}
	sets, spans, thresholds, heads := map[string]api.SetAnswer{}, map[string][]api.Span{}, map[string]float64{}, map[string]string{}
	parts := stateParts(body.State)
	for id, question := range body.Questions {
		kind, labels, preset := labelledQuestion(model, question)
		threshold := 0.5
		if question.Threshold != nil {
			threshold = *question.Threshold
		}
		switch kind {
		case "set":
			text := parts["request"]
			probabilities, selected := map[string]float64{}, []string{}
			for _, label := range labels {
				p := 0.1
				if names(text, label) {
					p = 0.9
				}
				probabilities[label] = p
				noul := p
				response.Answers[id+"."+label] = api.Answer{Type: stringPointer("noul"), Noul: &noul}
				if p > threshold {
					selected = append(selected, label)
				}
			}
			sets[id], thresholds[id] = api.SetAnswer{Probabilities: probabilities, Selected: selected}, threshold
		case "span":
			text := parts["request"]
			if preset == "halu" || parts["answer"] != "" {
				text = parts["answer"]
			}
			found := labelledSpans(text, labels, preset, parts["context"])
			best := 0.05
			if len(found) > 0 {
				best = 0.95
			}
			response.Answers[id] = api.Answer{Type: stringPointer("noul"), Noul: &best}
			spans[id], thresholds[id] = found, threshold
			if model.Labelled.BroadHead {
				heads[id] = "broad"
				if preset != "" || (question.Head != nil && *question.Head == "router") {
					heads[id] = "router"
				}
			}
		case "":
			response.Answers[id] = api.Answer{Type: question.Type, Error: itemError("invalid_question")}
		default:
			response.Answers[id] = jointAnswer(model, question, len(body.Questions))
		}
	}
	response.Sets, response.Spans, response.Thresholds = &sets, &spans, &thresholds
	if len(heads) > 0 {
		response.SpanHeads = &heads
	}
	return response
}

// labelledQuestion is a question's type, labels and preset; the type is ""
// for a preset the model does not define.
func labelledQuestion(model Model, question api.Question) (string, []string, string) {
	if question.Preset != nil {
		switch *question.Preset {
		case "pii":
			return "span", model.Labelled.PIILabels, "pii"
		case "halu":
			return "span", []string{"unsupported"}, "halu"
		}
		return "", nil, *question.Preset
	}
	kind := ""
	if question.Type != nil {
		kind = *question.Type
	}
	if kind != "set" && kind != "span" {
		return kind, nil, ""
	}
	return kind, criteriaLabels(question.Criteria), ""
}

// criteriaLabels reads a criteria object's labels in request order.
func criteriaLabels(criteria *interface{}) []string {
	if criteria == nil {
		return nil
	}
	encoded, err := json.Marshal(*criteria)
	if err != nil {
		return nil
	}
	decoder := json.NewDecoder(strings.NewReader(string(encoded)))
	var labels []string
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil
	}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return labels
		}
		labels = append(labels, token.(string))
		var skip interface{}
		if err := decoder.Decode(&skip); err != nil {
			return labels
		}
	}
	return labels
}

// stateParts reads a text state as its request, or an object state's request,
// context and answer fields.
func stateParts(state interface{}) map[string]string {
	switch value := state.(type) {
	case string:
		return map[string]string{"request": value}
	case map[string]interface{}:
		parts := map[string]string{}
		for key, field := range value {
			if text, ok := field.(string); ok {
				parts[key] = text
			}
		}
		return parts
	}
	return map[string]string{}
}

func names(text, label string) bool {
	for _, w := range words(text) {
		if strings.EqualFold(w.text, label) {
			return true
		}
	}
	return false
}

func labelledSpans(text string, labels []string, preset, context string) []api.Span {
	spans := []api.Span{}
	for _, w := range words(text) {
		if preset == "halu" {
			if !strings.Contains(" "+strings.ToLower(context)+" ", " "+strings.ToLower(w.text)+" ") {
				spans = append(spans, api.Span{Label: "unsupported", Start: w.start, End: w.end, Text: w.text, Probability: 0.95})
			}
			continue
		}
		for _, label := range labels {
			if strings.EqualFold(w.text, label) {
				spans = append(spans, api.Span{Label: label, Start: w.start, End: w.end, Text: w.text, Probability: 0.95})
				break
			}
		}
	}
	return spans
}

func stringPointer(value string) *string { return &value }

func itemError(code string) *api.ItemError {
	value := api.ItemError(code)
	return &value
}
