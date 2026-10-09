package systemone

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
)

// NativeRequest retains the wire document. Routing and policy inspect typed
// views without reconstructing, truncating or reordering the model's questions.
type NativeRequest struct {
	Body       json.RawMessage
	Questions  []Question
	StateBytes int
	SignalText string
	ReturnMeta bool
	// InferenceBody requests internal provenance when an algorithm needs it.
	// Only metadata visibility differs from Body; the native task stays raw.
	InferenceBody json.RawMessage
}

// Question identifies a question within one named state; the empty state is
// the request's top-level state. These identities never depend on map order.
type Question struct {
	State, Name, Type string
	FullInput         bool
	ChoiceKeys        map[string]bool
	Levels            int
}

// Observation is a typed answer and its available evidence. Missing evidence
// is different from a zero-valued statistic or a confidently false Noul answer.
type Observation struct {
	Question
	Valid        bool
	Confidence   *float64
	Probability  *float64
	Distribution []float64
}

type nativeState struct {
	State     json.RawMessage            `json:"state"`
	Questions map[string]json.RawMessage `json:"questions"`
}

// ParseNativeRequest admits explicit Decision 2.0 question types for auto.
// Concrete model requests continue through the native runtime unchanged.
func ParseNativeRequest(body json.RawMessage) (*NativeRequest, error) {
	var envelope struct {
		nativeState
		States map[string]nativeState `json:"states"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, errors.New("invalid native request")
	}
	request := &NativeRequest{Body: append(json.RawMessage(nil), body...)}
	if err := request.addState("", envelope.nativeState); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(envelope.States))
	for name := range envelope.States {
		if name == "" {
			return nil, errors.New("additional state names must be nonempty")
		}
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if err := request.addState(name, envelope.States[name]); err != nil {
			return nil, err
		}
	}
	// Signals see the complete native task, including all states and questions.
	// The model selector and transport options are not task evidence.
	var task map[string]json.RawMessage
	if json.Unmarshal(body, &task) == nil {
		options := map[string]json.RawMessage{}
		if raw, exists := task["options"]; exists && json.Unmarshal(raw, &options) != nil {
			return nil, errors.New("native options must be an object")
		}
		if raw, exists := options["return_meta"]; exists {
			var enabled *bool
			if json.Unmarshal(raw, &enabled) != nil || enabled == nil {
				return nil, errors.New("options.return_meta must be a boolean")
			}
			request.ReturnMeta = *enabled
		}
		request.InferenceBody = request.Body
		if !request.ReturnMeta {
			if options == nil {
				options = map[string]json.RawMessage{}
			}
			options["return_meta"] = json.RawMessage("true")
			task["options"], _ = json.Marshal(options)
			request.InferenceBody, _ = json.Marshal(task)
		}
		delete(task, "model")
		delete(task, "options")
		encoded, _ := json.Marshal(task)
		request.SignalText = string(encoded)
	}
	return request, nil
}

func (r *NativeRequest) addState(name string, state nativeState) error {
	trimmed := bytes.TrimSpace(state.State)
	if len(trimmed) == 0 || (trimmed[0] != '"' && trimmed[0] != '{' && trimmed[0] != '[') || len(state.Questions) == 0 {
		return errors.New("each state requires a text, object or array and named questions")
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, trimmed); err != nil {
		return errors.New("invalid state")
	}
	r.StateBytes += compact.Len()
	if name == "" {
		if json.Unmarshal(trimmed, &r.SignalText) != nil {
			r.SignalText = compact.String()
		}
	}
	keys := make([]string, 0, len(state.Questions))
	for key := range state.Questions {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		var question struct {
			Type             string            `json:"type"`
			Preset           string            `json:"preset"`
			RequireFullInput bool              `json:"require_full_input"`
			Criteria         json.RawMessage   `json:"criteria"`
			Levels           []json.RawMessage `json:"levels"`
			Choices          []struct {
				Key string `json:"key"`
			} `json:"choices"`
		}
		if key == "" || json.Unmarshal(state.Questions[key], &question) != nil || question.Preset != "" {
			return errors.New("auto requires explicit named questions without model-specific presets")
		}
		q := Question{State: name, Name: key, Type: question.Type, FullInput: question.RequireFullInput}
		switch question.Type {
		case "choice":
			q.ChoiceKeys = map[string]bool{}
			var criteria map[string]json.RawMessage
			if len(question.Choices) > 0 {
				for _, choice := range question.Choices {
					q.ChoiceKeys[choice.Key] = true
				}
			} else if json.Unmarshal(question.Criteria, &criteria) == nil {
				for choice := range criteria {
					q.ChoiceKeys[choice] = true
				}
			}
			if len(q.ChoiceKeys) < 2 {
				return errors.New("choice requires at least two named criteria")
			}
		case "score":
			levels := question.Levels
			if len(levels) == 0 {
				_ = json.Unmarshal(question.Criteria, &levels)
			}
			q.Levels = len(levels)
			if q.Levels < 2 || q.Levels > 10 {
				return errors.New("score requires two to ten levels")
			}
		case "noul":
		default:
			return fmt.Errorf("auto does not support question type %q", question.Type)
		}
		r.Questions = append(r.Questions, q)
	}
	return nil
}

// Observe preserves absent, failed and incomplete answers as invalid rather
// than dropping them from a bundle's quality denominator.
func (r *NativeRequest) Observe(body json.RawMessage) []Observation {
	type responseState struct {
		Answers map[string]json.RawMessage `json:"answers"`
	}
	var response struct {
		responseState
		States map[string]responseState `json:"states"`
	}
	if json.Unmarshal(body, &response) != nil {
		// json.Unmarshal may populate a valid prefix before reporting a bad
		// field later in the envelope. Never accept that partial decoding.
		response.Answers = nil
		response.States = nil
	}
	observations := make([]Observation, 0, len(r.Questions))
	for _, q := range r.Questions {
		answers := response.Answers
		if q.State != "" {
			answers = response.States[q.State].Answers
		}
		observations = append(observations, observeAnswer(q, answers[q.Name]))
	}
	return observations
}

func observeAnswer(q Question, raw json.RawMessage) Observation {
	out := Observation{Question: q}
	var answer struct {
		Type          string             `json:"type"`
		Error         string             `json:"error"`
		Choice        *string            `json:"choice"`
		Score         *float64           `json:"score"`
		Noul          *float64           `json:"noul"`
		Confidence    *float64           `json:"confidence"`
		Coverage      string             `json:"input_coverage"`
		Probabilities map[string]float64 `json:"probabilities"`
	}
	if json.Unmarshal(raw, &answer) != nil || answer.Type != q.Type || answer.Error != "" || (q.FullInput && answer.Coverage != "complete") {
		return out
	}
	switch q.Type {
	case "choice":
		out.Valid = answer.Choice != nil && q.ChoiceKeys[*answer.Choice]
	case "score":
		out.Valid = answer.Score != nil && finite(*answer.Score) && *answer.Score >= 0 && *answer.Score <= float64(q.Levels-1)
	case "noul":
		out.Valid = probability(answer.Noul)
		if out.Valid {
			out.Probability = answer.Noul
			out.Distribution = []float64{*answer.Noul, 1 - *answer.Noul}
		}
	}
	if !out.Valid {
		return out
	}
	if probability(answer.Confidence) {
		out.Confidence = answer.Confidence
	}
	if q.Type != "noul" {
		expected := q.Levels
		if q.Type == "choice" {
			expected = len(q.ChoiceKeys)
		}
		if len(answer.Probabilities) == expected {
			sum, valid := 0.0, true
			for key, p := range answer.Probabilities {
				valid = valid && probability(&p)
				if q.Type == "choice" {
					valid = valid && q.ChoiceKeys[key]
				} else {
					index, err := strconv.Atoi(key)
					valid = valid && err == nil && index >= 0 && index < q.Levels && strconv.Itoa(index) == key
				}
				sum += p
				out.Distribution = append(out.Distribution, p)
			}
			if !valid || math.Abs(sum-1) > 1e-4 {
				out.Distribution = nil
			}
		}
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(out.Distribution)))
	return out
}

func finite(v float64) bool       { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func probability(v *float64) bool { return v != nil && finite(*v) && *v >= 0 && *v <= 1 }
