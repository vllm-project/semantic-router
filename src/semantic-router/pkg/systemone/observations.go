package systemone

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// NativeRequest retains the wire document. Routing and acceptance inspect typed
// views without reconstructing, truncating or reordering the model's questions.
type NativeRequest struct {
	Body       json.RawMessage
	Questions  []Question
	SignalText string
	ReturnMeta bool
	// InferenceBody requests internal provenance when an algorithm needs it.
	// Only metadata visibility differs from Body; the native task stays raw.
	InferenceBody json.RawMessage
	// Requirements are what a candidate model must answer and read to answer
	// the whole task.
	Requirements Requirements
}

// Question identifies a question within one named state: a batch entry's
// name, or empty for a standard request. These identities never depend on
// map order.
type Question struct {
	State, Name, Type string
	FullInput         bool
	ChoiceKeys        map[string]bool
	Levels            int
	// Labels are the names a set or span question declares. A set answer gives
	// each a probability and selects any number of them; a span answer finds
	// any number of spans of each.
	Labels map[string]bool
	// Target is the text a span question reads. Its spans are ranges of it.
	Target string
}

// Observation is a typed answer and its available evidence. Missing evidence
// is different from a zero-valued statistic or a confidently false Noul answer.
type Observation struct {
	Question
	Valid        bool
	Confidence   *float64
	Probability  *float64
	Distribution []float64
	// LabelProbabilities are a set answer's probabilities, one for each
	// declared label. They are independent, not a distribution.
	LabelProbabilities map[string]float64
}

type nativeState struct {
	State     json.RawMessage            `json:"state"`
	Questions map[string]json.RawMessage `json:"questions"`
}

// ParseNativeRequest admits explicit Decision 2.0 question types for auto, in
// a standard request (state and questions) or a batch (states), never both.
// Concrete model requests continue through the native runtime unchanged.
func ParseNativeRequest(body json.RawMessage) (*NativeRequest, error) {
	var envelope struct {
		nativeState
		States map[string]nativeState `json:"states"`
	}
	var task map[string]json.RawMessage
	if json.Unmarshal(body, &envelope) != nil || json.Unmarshal(body, &task) != nil {
		return nil, errors.New("invalid native request")
	}
	request := &NativeRequest{Body: append(json.RawMessage(nil), body...)}
	if _, batch := task["states"]; !batch {
		if err := request.addState("", envelope.nativeState); err != nil {
			return nil, err
		}
	} else if err := batchOnly(task, envelope.States); err != nil {
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
	request.Requirements = requirementsOf(request.Questions, body)
	// Signals see the complete native task, including all states and questions.
	// The model selector and transport options are not task evidence.
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
	return request, nil
}

// batchOnly refuses a batch with a field of a standard request beside its
// states, or with no entry.
func batchOnly(task map[string]json.RawMessage, states map[string]nativeState) error {
	for _, field := range []string{"state", "questions", "images", "videos"} {
		if _, mixed := task[field]; mixed {
			return fmt.Errorf("a request is a standard request (state and questions) or a batch (states), not both; this batch also has a top-level %s", field)
		}
	}
	if len(states) == 0 {
		return errors.New("states must be a nonempty object of named requests")
	}
	return nil
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
			Choices          []keyed           `json:"choices"`
			Labels           []keyed           `json:"labels"`
			Over             json.RawMessage   `json:"over"`
		}
		if key == "" || json.Unmarshal(state.Questions[key], &question) != nil || question.Preset != "" {
			return errors.New("auto requires explicit named questions without model-specific presets")
		}
		q := Question{State: name, Name: key, Type: question.Type, FullInput: question.RequireFullInput}
		switch question.Type {
		case "choice":
			q.ChoiceKeys = optionNames(question.Choices, question.Criteria)
			if len(q.ChoiceKeys) < 2 {
				return errors.New("choice requires at least two named criteria")
			}
		case "set":
			q.Labels = optionNames(question.Labels, question.Criteria)
			if len(q.Labels) == 0 {
				return errors.New("set requires at least one named label")
			}
			// A set answers each label as a Noul under "<name>.<label>", so
			// no other question of the state may have that name.
			for label := range q.Labels {
				if _, taken := state.Questions[key+"."+label]; taken {
					return fmt.Errorf("a label answer %s.<label> is another question's name", key)
				}
			}
		case "span":
			q.Labels = optionNames(question.Labels, question.Criteria)
			if len(q.Labels) == 0 {
				return errors.New("span requires at least one named label")
			}
			// A span that carries no label must not pass for a label named "".
			for label := range q.Labels {
				if strings.TrimSpace(label) == "" {
					return errors.New("span label names must not be blank")
				}
			}
			target, err := spanTarget(trimmed, question.Over)
			if err != nil {
				return err
			}
			q.Target = target
		case "score":
			levels := question.Levels
			if len(levels) == 0 {
				_ = json.Unmarshal(question.Criteria, &levels)
			}
			q.Levels = len(levels)
			if q.Levels < 2 || q.Levels > 10 {
				return errors.New("score requires two to ten levels")
			}
		case "noul", "box":
		default:
			return fmt.Errorf("auto does not support question type %q", question.Type)
		}
		r.Questions = append(r.Questions, q)
	}
	return nil
}

// keyed is one entry of a vllm-sr/4.0 option list: choices, or the labels of a
// set or a span.
type keyed struct {
	Key string `json:"key"`
}

// optionNames are the options a question names: the keys of its option list
// when it has one, else the names of its criteria.
func optionNames(listed []keyed, criteria json.RawMessage) map[string]bool {
	names := map[string]bool{}
	var described map[string]json.RawMessage
	if len(listed) > 0 {
		for _, option := range listed {
			names[option.Key] = true
		}
	} else if json.Unmarshal(criteria, &described) == nil {
		for name := range described {
			names[name] = true
		}
	}
	return names
}

// spanTarget is the text a span question reads when its request settles it: a
// text state, or the text part of an object state that over names. Elsewhere
// the model that answers picks the part and joins its keys, each family in its
// own way, so auto could not tell which text a span is a range of. A text
// state that holds a JSON object is as unsettled: Vela 2.0 reads it as that
// object and counts into one of its parts, and other families read the text.
func spanTarget(state, over json.RawMessage) (string, error) {
	if target, isText := textOf(state); isText {
		if holdsObject(target) {
			return "", errors.New("auto checks a span against the text its question reads, and a model may read a text state that holds a JSON object as that object: send the object as the state and name a text part of it with over")
		}
		return target, nil
	}
	var parts map[string]json.RawMessage
	if name, named := textOf(over); named && json.Unmarshal(state, &parts) == nil {
		if target, isText := textOf(parts[name]); isText {
			return target, nil
		}
	}
	return "", errors.New("auto checks a span against the text its question reads: send a text state, or name a text part of an object state with over")
}

// holdsObject reports whether a text is a JSON object between white space, as
// Vela 2.0 finds one: Python's strip also takes the separators U+001C to
// U+001F.
func holdsObject(text string) bool {
	trimmed := strings.TrimFunc(text, func(r rune) bool { return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f) })
	var object map[string]json.RawMessage
	return strings.HasPrefix(trimmed, "{") && json.Unmarshal([]byte(trimmed), &object) == nil
}

// textOf reads a JSON string. A null and an absent value are not text.
func textOf(raw json.RawMessage) (string, bool) {
	var text *string
	if json.Unmarshal(raw, &text) != nil || text == nil {
		return "", false
	}
	return *text, true
}

// requirementsOf are what a task's questions and media need of its model:
// each question type, image or video for the media of the task and its
// states, and full input when a question requires it.
func requirementsOf(questions []Question, body json.RawMessage) Requirements {
	var needs Requirements
	for _, q := range questions {
		if !slices.Contains(needs.QuestionTypes, q.Type) {
			needs.QuestionTypes = append(needs.QuestionTypes, q.Type)
		}
		needs.FullInput = needs.FullInput || q.FullInput
	}
	sort.Strings(needs.QuestionTypes)
	type media struct {
		Images []json.RawMessage `json:"images"`
		Videos []json.RawMessage `json:"videos"`
	}
	var task struct {
		media
		Assets map[string]json.RawMessage `json:"assets"`
		States map[string]media           `json:"states"`
	}
	_ = json.Unmarshal(body, &task)
	image, video := len(task.Images) > 0, len(task.Videos) > 0
	for _, state := range task.States {
		image = image || len(state.Images) > 0
		video = video || len(state.Videos) > 0
	}
	for _, raw := range task.Assets {
		var asset struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal(raw, &asset)
		if asset.Type == "video" {
			video = true
		} else {
			image = true
		}
	}
	if image {
		needs.Modalities = append(needs.Modalities, config.SystemOneModalityImage)
	}
	if video {
		needs.Modalities = append(needs.Modalities, config.SystemOneModalityVideo)
	}
	return needs
}

// stateAnswers is one state's part of a response: its answers by question
// name and, beside them, the family-specific sets and spans. A set question
// answers in sets and has no entry of its own in answers, where each label
// repeats as a "<name>.<label>" Noul; a span question answers a Noul and lists
// its spans in spans. Entries stay raw, so an absent entry, a null and an
// empty result remain different.
type stateAnswers struct {
	Answers map[string]json.RawMessage `json:"answers"`
	Sets    map[string]json.RawMessage `json:"sets"`
	Spans   map[string]json.RawMessage `json:"spans"`
}

// answersByState decodes a response once and returns each state's part of it:
// a standard request's under the empty name, a batch entry's under its own.
func answersByState(body json.RawMessage) func(state string) stateAnswers {
	var response struct {
		stateAnswers
		States map[string]stateAnswers `json:"states"`
	}
	if json.Unmarshal(body, &response) != nil {
		// json.Unmarshal may populate a valid prefix before reporting a bad
		// field later in the envelope. Never accept that partial decoding.
		return func(string) stateAnswers { return stateAnswers{} }
	}
	return func(state string) stateAnswers {
		if state == "" {
			return response.stateAnswers
		}
		return response.States[state]
	}
}

// Observe preserves absent, failed and incomplete answers as invalid rather
// than dropping them from a bundle's quality denominator.
func (r *NativeRequest) Observe(body json.RawMessage) []Observation {
	answers := answersByState(body)
	observations := make([]Observation, 0, len(r.Questions))
	for _, q := range r.Questions {
		observations = append(observations, answers(q.State).observe(q))
	}
	return observations
}

// observe checks what the state answers to one question.
func (s stateAnswers) observe(q Question) Observation {
	switch q.Type {
	case "set":
		return observeSet(q, s)
	case "span":
		return observeSpan(q, s)
	}
	return observeAnswer(q, s.Answers[q.Name])
}

// agreement bounds how far a set's two forms of one probability may differ.
const agreement = 1e-4

// observeSet checks a set answer against the labels its question declares.
// The answer comes in two forms that must agree: the set's own entry, with a
// probability for every label and the labels selected, and one Noul for each
// label beside the other answers. Selecting no label is an answer; a set
// without its entry, its selection or a label's probability is not.
func observeSet(q Question, state stateAnswers) Observation {
	out := Observation{Question: q}
	if raw, answered := state.Answers[q.Name]; answered {
		// A set has no answer of its own. A failed one has its error here.
		var failed struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(raw, &failed) != nil || failed.Error != "" {
			return out
		}
	}
	var set struct {
		Selected      []string            `json:"selected"`
		Probabilities map[string]*float64 `json:"probabilities"`
		Coverage      string              `json:"input_coverage"`
	}
	if json.Unmarshal(state.Sets[q.Name], &set) != nil || set.Selected == nil ||
		len(set.Probabilities) != len(q.Labels) || (q.FullInput && set.Coverage != "complete") {
		return out
	}
	for _, label := range set.Selected {
		if !q.Labels[label] {
			return out
		}
	}
	probabilities := make(map[string]float64, len(q.Labels))
	for label := range q.Labels {
		var answer struct {
			Type  string   `json:"type"`
			Error string   `json:"error"`
			Noul  *float64 `json:"noul"`
		}
		p := set.Probabilities[label]
		if !probability(p) || json.Unmarshal(state.Answers[q.Name+"."+label], &answer) != nil ||
			answer.Type != "noul" || answer.Error != "" || !probability(answer.Noul) ||
			math.Abs(*answer.Noul-*p) > agreement {
			return out
		}
		probabilities[label] = *p
	}
	out.Valid, out.LabelProbabilities = true, probabilities
	return out
}

// observeSpan checks a span answer against the labels its question declares
// and the text it reads. The answer is a Noul, which carries its statistics,
// and a list of spans beside the answers. Finding no span is an answer; a
// missing or null list is not. Each span names a declared label and a range of
// the text in code points, the end excluded, and repeats that range's text.
func observeSpan(q Question, state stateAnswers) Observation {
	noul := q
	noul.Type = "noul"
	out := observeAnswer(noul, state.Answers[q.Name])
	out.Question = q
	// A span answer has no confidence: its Noul is the surest word's probability.
	out.Confidence = nil
	var spans []struct {
		Label       *string  `json:"label"`
		Start       *int     `json:"start"`
		End         *int     `json:"end"`
		Text        *string  `json:"text"`
		Probability *float64 `json:"probability"`
	}
	if !out.Valid || json.Unmarshal(state.Spans[q.Name], &spans) != nil || spans == nil {
		return Observation{Question: q}
	}
	var target []rune
	if len(spans) > 0 {
		target = []rune(q.Target)
	}
	for _, span := range spans {
		if span.Label == nil || !q.Labels[*span.Label] || !probability(span.Probability) || span.Start == nil || span.End == nil || span.Text == nil ||
			*span.Start < 0 || *span.Start >= *span.End || *span.End > len(target) ||
			string(target[*span.Start:*span.End]) != *span.Text {
			return Observation{Question: q}
		}
	}
	return out
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
		Box           *struct {
			Top    *float64 `json:"top"`
			Bottom *float64 `json:"bottom"`
			Left   *float64 `json:"left"`
			Right  *float64 `json:"right"`
		} `json:"box"`
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
	case "box":
		box := answer.Box
		out.Valid = box != nil && probability(box.Top) && probability(box.Bottom) && probability(box.Left) && probability(box.Right) &&
			*box.Top <= *box.Bottom && *box.Left <= *box.Right
	}
	if !out.Valid {
		return out
	}
	if probability(answer.Confidence) {
		out.Confidence = answer.Confidence
	}
	if q.Type == "choice" || q.Type == "score" {
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
