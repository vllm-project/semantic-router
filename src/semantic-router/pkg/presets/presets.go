// Package presets is the catalog of the Router's canonical System One
// questions. A preset is a zero-shot question definition every decision
// model can answer: its instructions, options with descriptions, the stage
// and input it reads, how much of the input it covers, and whether it needs
// span answers. Application signals, match judges and executors ask presets;
// custom questions are systemone rules.
//
// Trained bindings (bindings.go) and calibrated thresholds (calibration.go)
// are per-family optimizations of a preset, never requirements: a model
// without them answers the canonical question.
package presets

import "slices"

// Question types.
const (
	Choice = "choice"
	Noul   = "noul"
	Score  = "score"
	Set    = "set"
	Span   = "span"
)

// Stages a question is asked in.
const (
	StageRequest  = "request"
	StageResponse = "response"
)

// Preset IDs. Relevance, SameRequest, ToolRank, ToolNeed and AnswerCheck are
// the presets a configuration names (match judges, executors); the others back
// application signals.
const (
	Domain         = "domain"
	FactCheck      = "fact_check"
	Feedback       = "feedback"
	Jailbreak      = "jailbreak"
	Harm           = "harm"
	HarmCategories = "harm_categories"
	PII            = "pii"
	Hallucination  = "hallucination"
	Refusal        = "refusal"
	TurnRelation   = "turn_relation"
	Complexity     = "complexity"
	Modality       = "modality"
	Relevance      = "relevance"
	SameRequest    = "same_request"
	ToolRank       = "tool_rank"
	ToolNeed       = "tool_need"
	AnswerCheck    = "answer_check"
)

// System One protocol limits per question.
const (
	MinChoices = 2
	MaxChoices = 255
	MinLabels  = 1
	MaxLabels  = 255
	MinLevels  = 2
	MaxLevels  = 10
)

// Option is one Choice option or one Set or Span label.
type Option struct {
	Key         string
	Description string
}

// Input names the parts of a request a question reads besides the current
// user turn.
type Input struct {
	// History adds the earlier user turns.
	History bool
	// PreviousTurns adds the last N turns before the current one (user and
	// assistant).
	PreviousTurns int
	// PreviousAnswer adds the assistant turn the current turn answers.
	PreviousAnswer bool
	// Answer adds the model's answer (response stage).
	Answer bool
	// Context adds the grounding context: retrieved documents and tool results.
	Context bool
	// ToolResults adds the request's tool results.
	ToolResults bool
	// Candidate adds the item a match judge scores (a document, a cached
	// request, a candidate answer).
	Candidate bool
}

// Preset is a canonical question.
type Preset struct {
	ID           string
	Type         string
	Instructions string
	// Options are the Choice options or the Set and Span labels; a Noul has
	// none, and ToolRank's options are the candidates of each call.
	Options []Option
	// Positive are the options a rule matches when it names none.
	Positive []string
	Stage    string
	Input    Input
	// Full reads the whole input, in windows past one forward; otherwise
	// the question reads the first tokens of a long input.
	Full bool
	// RequiresSpan marks a question only a model that answers span
	// questions can ask.
	RequiresSpan bool
}

// OptionKeys are the preset's option keys in order.
func (p Preset) OptionKeys() []string {
	keys := make([]string, len(p.Options))
	for i, option := range p.Options {
		keys[i] = option.Key
	}
	return keys
}

// HasOption reports whether key is one of the preset's options.
func (p Preset) HasOption(key string) bool {
	return slices.ContainsFunc(p.Options, func(option Option) bool { return option.Key == key })
}

// Lookup returns a preset by ID and stage. Jailbreak and Harm are asked in
// either stage: the request stage reads the request, the response stage the
// model's answer.
func Lookup(id, stage string) (Preset, bool) {
	if stage == "" {
		stage = StageRequest
	}
	for _, preset := range catalog {
		if preset.ID == id && preset.Stage == stage {
			return clone(preset), true
		}
	}
	return Preset{}, false
}

// Judges are the presets a match judge names.
func Judges() []string {
	return []string{Relevance, SameRequest, ToolRank}
}

func clone(p Preset) Preset {
	p.Options = slices.Clone(p.Options)
	p.Positive = slices.Clone(p.Positive)
	return p
}
