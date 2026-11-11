package config

import (
	"slices"
	"strconv"
	"time"
)

// SignalTypeSystemOne asks a System One model a typed question: a
// model_runtime deployment, or a systemone deployment's remote API.
const SignalTypeSystemOne = "systemone"

const (
	SystemOneQuestionChoice = "choice"
	SystemOneQuestionNoul   = "noul"
	SystemOneQuestionScore  = "score"
	// SystemOneQuestionSet asks which of the labels apply; every label is
	// answered with its own probability.
	SystemOneQuestionSet = "set"
	// SystemOneQuestionSpan asks where in the text each label occurs; only a
	// span-capable model answers it.
	SystemOneQuestionSpan = "span"

	// SystemOneSpanHeadRouter and SystemOneSpanHeadBroad name the span heads
	// of models with two: the router head (PII, unsupported claims, toxic
	// spans) and the broad head (open extraction).
	SystemOneSpanHeadRouter = "router"
	SystemOneSpanHeadBroad  = "broad"

	// A rule is asked about the request, before a model is selected, or
	// about the model's answer.
	SystemOneStageRequest  = "request"
	SystemOneStageResponse = "response"

	// SystemOneCoverageTruncate reads the first tokens of a long input;
	// SystemOneCoverageFull reads all of it, in windows past one model input.
	SystemOneCoverageTruncate = "truncate"
	SystemOneCoverageFull     = "full"

	// What a full-coverage rule means when its model could not read the
	// whole input: a match, no match, or an unknown signal.
	SystemOneUnscannedMatch   = "match"
	SystemOneUnscannedNoMatch = "no_match"
	SystemOneUnscannedUnknown = "unknown"

	// DefaultSystemOneTimeoutMs bounds one call when a rule sets none.
	DefaultSystemOneTimeoutMs = 1000
	MaxSystemOneTimeoutMs     = 60000
	MinSystemOneChoices       = 2
	MaxSystemOneChoices       = 255
	MinSystemOneLevels        = 2
	MaxSystemOneLevels        = 10
	MinSystemOneLabels        = 1
	MaxSystemOneLabels        = 255
	MaxSystemOnePreviousTurns = 32
	// DefaultSystemOneNoulThreshold is the P(yes) a Noul answer needs to
	// match when its rule declares no predicate.
	DefaultSystemOneNoulThreshold = 0.5
)

// SystemOneRule is one named question to a System One model, and when its
// answer matches. The rule itself (a condition without a label) matches:
//   - noul: when P(yes) satisfies the predicate (default gte 0.5);
//   - score: when the expected level satisfies the predicate (required);
//   - choice: when the chosen option is one of Labels and, with a predicate,
//     its probability satisfies it; without Labels it has no match of its
//     own, and its conditions name a label;
//   - set: when a counted label (Labels, by default every label, without
//     IgnoreLabels) has a probability that satisfies the predicate or,
//     without one, the model selected it;
//   - span: as set, over each label's span probabilities, or without a
//     predicate, when the model found a span of the label.
//
// A condition that names a label reads that label alone: it is the chosen
// option (choice), or it matches as above (set, span). Every probability is
// also published as a value, so a condition can apply its own predicate. A
// late or failed answer leaves the signal unknown.
type SystemOneRule struct {
	Name         string            `yaml:"name"`
	Description  string            `yaml:"description,omitempty"`
	Question     SystemOneQuestion `yaml:"question"`
	Labels       []string          `yaml:"labels,omitempty"`
	IgnoreLabels []string          `yaml:"ignore_labels,omitempty"`
	Predicate    *NumericPredicate `yaml:"predicate,omitempty"`
	// Deployment names the deployment that answers the rule; without one
	// the decision model answers it.
	Deployment string         `yaml:"deployment,omitempty"`
	Stage      string         `yaml:"stage,omitempty" jsonschema:"enum=request,enum=response"`
	Input      SystemOneInput `yaml:"input,omitempty"`
	Coverage   string         `yaml:"coverage,omitempty" jsonschema:"enum=truncate,enum=full"`
	// OnUnscanned applies to a full-coverage rule whose model could not
	// read its whole input (default unknown).
	OnUnscanned string `yaml:"on_unscanned,omitempty" jsonschema:"enum=match,enum=no_match,enum=unknown"`
	TimeoutMs   int    `yaml:"timeout_ms,omitempty"`
}

// SystemOneInput names what a rule reads besides the current user turn.
type SystemOneInput struct {
	// History adds the earlier user turns.
	History bool `yaml:"history,omitempty"`
	// PreviousTurns adds the last N turns before the current one, user and
	// assistant alike.
	PreviousTurns int `yaml:"previous_turns,omitempty"`
	// Answer adds the model's answer, and Context the answer's grounding
	// context (retrieved documents and tool results); response stage only.
	Answer  bool `yaml:"answer,omitempty"`
	Context bool `yaml:"context,omitempty"`
	// ToolResults adds the request's tool results.
	ToolResults bool `yaml:"tool_results,omitempty"`
	// Image, Audio and Video add the request's parts of that medium; the
	// answering deployment must declare the input modality.
	Image bool `yaml:"image,omitempty"`
	Audio bool `yaml:"audio,omitempty"`
	Video bool `yaml:"video,omitempty"`
	// PreviousAnswer adds the assistant answer the current turn replies to.
	// Only a preset declares it, for the application rules that ask it.
	PreviousAnswer bool `yaml:"-" json:"-"`
}

// Input modalities a rule reads besides text.
const (
	SystemOneModalityImage = "image"
	SystemOneModalityAudio = "audio"
	SystemOneModalityVideo = "video"
)

// Media lists the input modalities the input reads besides text.
func (i SystemOneInput) Media() []string {
	var media []string
	for _, medium := range []struct {
		read     bool
		modality string
	}{{i.Image, SystemOneModalityImage}, {i.Audio, SystemOneModalityAudio}, {i.Video, SystemOneModalityVideo}} {
		if medium.read {
			media = append(media, medium.modality)
		}
	}
	return media
}

// SystemOneQuestion is a System One question with ordered options. Choice and
// Noul take choices and Score takes levels; Set and Span take labels, an
// optional threshold that replaces the model's own, and, for Span, the head
// that answers.
type SystemOneQuestion struct {
	Type         string            `yaml:"type" json:"type"`
	Instructions string            `yaml:"instructions" json:"instructions"`
	Choices      []SystemOneOption `yaml:"choices,omitempty" json:"choices,omitempty"`
	Levels       []string          `yaml:"levels,omitempty" json:"levels,omitempty"`
	Labels       []SystemOneOption `yaml:"labels,omitempty" json:"labels,omitempty"`
	Threshold    *float64          `yaml:"threshold,omitempty" json:"threshold,omitempty"`
	Head         string            `yaml:"head,omitempty" json:"head,omitempty"`
}

// SystemOneOption is one Choice option, the false/true descriptions of a
// Noul, or one Set or Span label.
type SystemOneOption struct {
	Key         string `yaml:"key" json:"key"`
	Description string `yaml:"description,omitempty" json:"description,omitempty"`
}

// SystemOneSelectionConfig asks a System One model to choose one of a routing
// decision's modelRefs. Options are the candidate model names; descriptions
// come from Candidates, then the model's configured description. Without a
// Deployment the Router's decision model chooses, on the deployment that
// already answers the request's signals.
type SystemOneSelectionConfig struct {
	Deployment   string            `yaml:"deployment,omitempty"`
	Instructions string            `yaml:"instructions"`
	Candidates   map[string]string `yaml:"candidates,omitempty"`
	TimeoutMs    int               `yaml:"timeout_ms,omitempty"`
}

// EffectiveTimeout is the rule's call deadline.
func (r SystemOneRule) EffectiveTimeout() time.Duration {
	return systemOneTimeout(r.TimeoutMs)
}

// EffectiveStage is the stage the rule is asked in.
func (r SystemOneRule) EffectiveStage() string {
	if r.Stage == "" {
		return SystemOneStageRequest
	}
	return r.Stage
}

// FullCoverage reports whether the rule reads its whole input.
func (r SystemOneRule) FullCoverage() bool {
	return r.Coverage == SystemOneCoverageFull
}

// EffectiveOnUnscanned is what an answer that could not read the whole input
// of a full-coverage rule means.
func (r SystemOneRule) EffectiveOnUnscanned() string {
	if r.OnUnscanned == "" {
		return SystemOneUnscannedUnknown
	}
	return r.OnUnscanned
}

// EffectivePredicate is the predicate a Noul or Score answer must satisfy, or
// the optional predicate on a Choice option's or a Set or Span label's
// probability.
func (r SystemOneRule) EffectivePredicate() *NumericPredicate {
	if r.Predicate != nil || r.Question.Type != SystemOneQuestionNoul {
		return r.Predicate
	}
	threshold := DefaultSystemOneNoulThreshold
	return &NumericPredicate{GTE: &threshold}
}

// MatchesItself reports whether a condition can name the rule without a
// label: every rule but a choice rule without labels.
func (r SystemOneRule) MatchesItself() bool {
	return r.Question.Type != SystemOneQuestionChoice || len(r.Labels) > 0
}

// CountedLabels are the options a rule-level match reads, in question order:
// a choice rule's labels, and a set or span rule's labels (every label by
// default) without its ignored ones. Noul and Score rules have none.
func (r SystemOneRule) CountedLabels() []string {
	switch r.Question.Type {
	case SystemOneQuestionChoice, SystemOneQuestionSet, SystemOneQuestionSpan:
	default:
		return nil
	}
	var counted []string
	for _, key := range r.Question.OptionKeys() {
		if (len(r.Labels) == 0 && r.Question.Type != SystemOneQuestionChoice) || slices.Contains(r.Labels, key) {
			if !slices.Contains(r.IgnoreLabels, key) {
				counted = append(counted, key)
			}
		}
	}
	return counted
}

// Labelled reports whether conditions on the question name one of its
// options: a Choice option or a Set or Span label.
func (q SystemOneQuestion) Labelled() bool {
	switch q.Type {
	case SystemOneQuestionChoice, SystemOneQuestionSet, SystemOneQuestionSpan:
		return true
	}
	return false
}

// OptionKeys are the answer keys in rendering order.
func (q SystemOneQuestion) OptionKeys() []string {
	switch q.Type {
	case SystemOneQuestionScore:
		keys := make([]string, len(q.Levels))
		for index := range q.Levels {
			keys[index] = strconv.Itoa(index)
		}
		return keys
	case SystemOneQuestionNoul:
		return []string{"false", "true"}
	case SystemOneQuestionSet, SystemOneQuestionSpan:
		return systemOneOptionKeys(q.Labels)
	default:
		return systemOneOptionKeys(q.Choices)
	}
}

func systemOneOptionKeys(choices []SystemOneOption) []string {
	keys := make([]string, len(choices))
	for index, choice := range choices {
		keys[index] = choice.Key
	}
	return keys
}

// EffectiveTimeout is the selector's call deadline.
func (c SystemOneSelectionConfig) EffectiveTimeout() time.Duration {
	return systemOneTimeout(c.TimeoutMs)
}

func systemOneTimeout(milliseconds int) time.Duration {
	if milliseconds <= 0 {
		milliseconds = DefaultSystemOneTimeoutMs
	}
	return time.Duration(milliseconds) * time.Millisecond
}
