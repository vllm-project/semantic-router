package config

// The application signals ask the Router's presets (pkg/presets) about a
// request or a model's answer. Each rule compiles into one systemone rule
// (ApplicationRules): the preset's question, stage, input and coverage, with
// the rule's own labels, predicate, deployment and timeout. Conditions name
// the application type, which a rule's matches and values are published
// under: domain:<rule>, safety:<rule>:<label>.

const (
	SignalTypeRefusal      = "refusal"
	SignalTypeTurnRelation = "turn_relation"
)

// Category is a domains rule: the request's subject area is one of Labels
// (by default the rule's name) and, with a predicate, its probability
// satisfies it. ModelScores weigh the candidate models of the static and Elo
// selectors for requests in the domain.
type Category struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Labels      []string          `yaml:"labels,omitempty"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	Deployment  string            `yaml:"deployment,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
	ModelScores []ModelScore      `yaml:"model_scores,omitempty"`
}

type ModelScore struct {
	Model        string  `yaml:"model"`
	Score        float64 `yaml:"score"`
	UseReasoning *bool   `yaml:"use_reasoning"`
}

// FactCheckRule matches when answering the request needs fact checking
// (labels FACT_CHECK_NEEDED, the default) or does not (NO_FACT_CHECK_NEEDED),
// each option on its own probability.
type FactCheckRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Labels      []string          `yaml:"labels,omitempty"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	Deployment  string            `yaml:"deployment,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
}

// UserFeedbackRule matches the feedback a user turn gives about the previous
// answer: SAT, NEED_CLARIFICATION, WRONG_ANSWER, WANT_DIFFERENT or
// NO_FEEDBACK, when the chosen feedback's probability satisfies the
// predicate. A rule without labels matches nothing itself; its conditions
// name one.
type UserFeedbackRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Labels      []string          `yaml:"labels,omitempty"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	Deployment  string            `yaml:"deployment,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
}

// JailbreakRule matches a prompt injection or jailbreak attempt in the
// request or, at the response stage, in the model's answer. The whole input
// is read, in windows when it is longer than one; any window that finds an
// attack flags the input.
type JailbreakRule struct {
	Name        string             `yaml:"name"`
	Description string             `yaml:"description,omitempty"`
	Predicate   *NumericPredicate  `yaml:"predicate,omitempty"`
	Stage       string             `yaml:"stage,omitempty" jsonschema:"enum=request,enum=response"`
	Input       JailbreakRuleInput `yaml:"input,omitempty"`
	Deployment  string             `yaml:"deployment,omitempty"`
	TimeoutMs   int                `yaml:"timeout_ms,omitempty"`
	// OnUnscanned is what the rule reads as when only some of its windows
	// were read: best_effort (the default) decides on those, match matches,
	// no_match does not, and unknown leaves the rule to OnUnknown.
	OnUnscanned string `yaml:"on_unscanned,omitempty" jsonschema:"enum=best_effort,enum=match,enum=no_match,enum=unknown"`
	// OnUnknown is what a request-stage rule reads as when its deployment
	// gives no answer: match (the default) or no_match.
	OnUnknown string `yaml:"on_unknown,omitempty" jsonschema:"enum=match,enum=no_match"`
}

// JailbreakRuleInput adds the earlier user turns to the text a jailbreak rule
// reads.
type JailbreakRuleInput struct {
	History bool `yaml:"history,omitempty"`
}

// SafetyRule matches harmful content in the request or, at the response
// stage, in the model's answer: the answer is unsafe or, with labels, it is
// harmful in one of those harm categories (presets.HarmCategoryOptions).
type SafetyRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Labels      []string          `yaml:"labels,omitempty"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	Stage       string            `yaml:"stage,omitempty" jsonschema:"enum=request,enum=response"`
	Deployment  string            `yaml:"deployment,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
}

// PIIRule matches personal data the model locates in the request: a span of
// a counted label (every pii label but IgnoreLabels) whose probability
// satisfies the predicate. A long input is read in windows, and the spans of
// every window count, at their places in the input.
type PIIRule struct {
	Name         string            `yaml:"name"`
	Description  string            `yaml:"description,omitempty"`
	Predicate    *NumericPredicate `yaml:"predicate,omitempty"`
	IgnoreLabels []string          `yaml:"ignore_labels,omitempty"`
	Input        PIIRuleInput      `yaml:"input,omitempty"`
	Deployment   string            `yaml:"deployment,omitempty"`
	TimeoutMs    int               `yaml:"timeout_ms,omitempty"`
	// OnUnscanned is what the rule reads as when only some of its windows
	// were read: best_effort (the default) decides on those, match matches,
	// no_match does not, and unknown leaves the rule to OnUnknown.
	OnUnscanned string `yaml:"on_unscanned,omitempty" jsonschema:"enum=best_effort,enum=match,enum=no_match,enum=unknown"`
	// OnUnknown is what the rule reads as when its deployment gives no
	// answer: match (the default) or no_match.
	OnUnknown string `yaml:"on_unknown,omitempty" jsonschema:"enum=match,enum=no_match"`
}

// PIIRuleInput adds the earlier user turns and the request's tool results to
// the text a pii rule reads; Request false leaves the current user turn out, as
// SystemOneInput.Request does.
type PIIRuleInput struct {
	History     bool  `yaml:"history,omitempty"`
	ToolResults bool  `yaml:"tool_results,omitempty"`
	Request     *bool `yaml:"request,omitempty"`
}

// HallucinationRule matches claims in the model's answer that its grounding
// context (retrieved documents and tool results) does not support. It is a
// response-stage signal.
type HallucinationRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	Deployment  string            `yaml:"deployment,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
}

// RefusalRule matches an answer in which the model refused the request. It
// is a response-stage signal.
type RefusalRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	Deployment  string            `yaml:"deployment,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
}

// TurnRelationRule matches how the current user turn relates to the turns
// before it: continue, reask, shift or return. A rule without labels matches
// nothing itself; its conditions name one.
type TurnRelationRule struct {
	Name        string                `yaml:"name"`
	Description string                `yaml:"description,omitempty"`
	Labels      []string              `yaml:"labels,omitempty"`
	Input       TurnRelationRuleInput `yaml:"input,omitempty"`
	Deployment  string                `yaml:"deployment,omitempty"`
	TimeoutMs   int                   `yaml:"timeout_ms,omitempty"`
}

// TurnRelationRuleInput is how many turns before the current one a
// turn_relation rule reads (default 4).
type TurnRelationRuleInput struct {
	PreviousTurns int `yaml:"previous_turns,omitempty"`
}

// ComplexityRule matches how difficult the request is: easy, medium or
// hard. A rule without labels matches nothing itself; its conditions name
// one.
type ComplexityRule struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description,omitempty"`
	Labels      []string `yaml:"labels,omitempty"`
	Deployment  string   `yaml:"deployment,omitempty"`
	TimeoutMs   int      `yaml:"timeout_ms,omitempty"`
}

// ModalityRule matches the output the request asks for: text, image,
// text_image, image_edit, speech, audio, video or video_edit. A rule without
// labels matches nothing itself; its conditions name one.
type ModalityRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Labels      []string          `yaml:"labels,omitempty"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	Deployment  string            `yaml:"deployment,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
}
