package graph

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"text/template"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extension"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// The built-in step types.
const (
	TypeCall      = "call"
	TypeParallel  = "parallel"
	TypeAggregate = "aggregate"
	TypeBranch    = "branch"
	TypeLoop      = "loop"
	TypeTransform = "transform"
	TypeRespond   = "respond"
	TypeSubgraph  = "subgraph"
)

// CallConfig configures a call step.
type CallConfig struct {
	Model string `json:"model"`
	// Decision names the decision whose plugin chain the hop runs; empty is
	// the request's.
	Decision string `json:"decision,omitempty"`
	// Fields are request fields the call sets, such as temperature.
	Fields map[string]json.RawMessage `json:"fields,omitempty"`
	// Fallback overrides the decision's cross-model fallback for this call.
	Fallback *fallback.FallbackOverride `json:"fallback,omitempty"`
}

// ParallelConfig configures a parallel step.
type ParallelConfig struct {
	Branches       [][]StepSpec `json:"branches"`
	MaxConcurrency int          `json:"max_concurrency,omitempty"`
	FirstK         int          `json:"first_k,omitempty"`
	MinSuccess     int          `json:"min_success,omitempty"`
	OnError        OnError      `json:"on_error,omitempty"`
}

// AggregateConfig configures an aggregate step: a strategy registered in
// Aggregators, and the strategy's options.
type AggregateConfig struct {
	Strategy string          `json:"strategy"`
	Options  json.RawMessage `json:"options,omitempty"`
}

// BranchConfig configures a branch step.
type BranchConfig struct {
	Cases []CaseConfig `json:"cases"`
	Else  []StepSpec   `json:"else,omitempty"`
}

// CaseConfig is one alternative of a branch step.
type CaseConfig struct {
	When ConditionConfig `json:"when"`
	Then []StepSpec      `json:"then"`
}

// LoopConfig configures a loop step.
type LoopConfig struct {
	Body      []StepSpec       `json:"body"`
	Until     *ConditionConfig `json:"until,omitempty"`
	MaxRounds int              `json:"max_rounds"`
}

// TransformConfig configures a transform step: a transformer registered in
// Transformers, and its options.
type TransformConfig struct {
	Transformer string          `json:"transformer"`
	Options     json.RawMessage `json:"options,omitempty"`
}

// RespondConfig configures a respond step.
type RespondConfig struct {
	// Final names the model whose answer streams to the client as the
	// gateway's final call.
	Final string `json:"final,omitempty"`
}

// SubgraphConfig configures a subgraph step.
type SubgraphConfig struct {
	Name string `json:"name"`
}

// named is a registration: a type name and its spec.
type named[S any] struct {
	typ  string
	spec S
}

func builtinNodeTypes() []named[NodeType] {
	return []named[NodeType]{
		{TypeCall, NodeType{
			Description: "Call one model with the request the steps before built.",
			NewPayload:  func() any { return &CallConfig{} },
			Strict:      true,
			Build: func(_ *Builder, at At, payload any) (Node, error) {
				cfg := payload.(*CallConfig)
				if strings.TrimSpace(cfg.Model) == "" {
					return nil, at.Errorf("model is required")
				}
				if err := cfg.Fallback.Validate(); err != nil {
					return nil, at.Errorf("fallback: %w", err)
				}
				return &Call{Model: cfg.Model, Decision: cfg.Decision, Fields: cfg.Fields, Fallback: cfg.Fallback}, nil
			},
		}},
		{TypeParallel, NodeType{
			Description: "Run branches concurrently, with a concurrency cap and an optional first-k rule.",
			NewPayload:  func() any { return &ParallelConfig{} },
			Strict:      true,
			Build:       buildParallel,
		}},
		{TypeAggregate, NodeType{
			Description: "Combine the latest results with a named strategy.",
			NewPayload:  func() any { return &AggregateConfig{} },
			Strict:      true,
			Build: func(_ *Builder, at At, payload any) (Node, error) {
				cfg := payload.(*AggregateConfig)
				strategy, ok := Aggregators.Lookup(cfg.Strategy)
				if !ok {
					return nil, at.Errorf("unknown aggregate strategy %q (known: %s)", cfg.Strategy, strings.Join(Aggregators.Types(), ", "))
				}
				aggregator, err := strategy.New(cfg.Options)
				if err != nil {
					return nil, at.Errorf("strategy %q: %w", cfg.Strategy, err)
				}
				return &Aggregate{Strategy: aggregator}, nil
			},
		}},
		{TypeBranch, NodeType{
			Description: "Run the first case whose condition holds.",
			NewPayload:  func() any { return &BranchConfig{} },
			Strict:      true,
			Build:       buildBranch,
		}},
		{TypeLoop, NodeType{
			Description: "Repeat a body until a condition holds or a round limit is reached.",
			NewPayload:  func() any { return &LoopConfig{} },
			Strict:      true,
			Build:       buildLoop,
		}},
		{TypeTransform, NodeType{
			Description: "Rewrite the request the next call sends.",
			NewPayload:  func() any { return &TransformConfig{} },
			Strict:      true,
			Build: func(_ *Builder, at At, payload any) (Node, error) {
				cfg := payload.(*TransformConfig)
				kind, ok := Transformers.Lookup(cfg.Transformer)
				if !ok {
					return nil, at.Errorf("unknown transformer %q (known: %s)", cfg.Transformer, strings.Join(Transformers.Types(), ", "))
				}
				transformer, err := kind.New(cfg.Options)
				if err != nil {
					return nil, at.Errorf("transformer %q: %w", cfg.Transformer, err)
				}
				return &Transform{Transformer: transformer}, nil
			},
		}},
		{TypeRespond, NodeType{
			Description: "Answer the request with the latest result, or stream a final model call.",
			NewPayload:  func() any { return &RespondConfig{} },
			Strict:      true,
			Build: func(_ *Builder, _ At, payload any) (Node, error) {
				return &Respond{Final: payload.(*RespondConfig).Final}, nil
			},
		}},
		{TypeSubgraph, NodeType{
			Description: "Run a named, reusable sequence of steps.",
			NewPayload:  func() any { return &SubgraphConfig{} },
			Strict:      true,
			Build: func(b *Builder, at At, payload any) (Node, error) {
				name := payload.(*SubgraphConfig).Name
				steps, err := b.Subgraph(at, name)
				if err != nil {
					return nil, err
				}
				return &Subgraph{Name: name, Steps: steps}, nil
			},
		}},
	}
}

func buildParallel(b *Builder, at At, payload any) (Node, error) {
	cfg := payload.(*ParallelConfig)
	switch {
	case len(cfg.Branches) == 0:
		return nil, at.Errorf("branches are required")
	case cfg.MaxConcurrency < 0, cfg.FirstK < 0, cfg.MinSuccess < 0:
		return nil, at.Errorf("max_concurrency, first_k and min_success must not be negative")
	case cfg.FirstK > len(cfg.Branches), cfg.MinSuccess > len(cfg.Branches):
		return nil, at.Errorf("first_k and min_success cannot exceed the %d branches", len(cfg.Branches))
	case cfg.OnError != "" && cfg.OnError != OnErrorFail && cfg.OnError != OnErrorSkip:
		return nil, at.Errorf("on_error must be %q or %q", OnErrorFail, OnErrorSkip)
	}
	branches, err := b.Parallel(at.Field("branches"), cfg.Branches)
	if err != nil {
		return nil, err
	}
	onError := cfg.OnError
	if onError == "" {
		onError = OnErrorFail
	}
	return &Parallel{
		Branches: branches, MaxConcurrency: cfg.MaxConcurrency,
		FirstK: cfg.FirstK, MinSuccess: cfg.MinSuccess, OnError: onError,
	}, nil
}

func buildBranch(b *Builder, at At, payload any) (Node, error) {
	cfg := payload.(*BranchConfig)
	if len(cfg.Cases) == 0 {
		return nil, at.Errorf("cases are required")
	}
	node := &Branch{Cases: make([]Case, len(cfg.Cases))}
	for i, c := range cfg.Cases {
		caseAt := at.Field("cases").Index(i)
		when, err := c.When.build(caseAt.Field("when"))
		if err != nil {
			return nil, err
		}
		then, err := b.Sequence(caseAt.Field("then"), c.Then)
		if err != nil {
			return nil, err
		}
		node.Cases[i] = Case{When: when, Then: then}
	}
	var err error
	node.Else, err = b.Sequence(at.Field("else"), cfg.Else)
	return node, err
}

func buildLoop(b *Builder, at At, payload any) (Node, error) {
	cfg := payload.(*LoopConfig)
	if cfg.MaxRounds < 1 {
		return nil, at.Errorf("max_rounds must be at least 1")
	}
	body, err := b.Sequence(at.Field("body"), cfg.Body)
	if err != nil {
		return nil, err
	}
	if len(body) == 0 {
		return nil, at.Errorf("body needs at least one step")
	}
	node := &Loop{Body: body, MaxRounds: cfg.MaxRounds}
	if cfg.Until != nil {
		if node.Until, err = cfg.Until.build(at.Field("until")); err != nil {
			return nil, err
		}
	}
	return node, nil
}

// AggregateStrategy is a named way to combine results.
type AggregateStrategy struct {
	Description string
	// New builds the aggregator from its options, which may be empty.
	New func(options json.RawMessage) (Aggregator, error)
}

// TransformerType is a named way to rewrite the state.
type TransformerType struct {
	Description string
	New         func(options json.RawMessage) (Transformer, error)
}

// Aggregators and Transformers hold the aggregate strategies and the
// transformers. Other packages register more, such as the Looper's.
var (
	Aggregators  = newStrategyRegistry("graph aggregate strategy", builtinAggregators())
	Transformers = newStrategyRegistry("graph transformer", builtinTransformers())
)

func newStrategyRegistry[S any](kind string, builtins []named[S]) *extension.Registry[S] {
	registry := extension.NewRegistry[S](kind)
	for _, entry := range builtins {
		registry.MustRegister(entry.typ, entry.spec)
	}
	return registry
}

func builtinAggregators() []named[AggregateStrategy] {
	static := func(description string, aggregator Aggregator) AggregateStrategy {
		return AggregateStrategy{Description: description, New: func(json.RawMessage) (Aggregator, error) {
			return aggregator, nil
		}}
	}
	return []named[AggregateStrategy]{
		{"first", static("Keep the first successful result.", First{})},
		{"vote", static("Keep the answer most results agree on.", Vote{})},
		{"concat", AggregateStrategy{Description: "Join the results' texts into one answer.", New: newConcat}},
		{"choices", static("Answer with one choice per result.", Choices{})},
	}
}

func newConcat(options json.RawMessage) (Aggregator, error) {
	var opts struct {
		Separator string `json:"separator"`
	}
	if err := decodeOptions(options, &opts); err != nil {
		return nil, err
	}
	return Concat{Separator: opts.Separator}, nil
}

func builtinTransformers() []named[TransformerType] {
	return []named[TransformerType]{
		{"system_prompt", TransformerType{Description: "Set the system prompt.", New: newSystemPrompt}},
		{"append_results", TransformerType{Description: "Add the latest results to the conversation.", New: newAppendResults}},
		{"prompt", TransformerType{Description: "Rewrite the last user message from a template.", New: newPrompt}},
	}
}

func newSystemPrompt(options json.RawMessage) (Transformer, error) {
	var opts struct {
		Content string     `json:"content"`
		Mode    SystemMode `json:"mode"`
	}
	if err := decodeOptions(options, &opts); err != nil {
		return nil, err
	}
	if opts.Mode == "" {
		opts.Mode = SystemReplace
	}
	if opts.Mode != SystemReplace && opts.Mode != SystemPrepend {
		return nil, fmt.Errorf("mode must be %q or %q", SystemReplace, SystemPrepend)
	}
	return SystemPrompt{Content: opts.Content, Mode: opts.Mode}, nil
}

func newAppendResults(options json.RawMessage) (Transformer, error) {
	var opts struct {
		Role string `json:"role"`
	}
	if err := decodeOptions(options, &opts); err != nil {
		return nil, err
	}
	return AppendResults{Role: opts.Role}, nil
}

func newPrompt(options json.RawMessage) (Transformer, error) {
	var opts struct {
		Template string `json:"template"`
	}
	if err := decodeOptions(options, &opts); err != nil {
		return nil, err
	}
	if strings.TrimSpace(opts.Template) == "" {
		return nil, fmt.Errorf("template is required")
	}
	parsed, err := template.New("prompt").Option("missingkey=error").Parse(opts.Template)
	if err != nil {
		return nil, fmt.Errorf("template: %w", err)
	}
	return Prompt{Template: parsed}, nil
}

func decodeOptions(options json.RawMessage, target any) error {
	if len(options) == 0 {
		return nil
	}
	decoder := json.NewDecoder(strings.NewReader(string(options)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode the options: %w", err)
	}
	return nil
}

// ConditionConfig is an authored condition: set exactly one field.
type ConditionConfig struct {
	Succeeded      *bool             `json:"succeeded,omitempty"`
	ContentMatches string            `json:"content_matches,omitempty"`
	Signal         *SignalRef        `json:"signal,omitempty"`
	LogprobAtLeast *float64          `json:"logprob_at_least,omitempty"`
	All            []ConditionConfig `json:"all,omitempty"`
	Any            []ConditionConfig `json:"any,omitempty"`
	Not            *ConditionConfig  `json:"not,omitempty"`
}

// SignalRef names a signal by its type and name.
type SignalRef struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

func (c ConditionConfig) build(at At) (Condition, error) {
	var built []Condition
	if c.Succeeded != nil {
		condition := Succeeded()
		if !*c.Succeeded {
			condition = Not(condition)
		}
		built = append(built, condition)
	}
	if c.ContentMatches != "" {
		pattern, err := regexp.Compile(c.ContentMatches)
		if err != nil {
			return nil, at.Errorf("content_matches: %w", err)
		}
		built = append(built, ContentMatches(pattern))
	}
	if c.Signal != nil {
		if c.Signal.Type == "" || c.Signal.Name == "" {
			return nil, at.Errorf("signal needs a type and a name")
		}
		built = append(built, SignalMatched(c.Signal.Type, c.Signal.Name))
	}
	if c.LogprobAtLeast != nil {
		built = append(built, LogprobAtLeast(*c.LogprobAtLeast))
	}
	for _, group := range []struct {
		name  string
		items []ConditionConfig
		join  func(...Condition) Condition
	}{{"all", c.All, All}, {"any", c.Any, Any}} {
		if group.items == nil {
			continue
		}
		parts := make([]Condition, len(group.items))
		for i, item := range group.items {
			part, err := item.build(at.Field(group.name).Index(i))
			if err != nil {
				return nil, err
			}
			parts[i] = part
		}
		built = append(built, group.join(parts...))
	}
	if c.Not != nil {
		inner, err := c.Not.build(at.Field("not"))
		if err != nil {
			return nil, err
		}
		built = append(built, Not(inner))
	}
	if len(built) != 1 {
		return nil, at.Errorf("a condition sets exactly one of succeeded, content_matches, signal, logprob_at_least, all, any or not")
	}
	return built[0], nil
}
