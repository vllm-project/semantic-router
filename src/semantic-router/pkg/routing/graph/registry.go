package graph

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extension"
)

// StepSpec is one authored step: an id, a registered type and the type's
// configuration as JSON.
type StepSpec struct {
	ID            string          `json:"id,omitempty"`
	Type          string          `json:"type"`
	Configuration json.RawMessage `json:"configuration,omitempty"`
}

// At locates an authored step, such as `steps[1].branches[0][2]`, in errors.
type At string

// Field locates a named part of the step at a.
func (a At) Field(name string) At { return At(string(a) + "." + name) }

// Index locates the i-th step of a sequence at a.
func (a At) Index(i int) At { return At(fmt.Sprintf("%s[%d]", a, i)) }

// Errorf scopes a message to a.
func (a At) Errorf(format string, args ...any) error {
	return fmt.Errorf("%s: %w", string(a), fmt.Errorf(format, args...))
}

// NodeType is a kind of graph step. Its payload's Go type is the schema of
// the step's configuration; Build makes the node from a decoded payload and
// builds any nested sequences through the Builder.
type NodeType struct {
	Description string
	// NewPayload returns an empty payload to decode the configuration into.
	NewPayload func() any
	// Strict rejects configuration fields the payload does not declare.
	Strict bool
	// Defaults, when set, fills a decoded payload's unset fields.
	Defaults func(payload any)
	// Validate, when set, checks a decoded payload after its defaults.
	Validate func(at At, payload any) error
	Build    func(b *Builder, at At, payload any) (Node, error)
	// Aliases are other accepted spellings of the type.
	Aliases []string
}

// Nodes holds the graph step types. The built-in types register first,
// through the same call as any other package.
var Nodes = newNodeRegistry()

func newNodeRegistry() *extension.Registry[NodeType] {
	registry := extension.NewRegistry[NodeType]("graph node")
	for _, entry := range builtinNodeTypes() {
		if err := registerNode(registry, entry.typ, entry.spec); err != nil {
			panic(err)
		}
	}
	return registry
}

// RegisterNode adds a step type, typically from an init function.
func RegisterNode(typ string, spec NodeType) error {
	return registerNode(Nodes, typ, spec)
}

func registerNode(registry *extension.Registry[NodeType], typ string, spec NodeType) error {
	if spec.NewPayload == nil || spec.Build == nil {
		return fmt.Errorf("graph node %q: a payload type and a builder are required", typ)
	}
	return registry.Register(typ, spec, spec.Aliases...)
}

// Builder builds authored steps into programs: it resolves step types in
// Nodes, decodes and checks their configuration, builds nested sequences and
// expands subgraphs, refusing cycles and duplicate ids.
type Builder struct {
	// Subgraphs are the reusable sequences that subgraph steps name.
	Subgraphs map[string][]StepSpec

	types     *extension.Registry[NodeType]
	expanding []string
	ids       map[string]At
	underPar  int
}

// NewBuilder returns a Builder that resolves subgraphs in subgraphs.
func NewBuilder(subgraphs map[string][]StepSpec) *Builder {
	return &Builder{Subgraphs: subgraphs, types: Nodes}
}

// Program builds a program from its authored steps.
func (b *Builder) Program(name string, steps []StepSpec, limits Limits) (*Program, error) {
	if err := limits.validate(); err != nil {
		return nil, At(name).Errorf("%w", err)
	}
	b.ids = map[string]At{}
	sequence, err := b.Sequence(At(name).Field("steps"), steps)
	if err != nil {
		return nil, err
	}
	if len(sequence) == 0 {
		return nil, At(name).Errorf("a graph needs at least one step")
	}
	return &Program{Name: name, Steps: sequence, Limits: limits}, nil
}

// Sequence builds the steps of one sequence.
func (b *Builder) Sequence(at At, specs []StepSpec) (Sequence, error) {
	sequence := make(Sequence, 0, len(specs))
	for i, spec := range specs {
		step, err := b.step(at.Index(i), spec)
		if err != nil {
			return nil, err
		}
		sequence = append(sequence, step)
	}
	return sequence, nil
}

func (b *Builder) step(at At, spec StepSpec) (Step, error) {
	typ := strings.TrimSpace(spec.Type)
	nodeType, ok := b.types.Lookup(typ)
	if !ok {
		return Step{}, at.Errorf("unknown step type %q (known: %s)", typ, strings.Join(b.types.Types(), ", "))
	}
	typ = b.types.Normalize(typ)
	id := strings.TrimSpace(spec.ID)
	if id == "" {
		id = string(at)
	} else if first, taken := b.ids[id]; taken {
		return Step{}, at.Errorf("step id %q is already used at %s", id, first)
	}
	if b.ids != nil {
		b.ids[id] = at
	}
	if typ == TypeRespond && b.underPar > 0 {
		return Step{}, at.Errorf("a respond step cannot run inside a parallel branch")
	}
	payload, err := decodePayload(nodeType, spec.Configuration)
	if err != nil {
		return Step{}, at.Errorf("%w", err)
	}
	if nodeType.Defaults != nil {
		nodeType.Defaults(payload)
	}
	if nodeType.Validate != nil {
		if invalid := nodeType.Validate(at, payload); invalid != nil {
			return Step{}, invalid
		}
	}
	node, err := nodeType.Build(b, at, payload)
	if err != nil {
		return Step{}, err
	}
	return Step{ID: id, Type: typ, Node: node}, nil
}

func decodePayload(nodeType NodeType, raw json.RawMessage) (any, error) {
	payload := nodeType.NewPayload()
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return payload, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if nodeType.Strict {
		decoder.DisallowUnknownFields()
	}
	if err := decoder.Decode(payload); err != nil {
		return nil, fmt.Errorf("decode the configuration: %w", err)
	}
	return payload, nil
}

// Parallel builds the sequences of a parallel step's branches, where a
// respond step is refused.
func (b *Builder) Parallel(at At, branches [][]StepSpec) ([]Sequence, error) {
	b.underPar++
	defer func() { b.underPar-- }()
	out := make([]Sequence, len(branches))
	for i, branch := range branches {
		sequence, err := b.Sequence(at.Index(i), branch)
		if err != nil {
			return nil, err
		}
		out[i] = sequence
	}
	return out, nil
}

// Subgraph builds the named subgraph's steps, refusing a subgraph that
// includes itself.
func (b *Builder) Subgraph(at At, name string) (Sequence, error) {
	specs, ok := b.Subgraphs[name]
	if !ok {
		return nil, at.Errorf("unknown subgraph %q", name)
	}
	for _, open := range b.expanding {
		if open == name {
			return nil, at.Errorf("subgraph %q includes itself (%s)", name, strings.Join(append(b.expanding, name), " -> "))
		}
	}
	b.expanding = append(b.expanding, name)
	defer func() { b.expanding = b.expanding[:len(b.expanding)-1] }()
	// A subgraph's own ids repeat at every use, so they are scoped to it.
	ids := b.ids
	b.ids = nil
	defer func() { b.ids = ids }()
	return b.Sequence(At("subgraphs."+name), specs)
}

func (l Limits) validate() error {
	switch {
	case l.Timeout < 0:
		return fmt.Errorf("timeout must not be negative")
	case l.MaxHops < 0:
		return fmt.Errorf("max_hops must not be negative")
	case l.MaxTokens < 0:
		return fmt.Errorf("max_tokens must not be negative")
	case l.MaxCost < 0:
		return fmt.Errorf("max_cost must not be negative")
	}
	return nil
}
