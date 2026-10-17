package config

const SignalTypeAgenticFacts = "agentic_facts"

// AgenticFactsField enumerates the accepted-facts fields an
// AgenticFactsRule may match against.
const (
	AgenticFactsFieldDelegatedRole = "delegated_role"
	AgenticFactsFieldTaskPhase     = "task_phase"
)

// AgenticFactsRule matches validated, trusted agentic selection facts
// accepted at the request boundary. Unlike MetadataRule, Field is a closed
// set (AgenticFactsFieldDelegatedRole, AgenticFactsFieldTaskPhase) rather
// than an arbitrary caller-supplied key, because the accepted-facts schema
// exposes exactly these two matchable string fields.
type AgenticFactsRule struct {
	Name        string                `yaml:"name"`
	Description string                `yaml:"description,omitempty"`
	Field       string                `yaml:"field"`
	Predicate   AgenticFactsPredicate `yaml:"predicate"`
}

// AgenticFactsPredicate is a tagged union. Exactly one comparator must be set.
type AgenticFactsPredicate struct {
	Equals *string  `yaml:"equals,omitempty"`
	In     []string `yaml:"in,omitempty"`
}
