package config

const SignalTypeAction = "action"

// The fixed action vocabulary. Every request carries exactly one of these,
// read from its latest user message; other means no action was named.
const (
	ActionGenerate = "generate"
	ActionExplain  = "explain"
	ActionFix      = "fix"
	ActionRefactor = "refactor"
	ActionTest     = "test"
	ActionOther    = "other"
)

// ActionRule makes one action from the fixed vocabulary referenceable by
// decisions. The rule name is the action, the same way a domain rule name is
// the category.
type ActionRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
}

// SupportedActions lists the action names an ActionRule may declare.
func SupportedActions() []string {
	return []string{ActionGenerate, ActionExplain, ActionFix, ActionRefactor, ActionTest, ActionOther}
}
