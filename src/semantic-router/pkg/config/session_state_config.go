package config

// SessionStateFieldConfig is one typed field inside a SESSION_STATE declaration.
type SessionStateFieldConfig struct {
	Name     string `yaml:"name"`
	TypeName string `yaml:"type"`
}

// SessionStateConfig represents a SESSION_STATE declaration, naming the
// cross-turn fields that session-aware routing policies can reference.
type SessionStateConfig struct {
	Name   string                    `yaml:"name"`
	Fields []SessionStateFieldConfig `yaml:"fields,omitempty"`
}
