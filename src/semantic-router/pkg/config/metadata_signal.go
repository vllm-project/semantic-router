package config

const SignalTypeMetadata = "metadata"

// MetadataRule matches caller-provided, untrusted request metadata.
// Metadata must never be used as an authorization identity source.
type MetadataRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Key         string            `yaml:"key"`
	Predicate   MetadataPredicate `yaml:"predicate"`
}

// MetadataPredicate is a tagged union. Exactly one comparator must be set.
type MetadataPredicate struct {
	Equals *string  `yaml:"equals,omitempty"`
	In     []string `yaml:"in,omitempty"`
	Exists *bool    `yaml:"exists,omitempty"`
}
