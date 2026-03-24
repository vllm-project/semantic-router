package config

// CategoryKBRule configures per-category knowledge base classification.
// Each rule points to a directory of JSON KB files (one per category) and
// a taxonomy file that maps categories to routing tiers.
type CategoryKBRule struct {
	Name              string  `yaml:"name"`
	KBDir             string  `yaml:"kb_dir"`
	TaxonomyPath      string  `yaml:"taxonomy_path,omitempty"`
	Threshold         float32 `yaml:"threshold"`
	SecurityThreshold float32 `yaml:"security_threshold,omitempty"`
}
