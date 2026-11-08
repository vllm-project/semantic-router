package config

const SignalTypeClassifier = "classifier"

// Backend types a ClassifierSignalRule may declare. They are named here rather
// than spelled inline so the validator and the builder cannot drift apart.
const (
	ClassifierSignalTypeLocal              = "local"
	ClassifierSignalTypeLLM                = "llm"
	ClassifierSignalTypeSequenceClassifier = "sequence_classifier"
)

// ClassifierSignalRule exposes a reusable label-score classifier as a signal.
// Supported types are "local" for native sequence-classification models, "llm"
// for configured external chat classifiers, and "sequence_classifier" for a
// remote sequence classifier reached over the shared http_classify contract.
type ClassifierSignalRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
	// Deployment names the deployment that answers the rule's question; its
	// provider decides how it is asked. Without one, the rule follows its
	// recipe's and the global classifier bindings, then the decision model.
	Deployment string `yaml:"deployment,omitempty"`
	// Type, Model, ModelPath and UseCPU are the deprecated way to choose the
	// rule's model; they are read as a deployment.
	Type             string   `yaml:"type,omitempty"`
	Model            string   `yaml:"model,omitempty"`
	ModelPath        string   `yaml:"model_path,omitempty"`
	Labels           []string `yaml:"labels"`
	Instructions     string   `yaml:"instructions,omitempty"`
	DisableRationale bool     `yaml:"disable_rationale,omitempty"`
	UseCPU           bool     `yaml:"use_cpu,omitempty"`
	// legacyType is the deprecated type the rule declared before it was
	// translated, which still constrains a binding as it did.
	legacyType string
}
