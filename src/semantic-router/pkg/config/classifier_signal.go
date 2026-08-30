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
	Name         string   `yaml:"name"`
	Description  string   `yaml:"description,omitempty"`
	Type         string   `yaml:"type"`
	Model        string   `yaml:"model,omitempty"`
	ModelPath    string   `yaml:"model_path,omitempty"`
	Labels       []string `yaml:"labels"`
	Instructions string   `yaml:"instructions,omitempty"`
	UseCPU       bool     `yaml:"use_cpu,omitempty"`
}
