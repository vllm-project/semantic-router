package config

const SignalTypeClassifier = "classifier"

// ClassifierSignalRule exposes a reusable label-score classifier as a signal.
// Supported types are "local" for native sequence-classification models and
// "llm" for configured external chat classifiers.
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
