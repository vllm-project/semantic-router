package config

// AlgorithmBudget bounds only the selected algorithm's inference exchanges,
// including transport retries, and its execution deadline. Signal evaluation
// precedes this budget and retains its own timeouts and context cancellation.
// Calls inside an opaque external provider are not visible to the local ledger.
type AlgorithmBudget struct {
	Deadline string `yaml:"deadline" jsonschema:"required"`
	MaxCalls int    `yaml:"max_calls" jsonschema:"required,minimum=1"`
}

// NativeQualityConfig states the acceptance rule for a complete native
// response. A calibrated risk is a target backed by matching evaluation
// evidence, not a probability inferred from a model's raw score.
type NativeQualityConfig struct {
	Type        string            `yaml:"type" jsonschema:"required,enum=uncalibrated,enum=calibrated"`
	Acceptance  *NativeAcceptance `yaml:"acceptance,omitempty"`
	Calibration string            `yaml:"calibration,omitempty"`
	Loss        string            `yaml:"loss,omitempty"`
	MaxRisk     *float64          `yaml:"max_risk,omitempty"`
}

// NativeAcceptance combines typed observation predicates with AND. Missing
// observations remain unknown; every required answer needs rule coverage.
type NativeAcceptance struct {
	Rules []NativeAcceptanceRule `yaml:"rules" jsonschema:"required,minItems=1"`
}

// NativeAcceptanceRule applies to every matching question in the selected
// state, or all states when State is omitted. Question and QuestionType narrow
// the scope together; at least one is required. Score values are not confidence.
type NativeAcceptanceRule struct {
	Question     string           `yaml:"question,omitempty"`
	QuestionType string           `yaml:"question_type,omitempty"`
	State        *string          `yaml:"state,omitempty"`
	Field        string           `yaml:"field" jsonschema:"required,enum=confidence,enum=top_probability,enum=probability_margin"`
	Predicate    NumericPredicate `yaml:"predicate" jsonschema:"required"`
}

// CascadeStage is one operator-authored model action. Model names a provider
// alias in ModelRefs; replicas of that model belong in its backend_refs. Cascade
// runs these actions in authored order.
type CascadeStage struct {
	Name         string                  `yaml:"name" jsonschema:"required"`
	Kind         string                  `yaml:"kind" jsonschema:"required,enum=native,enum=judge"`
	Model        string                  `yaml:"model" jsonschema:"required"`
	Enabled      *bool                   `yaml:"enabled,omitempty"`
	Timeout      string                  `yaml:"timeout,omitempty"`
	Accept       *NativeAcceptance       `yaml:"accept,omitempty"`
	Generation   *NativeGenerationConfig `yaml:"generation,omitempty"`
	Instructions string                  `yaml:"instructions,omitempty"`
}

func (s CascadeStage) IsEnabled() bool { return s.Enabled == nil || *s.Enabled }

// NativeGenerationConfig makes the LLM exception's output budget explicit.
// It does not authorize replacing native probability or span semantics.
type NativeGenerationConfig struct {
	MaxOutputTokens int `yaml:"max_output_tokens" jsonschema:"required,minimum=1"`
}

// CalibrationArtifact is an immutable evaluation resource. The execution
// owner checks its bytes and applicability to the model, task, loss and
// conditional arrival population before reporting calibrated acceptance.
type CalibrationArtifact struct {
	Name   string `yaml:"name" jsonschema:"required"`
	Source string `yaml:"source" jsonschema:"required"`
	SHA256 string `yaml:"sha256" jsonschema:"required,pattern=^[a-fA-F0-9]{64}$"`
}

func (a *AlgorithmConfig) IsNative() bool {
	return a != nil && a.Type == DecisionAlgorithmCascade
}
