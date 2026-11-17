package config

// AlgorithmBudget is every executor's deadline and inference-call limit. On
// Chat it bounds the executor's run: its model calls and their wall-clock
// time. On System One both are measured from request arrival: before
// selection, the largest deadline and call limit among the recipe's native
// decisions bound model-backed signals, whose own timeouts remain upper
// bounds; the selected decision's budget then tightens the same deadline and
// ledger, so signal calls, stages, judges and transport retries all count.
// Calls inside an opaque external provider are not visible to the local
// ledger.
type AlgorithmBudget struct {
	Deadline string `yaml:"deadline" jsonschema:"required"`
	MaxCalls int    `yaml:"max_calls" jsonschema:"required,minimum=1"`
}

// Kinds of an escalate stage on System One.
const (
	EscalateStageNative = "native"
	EscalateStageJudge  = "judge"
)

// Types of an escalate quality policy on System One.
const (
	EscalateQualityUncalibrated = "uncalibrated"
	EscalateQualityCalibrated   = "calibrated"
)

// EscalateQuality is the policy every answer a System One escalation accepts
// meets. An uncalibrated policy may require Accept rules of the whole answer
// bundle; a calibrated one accepts by certified evaluation evidence. A
// calibrated risk is a target backed by matching evidence, not a probability
// inferred from a model's raw score.
type EscalateQuality struct {
	Type        string          `yaml:"type,omitempty" jsonschema:"enum=uncalibrated,enum=calibrated"`
	Accept      *EscalateAccept `yaml:"accept,omitempty"`
	Calibration string          `yaml:"calibration,omitempty"`
	Loss        string          `yaml:"loss,omitempty"`
	MaxRisk     *float64        `yaml:"max_risk,omitempty"`
}

// EffectiveType is the policy's type: uncalibrated unless declared.
func (q *EscalateQuality) EffectiveType() string {
	if q == nil || q.Type == "" {
		return EscalateQualityUncalibrated
	}
	return q.Type
}

// NativeAcceptanceRule applies to every matching question in the selected
// state, or all states when State is omitted. Question and QuestionType narrow
// the scope together; at least one is required. Score values are not
// confidence. The predicate bounds the observed field.
type NativeAcceptanceRule struct {
	Question         string  `yaml:"question,omitempty"`
	QuestionType     string  `yaml:"question_type,omitempty"`
	State            *string `yaml:"state,omitempty"`
	Field            string  `yaml:"field" jsonschema:"required,enum=confidence,enum=top_probability,enum=probability_margin"`
	NumericPredicate `yaml:",inline"`
}

// EscalateStage is one operator-authored System One action: a native model
// answering the typed request, or an LLM judge choosing among the complete
// answers before it. Model names a provider alias in ModelRefs; replicas of
// that model belong in its backend_refs. Stages run in authored order.
type EscalateStage struct {
	Name         string                  `yaml:"name,omitempty"`
	Kind         string                  `yaml:"kind,omitempty" jsonschema:"enum=native,enum=judge"`
	Model        string                  `yaml:"model" jsonschema:"required"`
	Enabled      *bool                   `yaml:"enabled,omitempty"`
	Timeout      string                  `yaml:"timeout,omitempty"`
	Accept       *EscalateAccept         `yaml:"accept,omitempty"`
	Generation   *NativeGenerationConfig `yaml:"generation,omitempty"`
	Instructions string                  `yaml:"instructions,omitempty"`
}

func (s EscalateStage) IsEnabled() bool { return s.Enabled == nil || *s.Enabled }

// StageName names the stage in evidence, metrics and the routing outcome:
// its name, or its model when it has none.
func (s EscalateStage) StageName() string {
	if s.Name != "" {
		return s.Name
	}
	return s.Model
}

// StageKind is native unless the stage is a judge.
func (s EscalateStage) StageKind() string {
	if s.Kind == "" {
		return EscalateStageNative
	}
	return s.Kind
}

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

// IsNative reports an escalation of a System One recipe: one that runs
// authored stages over native answers.
func (a *AlgorithmConfig) IsNative() bool {
	return a != nil && a.Type == DecisionAlgorithmEscalate && a.Escalate != nil && len(a.Escalate.Stages) > 0
}

// NativeStages are the System One stages of a native escalation.
func (a *AlgorithmConfig) NativeStages() []EscalateStage {
	if !a.IsNative() {
		return nil
	}
	return a.Escalate.Stages
}

// NativeQuality is the quality policy of a native escalation.
func (a *AlgorithmConfig) NativeQuality() *EscalateQuality {
	if !a.IsNative() {
		return nil
	}
	return a.Escalate.Quality
}
