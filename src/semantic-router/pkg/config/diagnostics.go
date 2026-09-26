package config

// DiagnosticContractVersion is the versioned validate/diff response contract.
const DiagnosticContractVersion = "v1"

// DiffEntryLimit bounds added + removed + changed entries in one evaluate call.
const DiffEntryLimit = 500

const (
	DiagnosticYAMLParseError   = "YAML_PARSE_ERROR"
	DiagnosticUnknownField     = "CONFIG_UNKNOWN_FIELD"
	DiagnosticValidationError  = "CONFIG_VALIDATION_ERROR"
	DiagnosticReferenceError   = "CONFIG_REFERENCE_ERROR"
	DiagnosticCycleError       = "CONFIG_CYCLE_ERROR"
	DiagnosticConflict         = "CONFIG_CONFLICT"
	DiagnosticQuorumError      = "CONFIG_QUORUM_ERROR"
	DiagnosticBudgetError      = "CONFIG_BUDGET_ERROR"
	DiagnosticCapabilityError  = "CONFIG_CAPABILITY_ERROR"
	DiagnosticFallbackError    = "CONFIG_FALLBACK_ERROR"
	DiagnosticNoActiveSnapshot = "CONFIG_NO_ACTIVE_SNAPSHOT"
)

const (
	SeverityError   = "error"
	SeverityWarning = "warning"
)

const (
	StageParse     = "parse"
	StageNormalize = "normalize"
	StageResolve   = "resolve"
	StageValidate  = "validate"
)

// Diagnostic is one field-addressable validate/diff finding.
type Diagnostic struct {
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Resource string `json:"resource,omitempty"`
	Recipe   string `json:"recipe,omitempty"`
	Stage    string `json:"stage"`
	Field    string `json:"field,omitempty"`
	Message  string `json:"message"`
}

// DiffEntry is one schema-aware added, removed, or changed field.
type DiffEntry struct {
	Field string `json:"field"`
	Old   any    `json:"old,omitempty"`
	New   any    `json:"new,omitempty"`
}

// ConfigDiff is a bounded active-versus-candidate structured diff.
type ConfigDiff struct {
	Added     []DiffEntry `json:"added"`
	Removed   []DiffEntry `json:"removed"`
	Changed   []DiffEntry `json:"changed"`
	Truncated bool        `json:"truncated"`
}

// EvaluateOptions controls optional active-snapshot comparison.
type EvaluateOptions struct {
	CompareToActive bool
	ActiveYAML      []byte
}

// EvaluateResult is the canonical side-effect-free validate/diff output.
type EvaluateResult struct {
	Valid           bool
	ContractVersion string
	NormalizedYAML  string
	Errors          []Diagnostic
	Warnings        []Diagnostic
	Diff            *ConfigDiff
}
