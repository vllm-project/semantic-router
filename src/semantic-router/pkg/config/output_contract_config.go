package config

const (
	OutputContractTypeChoice                       = "choice"
	OutputContractTypeStructuredJSON               = "structured_json"
	OutputContractTypeReferenceSelect              = "reference_selection"
	OutputContractRenderModeValue                  = "value"
	OutputContractRenderModeTemplate               = "template"
	OutputContractExtractModeExact                 = "exact"
	OutputContractExtractModeJSONObject            = "json_object"
	OutputContractExtractSourceContent             = "content"
	OutputContractExtractSourceReasoningContent    = "reasoning_content"
	OutputContractExtractSourceCandidateResponses  = "candidate_responses"
	OutputContractJSONTerminalActionV1             = "terminal_action_v1"
	OutputContractReferenceIDFormatIndex           = "index"
	OutputContractReferenceIDFormatReferenceNumber = "reference_number"

	OutputContractPostprocessDereferenceSelectedReference = "dereference_selected_reference"
)

// OutputContractSpec is the router-executable counterpart to output_contract.
// output_contract remains model-facing prompt text; this typed spec controls
// runtime extraction, normalization, fallback, and post-processing.
type OutputContractSpec struct {
	Type        string                         `yaml:"type,omitempty" json:"type,omitempty"`
	ChoiceSet   *OutputContractChoiceSetSpec   `yaml:"choice_set,omitempty" json:"choice_set,omitempty"`
	JSONSchema  *OutputContractJSONSchemaSpec  `yaml:"json_schema,omitempty" json:"json_schema,omitempty"`
	Reference   *OutputContractReferenceSpec   `yaml:"reference,omitempty" json:"reference,omitempty"`
	Render      *OutputContractRenderSpec      `yaml:"render,omitempty" json:"render,omitempty"`
	Extract     *OutputContractExtractSpec     `yaml:"extract,omitempty" json:"extract,omitempty"`
	Normalize   *OutputContractNormalizeSpec   `yaml:"normalize,omitempty" json:"normalize,omitempty"`
	OnViolation *OutputContractViolationPolicy `yaml:"on_violation,omitempty" json:"on_violation,omitempty"`
	Postprocess []OutputContractPostprocess    `yaml:"postprocess,omitempty" json:"postprocess,omitempty"`
}

type OutputContractChoiceSetSpec struct {
	Values []string `yaml:"values,omitempty" json:"values,omitempty"`
}

type OutputContractJSONSchemaSpec struct {
	SchemaRef string `yaml:"schema_ref,omitempty" json:"schema_ref,omitempty"`
}

type OutputContractReferenceSpec struct {
	Source   string `yaml:"source,omitempty" json:"source,omitempty"`
	IDFormat string `yaml:"id_format,omitempty" json:"id_format,omitempty"`
}

type OutputContractRenderSpec struct {
	Mode     string `yaml:"mode,omitempty" json:"mode,omitempty"`
	Template string `yaml:"template,omitempty" json:"template,omitempty"`
}

type OutputContractExtractSpec struct {
	Mode    string   `yaml:"mode,omitempty" json:"mode,omitempty"`
	Sources []string `yaml:"sources,omitempty" json:"sources,omitempty"`
}

type OutputContractNormalizeSpec struct {
	FieldOrder []string          `yaml:"field_order,omitempty" json:"field_order,omitempty"`
	Defaults   map[string]string `yaml:"defaults,omitempty" json:"defaults,omitempty"`
}

type OutputContractViolationPolicy struct {
	Repair   bool   `yaml:"repair,omitempty" json:"repair,omitempty"`
	Fallback string `yaml:"fallback,omitempty" json:"fallback,omitempty"`
}

type OutputContractPostprocess struct {
	Type string `yaml:"type,omitempty" json:"type,omitempty"`
}
