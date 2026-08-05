package config

import (
	"fmt"
	"strings"
)

func validateDecisionOutputContractSpec(decision Decision) error {
	spec := decision.OutputContractSpec
	if spec == nil {
		return nil
	}
	context := fmt.Sprintf("decision '%s': output_contract_spec", decision.Name)
	if err := validateOutputContractSpecType(spec, context); err != nil {
		return err
	}
	return validateOutputContractPostprocess(spec, context)
}

func validateOutputContractSpecType(spec *OutputContractSpec, context string) error {
	switch strings.TrimSpace(spec.Type) {
	case "":
		if len(spec.Postprocess) == 0 {
			return fmt.Errorf("%s: type or postprocess must be specified", context)
		}
	case OutputContractTypeChoice:
		return validateChoiceOutputContractSpec(spec, context)
	case OutputContractTypeStructuredJSON:
		return validateStructuredJSONOutputContractSpec(spec, context)
	case OutputContractTypeReferenceSelect:
		return validateReferenceSelectionOutputContractSpec(spec, context)
	default:
		return fmt.Errorf("%s: unsupported type %q", context, spec.Type)
	}
	return validateOutputContractExtractSpec(spec.Extract, context)
}

func validateChoiceOutputContractSpec(spec *OutputContractSpec, context string) error {
	if spec.ChoiceSet == nil || len(spec.ChoiceSet.Values) == 0 {
		return fmt.Errorf("%s: choice type requires choice_set.values", context)
	}
	seen := map[string]bool{}
	for i, value := range spec.ChoiceSet.Values {
		normalized := strings.TrimSpace(value)
		if normalized == "" {
			return fmt.Errorf("%s: choice_set.values[%d] cannot be empty", context, i)
		}
		key := strings.ToLower(normalized)
		if seen[key] {
			return fmt.Errorf("%s: choice_set.values[%d] duplicates %q", context, i, normalized)
		}
		seen[key] = true
	}
	if err := validateOutputContractExtractSpec(spec.Extract, context); err != nil {
		return err
	}
	if err := validateOutputContractExactExtractMode(spec.Extract, context); err != nil {
		return err
	}
	return validateOutputContractRenderSpec(spec.Render, context)
}

func validateStructuredJSONOutputContractSpec(spec *OutputContractSpec, context string) error {
	if spec.JSONSchema == nil || strings.TrimSpace(spec.JSONSchema.SchemaRef) == "" {
		return fmt.Errorf("%s: structured_json type requires json_schema.schema_ref", context)
	}
	if strings.TrimSpace(spec.JSONSchema.SchemaRef) != OutputContractJSONTerminalActionV1 {
		return fmt.Errorf("%s: unsupported json_schema.schema_ref %q", context, spec.JSONSchema.SchemaRef)
	}
	return validateOutputContractExtractSpec(spec.Extract, context)
}

func validateReferenceSelectionOutputContractSpec(spec *OutputContractSpec, context string) error {
	if spec.Reference != nil {
		switch strings.TrimSpace(spec.Reference.Source) {
		case "", OutputContractExtractSourceCandidateResponses:
		default:
			return fmt.Errorf("%s: reference.source must be candidate_responses when set", context)
		}
		switch strings.TrimSpace(spec.Reference.IDFormat) {
		case "", OutputContractReferenceIDFormatIndex, OutputContractReferenceIDFormatReferenceNumber:
		default:
			return fmt.Errorf("%s: unsupported reference.id_format %q", context, spec.Reference.IDFormat)
		}
	}
	if err := validateOutputContractExtractSpec(spec.Extract, context); err != nil {
		return err
	}
	return validateOutputContractExactExtractMode(spec.Extract, context)
}

func validateOutputContractExtractSpec(extract *OutputContractExtractSpec, context string) error {
	if extract == nil {
		return nil
	}
	switch strings.TrimSpace(extract.Mode) {
	case "", OutputContractExtractModeExact, OutputContractExtractModeJSONObject:
	default:
		return fmt.Errorf("%s: unsupported extract.mode %q", context, extract.Mode)
	}
	seen := map[string]bool{}
	for i, source := range extract.Sources {
		source = strings.TrimSpace(source)
		switch source {
		case OutputContractExtractSourceContent, OutputContractExtractSourceReasoningContent, OutputContractExtractSourceCandidateResponses:
		default:
			return fmt.Errorf("%s: unsupported extract.sources[%d] %q", context, i, source)
		}
		if seen[source] {
			return fmt.Errorf("%s: extract.sources[%d] duplicates %q", context, i, source)
		}
		seen[source] = true
	}
	return nil
}

func validateOutputContractExactExtractMode(extract *OutputContractExtractSpec, context string) error {
	if extract == nil {
		return nil
	}
	mode := strings.TrimSpace(extract.Mode)
	if mode == "" || mode == OutputContractExtractModeExact {
		return nil
	}
	return fmt.Errorf("%s: extract.mode %q is only supported for structured_json contracts", context, mode)
}

func validateOutputContractRenderSpec(render *OutputContractRenderSpec, context string) error {
	if render == nil {
		return nil
	}
	switch strings.TrimSpace(render.Mode) {
	case "", OutputContractRenderModeValue:
		return nil
	case OutputContractRenderModeTemplate:
		if !strings.Contains(render.Template, "{choice}") &&
			!strings.Contains(render.Template, "{{choice}}") {
			return fmt.Errorf("%s: render.template must contain {choice} or {{choice}} when render.mode=template", context)
		}
		return nil
	default:
		return fmt.Errorf("%s: unsupported render.mode %q", context, render.Mode)
	}
}

func validateOutputContractPostprocess(spec *OutputContractSpec, context string) error {
	for i, item := range spec.Postprocess {
		switch strings.TrimSpace(item.Type) {
		case OutputContractPostprocessDereferenceSelectedReference:
			if strings.TrimSpace(spec.Type) != OutputContractTypeReferenceSelect {
				return fmt.Errorf("%s: postprocess[%d] requires type=%s", context, i, OutputContractTypeReferenceSelect)
			}
			continue
		default:
			return fmt.Errorf("%s: postprocess[%d] has unsupported type %q", context, i, item.Type)
		}
	}
	return nil
}
