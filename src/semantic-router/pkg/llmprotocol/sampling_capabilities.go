package llmprotocol

import "strconv"

// unsupportedSamplingControls are the sampling controls a model may declare it
// lacks, each with the request field the Router then leaves out. A vLLM
// backend that runs speculative decoding, for example, rejects min_p and
// logit_bias.
var unsupportedSamplingControls = []struct {
	capability Capability
	field      string
}{
	{CapabilitySamplingMinP, "min_p"},
	{CapabilitySamplingLogitBias, "logit_bias"},
}

// ProjectUnsupportedSampling leaves out of the request each sampling control
// the selected model declares it lacks and reports a dropped diagnostic for
// it. The caller projects per candidate, so a model that supports a control
// still receives it.
func ProjectUnsupportedSampling(request Request, unsupported CapabilitySet, target WireFormat) (Request, Diagnostics) {
	projected := request
	var diagnostics Diagnostics
	for _, control := range unsupportedSamplingControls {
		if !unsupported.Supports(control.capability) {
			continue
		}
		if dropped, _ := BlockRequestField(&projected, control.field); dropped {
			diagnostics = append(diagnostics, Diagnostic{
				Source: request.Trusted.SourceFormat, Target: target, Field: control.field,
				Action: DiagnosticDropped, Reason: "the selected model does not support " + control.field,
			})
		}
	}
	if len(diagnostics) == 0 {
		return request, nil
	}
	projected.Generation++ // The source envelope still carries the dropped fields.
	return projected, diagnostics
}

func validateLogitBias(bias map[string]float64, limits Limits) error {
	if limits.LogitBiasEntries > 0 && len(bias) > limits.LogitBiasEntries {
		return NewError(ErrorInvalidRequest, "logit_bias_limit", "logit_bias entry limit exceeded", nil)
	}
	for token, value := range bias {
		if _, err := strconv.ParseUint(token, 10, 32); err != nil {
			return NewError(ErrorInvalidRequest, "invalid_logit_bias", "logit_bias keys must be token IDs", nil)
		}
		if !finiteFloat(value) || value < -100 || value > 100 {
			return NewError(ErrorInvalidRequest, "invalid_logit_bias", "logit_bias values must be between -100 and 100", nil)
		}
	}
	return nil
}
