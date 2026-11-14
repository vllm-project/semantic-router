package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// validateDecisionOutputContracts checks every decision's output contract at
// load: its kinds and action, that this release executes them, and that each
// model it references declares what the contract needs and answers on an API
// that carries it. A decision without a contract answers in text, so its
// models must answer in text.
func validateDecisionOutputContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for _, decision := range cfg.Decisions {
		if err := validateDecisionOutput(cfg, decision); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionOutput(cfg *RouterConfig, decision Decision) error {
	output := decision.Output
	path := cfg.ScopePath(fmt.Sprintf("decisions[%s].output", decision.Name))
	if output != nil {
		if err := output.Validate(); err != nil {
			return fmt.Errorf("decisions[%s].output: %w", decision.Name, err)
		}
		if codec := output.missingCodec(); codec != "" {
			return CheckError(path, CodeOutputNotExecutable, fmt.Sprintf("kind %s with action %s needs %s, which this release does not have", output, output.EffectiveAction(), codec))
		}
		if algorithm := decision.Algorithm; algorithm != nil && IsExecutorAlgorithmType(algorithm.Type) && !slices.Equal(output.Kinds(), []string{OutputKindText}) {
			return CheckError(path, CodeOutputNotExecutable, fmt.Sprintf("algorithm %s is an executor, whose model calls answer in text only; select one model for kind %s", algorithm.Type, output))
		}
	}
	for _, ref := range decision.ModelRefs {
		params, ok := cfg.ModelConfig[ref.Model]
		if !ok {
			continue
		}
		if missing := missingOutputFacts(output, params); len(missing) > 0 {
			return CheckError(path, CodeOutputCapabilityMissing, fmt.Sprintf("model %q does not declare %s, which kind %s needs", ref.Model, strings.Join(missing, " or "), outputContractName(output)))
		}
		if cfg.RoutingFragmentOnly {
			continue
		}
		if reason := outputWireMismatch(output, params.APIFormat); reason != "" {
			return CheckError(path, CodeOutputNotExecutable, fmt.Sprintf("model %q %s", ref.Model, reason))
		}
	}
	return nil
}

// outputContractName names a decision's contract, the text default included.
func outputContractName(output *DecisionOutput) string {
	if output == nil {
		return OutputKindText + " (the default)"
	}
	return output.String()
}

// missingOutputFacts lists the facts of a model a contract needs and its
// providers.models[] entry does not state.
func missingOutputFacts(output *DecisionOutput, params ModelParams) []string {
	var missing []string
	if output.Includes(OutputKindText) && !modelOutputsText(params) {
		missing = append(missing, "text in modalities.output")
	}
	declared, _ := llmprotocol.ModelCapabilities(params.Capabilities)
	for _, name := range output.requiredCapabilities() {
		capability, err := llmprotocol.ParseCapabilities([]string{name})
		if err != nil || !declared.Contains(capability) {
			missing = append(missing, name+" in capabilities")
		}
	}
	return missing
}

// modelOutputsText reports whether a model answers in text: every model does
// unless its modalities name an output without text.
func modelOutputsText(params ModelParams) bool {
	return len(params.Modalities.Output) == 0 || slices.Contains(params.Modalities.Output, OutputKindText)
}

// outputWireMismatch explains why the API a model answers on cannot carry a
// contract's output, or "" when it can. A generated video needs the Videos
// API; the images API carries a generated image and nothing else.
func outputWireMismatch(output *DecisionOutput, apiFormat string) string {
	if !output.GeneratesImages() && !output.GeneratesVideos() {
		return ""
	}
	if output.GeneratesVideos() {
		if CanonicalAPIFormat(strings.ToLower(strings.TrimSpace(apiFormat))) == APIFormatVideos {
			return ""
		}
		return "answers on an API that cannot carry a generated video; reach it with api_format: videos"
	}
	switch format := CanonicalAPIFormat(strings.ToLower(strings.TrimSpace(apiFormat))); {
	case format == APIFormatResponses:
		return ""
	case format == APIFormatImages && output.ImageOnly():
		return ""
	case format == APIFormatImages:
		return "answers on the images API, which carries no text; reach it with api_format: responses"
	default:
		if format == "" {
			format = APIFormatChatCompletions
		}
		return fmt.Sprintf("answers on the %s API, which cannot carry a generated image; reach it with api_format: images or responses", format)
	}
}
