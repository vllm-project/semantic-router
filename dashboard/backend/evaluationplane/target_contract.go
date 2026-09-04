package evaluationplane

import (
	"fmt"
	"math"
	"regexp"
	"strings"
)

var (
	portableIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	secretEnvPattern  = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
	digestPattern     = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

func validateTargetContract(routerRef, envoyRef *SecretRef, arms []ModelArm, backendTopologyDigest string) error {
	if backendTopologyDigest != "" && !digestPattern.MatchString(backendTopologyDigest) {
		return fmt.Errorf("backend_topology_digest must be a sha256 digest")
	}
	for name, ref := range map[string]*SecretRef{"router_api_key": routerRef, "envoy_api_key": envoyRef} {
		if ref == nil {
			continue
		}
		if ref.SchemaVersion != SchemaVersion || !secretEnvPattern.MatchString(ref.Env) {
			return fmt.Errorf("%s must be a %s uppercase environment reference", name, SchemaVersion)
		}
	}
	if routerRef != nil && envoyRef != nil && routerRef.Env == envoyRef.Env {
		return fmt.Errorf("router and Envoy credential references must use different environment variables")
	}
	if envoyRef != nil && (envoyRef.Env == routerManagementCredentialEnv || envoyRef.Env == routerEvaluationCredentialEnv) {
		return fmt.Errorf("envoy credential reference cannot reuse a Router credential environment variable")
	}
	seenIDs := make(map[string]bool, len(arms))
	seenModels := make(map[string]bool, len(arms))
	for _, arm := range arms {
		if !portableIDPattern.MatchString(arm.ID) {
			return fmt.Errorf("model arm id %q is not portable", arm.ID)
		}
		model := strings.TrimSpace(arm.Model)
		if model == "" || len(model) > 512 {
			return fmt.Errorf("model arm %q has an invalid logical model", arm.ID)
		}
		if seenIDs[arm.ID] || seenModels[model] {
			return fmt.Errorf("model arm ids and logical models must be unique")
		}
		seenIDs[arm.ID], seenModels[model] = true, true
		if !digestPattern.MatchString(arm.ProviderModelIDDigest) {
			return fmt.Errorf("model arm %q provider identity digest is invalid", arm.ID)
		}
		if invalidCost(arm.InputCostPerMillionTokensUSD) || invalidCost(arm.OutputCostPerMillionTokensUSD) {
			return fmt.Errorf("model arm %q pricing must be finite and non-negative", arm.ID)
		}
		if err := validateArmMetadata(arm); err != nil {
			return fmt.Errorf("model arm %q: %w", arm.ID, err)
		}
	}
	return nil
}

func invalidCost(value float64) bool {
	return value < 0 || math.IsNaN(value) || math.IsInf(value, 0)
}

func validateArmMetadata(arm ModelArm) error {
	seen := make(map[string]bool, len(arm.Capabilities))
	for _, capability := range arm.Capabilities {
		if capability == "" || seen[capability] {
			return fmt.Errorf("capabilities must be non-empty and unique")
		}
		seen[capability] = true
	}
	seen = make(map[string]bool, len(arm.Modalities))
	for _, modality := range arm.Modalities {
		if seen[modality] || !allowedModality(modality) {
			return fmt.Errorf("modalities must be supported and unique")
		}
		seen[modality] = true
	}
	if arm.ContextWindowTokens != nil && *arm.ContextWindowTokens <= 0 {
		return fmt.Errorf("context_window_tokens must be positive")
	}
	for name, value := range map[string]*string{
		"parameter_size": arm.ParameterSize, "runtime_revision": arm.RuntimeRevision,
	} {
		if value != nil && (strings.TrimSpace(*value) == "" || (name == "parameter_size" && len(*value) > 64) || (name == "runtime_revision" && len(*value) > 160)) {
			return fmt.Errorf("%s is invalid", name)
		}
	}
	if arm.ConfigDigest != nil && !digestPattern.MatchString(*arm.ConfigDigest) {
		return fmt.Errorf("config_digest is invalid")
	}
	return nil
}

func allowedModality(value string) bool {
	switch value {
	case "text", "image", "document", "audio", "video":
		return true
	default:
		return false
	}
}

func hasMultimodalArm(arms []ModelArm) bool {
	for _, arm := range arms {
		for _, modality := range arm.Modalities {
			if modality != "text" {
				return true
			}
		}
	}
	return false
}

func copySecretRef(ref *SecretRef) *SecretRef {
	if ref == nil {
		return nil
	}
	copy := *ref
	return &copy
}

func copyModelArms(arms []ModelArm) []ModelArm {
	result := make([]ModelArm, len(arms))
	for i, arm := range arms {
		result[i] = arm
		result[i].Capabilities = append([]string(nil), arm.Capabilities...)
		result[i].Modalities = append([]string(nil), arm.Modalities...)
	}
	return result
}
