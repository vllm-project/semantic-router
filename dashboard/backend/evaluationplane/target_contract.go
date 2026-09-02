package evaluationplane

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	if routerRef != nil && routerRef.Env == routerManagementCredentialEnv {
		return fmt.Errorf("router evaluation credential cannot reuse the Dashboard management credential environment variable")
	}
	if envoyRef != nil && (envoyRef.Env == routerManagementCredentialEnv || envoyRef.Env == routerEvaluationCredentialEnv) {
		return fmt.Errorf("envoy credential reference cannot reuse a Router credential environment variable")
	}
	seenIDs := make(map[string]bool, len(arms))
	seenModels := make(map[string]bool, len(arms))
	selectorOwners := make(map[string]string, len(arms)*2)
	for _, arm := range arms {
		if !portableIDPattern.MatchString(arm.ID) {
			return fmt.Errorf("model arm id %q is not portable", arm.ID)
		}
		model := strings.TrimSpace(arm.Model)
		if model == "" || model != arm.Model || len(model) > 512 {
			return fmt.Errorf("model arm %q has an invalid logical model", arm.ID)
		}
		if seenIDs[arm.ID] || seenModels[model] {
			return fmt.Errorf("model arm ids and logical models must be unique")
		}
		seenIDs[arm.ID], seenModels[model] = true, true
		for _, selector := range []string{arm.ID, model} {
			if owner, present := selectorOwners[selector]; present && owner != arm.ID {
				return fmt.Errorf("model arm ids and logical models must be unambiguous")
			}
			selectorOwners[selector] = arm.ID
		}
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

func validateMixtureContract(mixture *ManifestMixture) error {
	if mixture == nil {
		return nil
	}
	if mixture.SchemaVersion != SchemaVersion ||
		mixture.ID != "mom-"+strings.TrimPrefix(digestString(mixture.RecipeName), "sha256:") ||
		!portableIDPattern.MatchString(mixture.ID) {
		return fmt.Errorf("mixture identity is invalid")
	}
	if mixture.EntrypointModel == "" || mixture.EntrypointModel != strings.TrimSpace(mixture.EntrypointModel) ||
		mixture.RecipeName == "" || mixture.RecipeName != strings.TrimSpace(mixture.RecipeName) ||
		mixture.RecipeDescription != strings.TrimSpace(mixture.RecipeDescription) {
		return fmt.Errorf("mixture entrypoint or recipe metadata is invalid")
	}
	if !digestPattern.MatchString(mixture.RecipeDigest) || !digestPattern.MatchString(mixture.PoolDigest) ||
		!digestPattern.MatchString(mixture.SelectorPolicyDigest) ||
		!digestPattern.MatchString(mixture.SelectorDigest) ||
		!digestPattern.MatchString(mixture.AdaptationDigest) ||
		!digestPattern.MatchString(mixture.BindingDigest) {
		return fmt.Errorf("mixture snapshot digest is invalid")
	}
	if len(mixture.Aliases) == 0 {
		return fmt.Errorf("mixture requires at least one entrypoint alias")
	}
	seenAliases := make(map[string]bool, len(mixture.Aliases))
	for _, alias := range mixture.Aliases {
		if alias == "" || alias != strings.TrimSpace(alias) || seenAliases[alias] {
			return fmt.Errorf("mixture aliases must be non-empty, trimmed, and unique")
		}
		seenAliases[alias] = true
	}
	if !seenAliases[mixture.EntrypointModel] {
		return fmt.Errorf("mixture entrypoint_model must be one of its aliases")
	}
	if err := validateTargetContract(nil, nil, mixture.ModelArms, ""); err != nil {
		return fmt.Errorf("mixture model pool: %w", err)
	}
	if mixture.PoolDigest != modelPoolSnapshotDigest(mixture.ModelArms) {
		return fmt.Errorf("mixture pool digest does not bind its model arms")
	}
	armIDs := make(map[string]bool, len(mixture.ModelArms))
	armModels := make(map[string]bool, len(mixture.ModelArms))
	previousModel := ""
	for _, arm := range mixture.ModelArms {
		if previousModel != "" && arm.Model <= previousModel {
			return fmt.Errorf("mixture model arms must be ordered by logical model")
		}
		previousModel = arm.Model
		armIDs[arm.ID] = true
		armModels[arm.Model] = true
	}
	if mixture.FallbackArmID != "" && !armIDs[mixture.FallbackArmID] {
		return fmt.Errorf("mixture fallback_arm_id must reference its model pool")
	}
	seenSupport := make(map[string]bool, len(mixture.SupportModels))
	previousSupport := ""
	for _, support := range mixture.SupportModels {
		model := support.Model
		if model == "" || model != strings.TrimSpace(model) || len(model) > 512 || seenSupport[model] || armModels[model] ||
			(previousSupport != "" && model <= previousSupport) ||
			!digestPattern.MatchString(support.ProviderModelIDDigest) ||
			!digestPattern.MatchString(support.ConfigDigest) ||
			!digestPattern.MatchString(support.BackendTopologyDigest) ||
			(support.RuntimeRevision != nil && (strings.TrimSpace(*support.RuntimeRevision) == "" || len(*support.RuntimeRevision) > 160)) {
			return fmt.Errorf("mixture support models must be sorted, unique, and outside the candidate pool")
		}
		seenSupport[model] = true
		previousSupport = model
	}
	if mixture.SelectorDigest != selectorSnapshotDigest(mixture.SelectorPolicyDigest, mixture.SupportModels) {
		return fmt.Errorf("mixture selector digest does not bind its support models")
	}
	seenDecisions := make(map[string]bool, len(mixture.Decisions))
	for _, decision := range mixture.Decisions {
		if decision.Name == "" || decision.Name != strings.TrimSpace(decision.Name) || seenDecisions[decision.Name] ||
			!validMixtureDecisionAlgorithm(decision.Algorithm) || len(decision.ArmIDs) == 0 {
			return fmt.Errorf("mixture decision identity is invalid")
		}
		seenDecisions[decision.Name] = true
		seenDecisionArms := make(map[string]bool, len(decision.ArmIDs))
		previousArmID := ""
		for _, armID := range decision.ArmIDs {
			if !armIDs[armID] || seenDecisionArms[armID] || (previousArmID != "" && armID <= previousArmID) {
				return fmt.Errorf("mixture decision %q has an invalid arm boundary", decision.Name)
			}
			seenDecisionArms[armID] = true
			previousArmID = armID
		}
	}
	if err := validateMixtureRoutingRecipePlan(mixture, armIDs); err != nil {
		return err
	}
	return nil
}

func validMixtureDecisionAlgorithm(algorithm string) bool {
	switch algorithm {
	case "default", "single", routerconfig.DecisionPluginFastResponse:
		return true
	default:
		return routerconfig.IsSupportedDecisionAlgorithmType(algorithm)
	}
}

func validateMixtureRoutingRecipePlan(mixture *ManifestMixture, armIDs map[string]bool) error {
	plan := mixture.RoutingRecipePlan
	if err := ValidateRoutingRecipePlan(plan); err != nil {
		return fmt.Errorf("mixture routing recipe plan is invalid: %w", err)
	}
	wantTargetDigest, err := routingRecipeTargetSnapshotDigest(*mixture)
	if err != nil || plan.TargetSnapshotDigest != wantTargetDigest {
		return fmt.Errorf("mixture routing recipe plan does not bind its immutable component digests")
	}
	wantArmIDs := make([]string, 0, len(armIDs))
	for armID := range armIDs {
		wantArmIDs = append(wantArmIDs, armID)
	}
	sort.Strings(wantArmIDs)
	gotArmIDs := append([]string(nil), plan.ArmIDs...)
	sort.Strings(gotArmIDs)
	if len(gotArmIDs) != len(wantArmIDs) {
		return fmt.Errorf("mixture routing recipe plan does not bind its frozen model pool")
	}
	for index := range wantArmIDs {
		if gotArmIDs[index] != wantArmIDs[index] {
			return fmt.Errorf("mixture routing recipe plan does not bind its frozen model pool")
		}
	}
	if plan.FallbackArmID != mixture.FallbackArmID {
		return fmt.Errorf("mixture routing recipe plan does not bind its frozen fallback arm")
	}
	wantTopK := routingRecipeTopK(len(wantArmIDs))
	if len(plan.TopK) != len(wantTopK) {
		return fmt.Errorf("mixture routing recipe plan does not use the frozen pool top-k schedule")
	}
	for index := range wantTopK {
		if plan.TopK[index] != wantTopK[index] {
			return fmt.Errorf("mixture routing recipe plan does not use the frozen pool top-k schedule")
		}
	}
	for _, signal := range plan.Signals {
		if signal.ValueKind != "numeric" {
			return fmt.Errorf("mixture routing recipe signals must be numeric")
		}
	}
	for _, projection := range plan.Projections {
		if projection.ValueKind != "probability" || projection.OutcomeBinding != "selected_is_oracle" {
			return fmt.Errorf("mixture routing recipe projections must bind oracle-selection probability")
		}
	}
	return nil
}

func validateEndpointCredentialBindings(routerURL, envoyURL string, routerRef, envoyRef *SecretRef) error {
	if routerRef != nil && routerURL == "" {
		return fmt.Errorf("router_api_key requires router_api_url")
	}
	if envoyRef != nil && envoyURL == "" {
		return fmt.Errorf("envoy_api_key requires envoy_url")
	}
	return nil
}

func validateServiceEndpoint(name string, endpoint *ServiceEndpoint) error {
	if endpoint == nil {
		return nil
	}
	if endpoint.SchemaVersion != SchemaVersion {
		return fmt.Errorf("%s must use schema version %s", name, SchemaVersion)
	}
	if err := validateServerOrigin(endpoint.URL); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if endpoint.URL == "" || endpoint.TimeoutSeconds <= 0 || endpoint.TimeoutSeconds > 600 ||
		math.IsNaN(endpoint.TimeoutSeconds) || math.IsInf(endpoint.TimeoutSeconds, 0) {
		return fmt.Errorf("%s must declare a bounded endpoint timeout", name)
	}
	if endpoint.APIKey != nil && (endpoint.APIKey.SchemaVersion != SchemaVersion || !secretEnvPattern.MatchString(endpoint.APIKey.Env)) {
		return fmt.Errorf("%s api_key must be a %s uppercase environment reference", name, SchemaVersion)
	}
	return nil
}

func validateDistinctTargetCredentials(refs map[string]*SecretRef) error {
	owners := make(map[string]string, len(refs))
	for name, ref := range refs {
		if ref == nil {
			continue
		}
		if owner, duplicate := owners[ref.Env]; duplicate {
			return fmt.Errorf("%s and %s credential references must use different environment variables", owner, name)
		}
		owners[ref.Env] = name
	}
	return nil
}

func invalidCost(value float64) bool {
	return value < 0 || math.IsNaN(value) || math.IsInf(value, 0)
}

func validateArmMetadata(arm ModelArm) error {
	seen := make(map[string]bool, len(arm.Capabilities))
	for _, capability := range arm.Capabilities {
		if capability == "" || capability != strings.TrimSpace(capability) || seen[capability] {
			return fmt.Errorf("capabilities must be non-empty, trimmed, and unique")
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

func copyServiceEndpoint(endpoint *ServiceEndpoint) *ServiceEndpoint {
	if endpoint == nil {
		return nil
	}
	copy := *endpoint
	copy.APIKey = copySecretRef(endpoint.APIKey)
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
