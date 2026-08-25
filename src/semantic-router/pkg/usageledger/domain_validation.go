package usageledger

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func validateDispatch(index int, dispatch Dispatch) (quota.QuotaInteger, quota.QuotaInteger, error) {
	zero := quota.QuotaInteger{}
	if err := validateDispatchIdentity(index, dispatch); err != nil {
		return zero, zero, err
	}
	input, output, err := validateDispatchUsage(dispatch)
	if err != nil {
		return zero, zero, err
	}
	if err := validateDispatchAttempts(dispatch); err != nil {
		return zero, zero, err
	}
	return input, output, nil
}

func validateDispatchIdentity(index int, dispatch Dispatch) error {
	if err := boundedIdentifier("dispatch ID", dispatch.DispatchID, 256); err != nil {
		return err
	}
	if dispatch.ParentDispatchID != "" {
		if err := boundedIdentifier("parent dispatch ID", dispatch.ParentDispatchID, 256); err != nil {
			return err
		}
	}
	if dispatch.Ordinal < 0 || dispatch.Ordinal != index {
		return invalid("dispatch ordinals must be contiguous and ordered")
	}
	if err := boundedCode("dispatch type", dispatch.DispatchType, true); err != nil {
		return err
	}
	if dispatch.DecisionTier < 0 {
		return invalid("decision tier cannot be negative")
	}
	if err := boundedCode("parallel group", dispatch.ParallelGroup, false); err != nil {
		return err
	}
	for label, value := range map[string]string{
		"decision ID": dispatch.DecisionID, "model ID": dispatch.ModelID,
	} {
		if err := boundedCode(label, value, false); err != nil {
			return err
		}
	}
	if err := requireUUID("backend ID", dispatch.BackendID, true); err != nil {
		return err
	}
	for label, value := range map[string]string{
		"decision display snapshot": dispatch.DecisionName,
		"model display snapshot":    dispatch.ModelName,
		"provider Model ID":         dispatch.ProviderModelID,
	} {
		if err := boundedSafeText(label, value, 512, true); err != nil {
			return err
		}
	}
	if dispatch.ModelID != "" && dispatch.ModelRevision <= 0 {
		return invalid("model revision is required with a model ID")
	}
	if dispatch.PricingRevision < 0 {
		return invalid("pricing revision cannot be negative")
	}
	for label, value := range map[string]string{
		"provider ID": dispatch.ProviderID, "retry class": dispatch.RetryClass, "cache state": dispatch.CacheState,
	} {
		if err := boundedCode(label, value, false); err != nil {
			return err
		}
	}
	return nil
}

func validateDispatchUsage(dispatch Dispatch) (quota.QuotaInteger, quota.QuotaInteger, error) {
	zero := quota.QuotaInteger{}
	input, err := parseQuantity("dispatch input tokens", dispatch.InputTokens)
	if err != nil {
		return zero, zero, err
	}
	cacheRead, err := parseQuantity("dispatch cache-read tokens", dispatch.CacheReadTokens)
	if err != nil {
		return zero, zero, err
	}
	cacheWrite, err := parseQuantity("dispatch cache-write tokens", dispatch.CacheWriteTokens)
	if err != nil {
		return zero, zero, err
	}
	output, err := parseQuantity("dispatch output tokens", dispatch.OutputTokens)
	if err != nil {
		return zero, zero, err
	}
	cacheTotal, err := cacheRead.Add(cacheWrite)
	if err != nil || cacheTotal.Compare(input) > 0 {
		return zero, zero, invalid("dispatch cache buckets exceed input tokens")
	}
	switch dispatch.UsageState {
	case UsageKnownZero:
		if !input.IsZero() || !output.IsZero() || !cacheRead.IsZero() || !cacheWrite.IsZero() || dispatch.UnknownReason != "" {
			return zero, zero, invalid("known-zero dispatch must contain zero usage and no unknown reason")
		}
	case UsageKnownActual:
		if dispatch.UnknownReason != "" {
			return zero, zero, invalid("known-actual dispatch cannot carry an unknown reason")
		}
	case UsageUnknown:
		if !input.IsZero() || !output.IsZero() || !cacheRead.IsZero() || !cacheWrite.IsZero() {
			return zero, zero, invalid("unknown dispatch cannot claim token values")
		}
		if err := boundedCode("unknown reason", dispatch.UnknownReason, true); err != nil {
			return zero, zero, err
		}
	default:
		return zero, zero, invalid("unsupported dispatch usage state %q", dispatch.UsageState)
	}
	if err := validateCost(dispatch.Cost); err != nil {
		return zero, zero, err
	}
	if !isHexDigest(dispatch.EvidenceDigest) {
		return zero, zero, invalid("dispatch evidence digest must be 32-byte lowercase hex")
	}
	return input, output, nil
}

func validateDispatchAttempts(dispatch Dispatch) error {
	if dispatch.StartedAt.IsZero() || dispatch.CompletedAt.IsZero() || dispatch.CompletedAt.Before(dispatch.StartedAt) {
		return invalid("dispatch timestamps are required and ordered")
	}
	if len(dispatch.Attempts) == 0 || len(dispatch.Attempts) > 6 {
		return invalid("dispatch must contain between 1 and 6 attempts")
	}
	seen := make(map[string]struct{}, len(dispatch.Attempts))
	for ordinal, attempt := range dispatch.Attempts {
		if err := validateAttempt(ordinal, attempt, dispatch.StartedAt, dispatch.CompletedAt); err != nil {
			return err
		}
		if _, exists := seen[attempt.AttemptID]; exists {
			return invalid("duplicate attempt ID %q", attempt.AttemptID)
		}
		seen[attempt.AttemptID] = struct{}{}
		if ordinal < len(dispatch.Attempts)-1 && attempt.State != UsageKnownZero {
			return invalid("only proven known-zero attempts may precede a retry")
		}
	}
	terminal := dispatch.Attempts[len(dispatch.Attempts)-1]
	if dispatch.UsageState == UsageKnownZero && terminal.State != UsageKnownZero ||
		dispatch.UsageState == UsageKnownActual && terminal.State != UsageKnownActual ||
		dispatch.UsageState == UsageUnknown && terminal.State != UsageUnknown {
		return invalid("terminal attempt state does not match dispatch usage state")
	}
	return nil
}

func validateAttempt(ordinal int, attempt Attempt, dispatchStart, dispatchEnd time.Time) error {
	if err := boundedIdentifier("attempt ID", attempt.AttemptID, 256); err != nil {
		return err
	}
	if attempt.Ordinal != ordinal {
		return invalid("attempt ordinals must be contiguous and ordered")
	}
	if err := requireUUID("attempt backend ID", attempt.BackendID, true); err != nil {
		return err
	}
	if err := boundedCode("attempt provider ID", attempt.ProviderID, false); err != nil {
		return err
	}
	if attempt.State != UsageKnownZero && attempt.State != UsageKnownActual && attempt.State != UsageUnknown {
		return invalid("unsupported attempt state %q", attempt.State)
	}
	if attempt.StatusCode != 0 && (attempt.StatusCode < 100 || attempt.StatusCode > 599) {
		return invalid("attempt status code is outside HTTP range")
	}
	if err := boundedCode("attempt error code", attempt.ErrorCode, false); err != nil {
		return err
	}
	if attempt.StartedAt.Before(dispatchStart) || attempt.CompletedAt.After(dispatchEnd) || attempt.CompletedAt.Before(attempt.StartedAt) {
		return invalid("attempt timestamps are outside dispatch lifetime")
	}
	return nil
}

func validateCost(cost DispatchCost) error {
	if !currencyPattern.MatchString(cost.Currency) {
		return invalid("dispatch cost requires an ISO-4217 currency")
	}
	amount, err := parseQuantity("dispatch cost numerator", cost.Numerator)
	if err != nil {
		return err
	}
	switch cost.State {
	case CostComplete:
		if cost.Reason != "" {
			return invalid("complete cost cannot carry an unknown reason")
		}
	case CostUnknown:
		if !amount.IsZero() {
			return invalid("unknown cost cannot claim a numerator")
		}
		if err := boundedCode("unknown cost reason", cost.Reason, true); err != nil {
			return err
		}
	default:
		return invalid("unsupported cost state %q", cost.State)
	}
	return nil
}

func validatePrincipal(principal PrincipalSnapshot) error {
	for label, value := range map[string]string{
		"API key ID": principal.APIKeyID, "credential ID": principal.CredentialID,
		"user ID": principal.UserID, "team ID": principal.TeamID,
	} {
		if err := requireUUID(label, value, true); err != nil {
			return err
		}
	}
	if principal.APIKeyID == "" {
		return invalid("API key ID is required")
	}
	for label, value := range map[string]string{
		"API key display snapshot": principal.APIKeyName, "user display snapshot": principal.UserName,
		"team display snapshot": principal.TeamName,
	} {
		if err := boundedSafeText(label, value, 256, true); err != nil {
			return err
		}
	}
	return nil
}

func validateRouting(routing RoutingSnapshot) error {
	for label, value := range map[string]string{
		"entrypoint ID": routing.EntrypointID, "entrypoint rule ID": routing.EntrypointRuleID,
		"recipe ID": routing.RecipeID,
	} {
		if err := boundedCode(label, value, false); err != nil {
			return err
		}
	}
	if routing.RoutingRevision < 0 || routing.AccessRevision < 0 || routing.RecipeRevision < 0 {
		return invalid("routing revisions cannot be negative")
	}
	for _, value := range []string{routing.EntrypointName, routing.EntrypointRuleName, routing.RecipeName} {
		if err := boundedSafeText("routing display snapshot", value, 256, true); err != nil {
			return err
		}
	}
	return nil
}

func validateMetadata(metadata map[string]string) error {
	allowed := map[string]struct{}{
		"client_name": {}, "client_version": {}, "correlation_id": {}, "content_type": {},
		"request_digest": {}, "sdk": {}, "sdk_version": {}, "user_agent_family": {},
	}
	for key, value := range metadata {
		if !metaPattern.MatchString(key) {
			return invalid("metadata key %q is not canonical", key)
		}
		if _, ok := allowed[key]; !ok {
			return invalid("metadata key %q is not in the safe allowlist", key)
		}
		if len(value) > 256 || strings.ContainsRune(value, '\x00') || looksSensitive(value) {
			return invalid("metadata value for %q is unsafe", key)
		}
	}
	return nil
}

func validateFence(fence UnknownFence) error {
	if err := requireUUID("fence ID", fence.FenceID, false); err != nil {
		return err
	}
	if err := boundedCode("fence reason", fence.Reason, true); err != nil {
		return err
	}
	if len(fence.Bindings) == 0 || len(fence.Bindings) > 256 {
		return invalid("fence must identify affected bindings")
	}
	seen := make(map[string]struct{}, len(fence.Bindings))
	for _, binding := range fence.Bindings {
		if err := requireUUID("fence binding ID", binding.BindingID, false); err != nil {
			return err
		}
		if err := requireUUID("fence rule ID", binding.RuleID, false); err != nil {
			return err
		}
		key := binding.BindingID + "/" + binding.RuleID
		if _, exists := seen[key]; exists {
			return invalid("duplicate fence binding %q", key)
		}
		seen[key] = struct{}{}
		for label, value := range map[string]string{"admission limit": binding.AdmissionLimit, "maximum debit": binding.MaximumDebit} {
			if value != "" {
				if _, err := parseQuantity(label, value); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func validateReceipts(receipts []QuotaReceipt) error {
	if len(receipts) > 512 {
		return invalid("too many quota receipts")
	}
	seen := make(map[string]struct{}, len(receipts))
	for _, receipt := range receipts {
		if err := requireUUID("quota binding ID", receipt.BindingID, false); err != nil {
			return err
		}
		if err := requireUUID("quota rule ID", receipt.RuleID, false); err != nil {
			return err
		}
		if err := boundedCode("quota metric", receipt.Metric, true); err != nil {
			return err
		}
		if _, err := parseQuantity("quota receipt amount", receipt.Amount); err != nil {
			return err
		}
		key := receipt.BindingID + "/" + receipt.RuleID
		if _, exists := seen[key]; exists {
			return invalid("duplicate quota receipt %q", key)
		}
		seen[key] = struct{}{}
	}
	return nil
}
