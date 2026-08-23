package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const pendingFinalizationDigest = "0000000000000000000000000000000000000000000000000000000000000000"

// buildTerminalUsageEvent projects the request-local accounting state into the
// only durable inference event schema. It deliberately has no access to the
// presented credential, headers, prompt, response body, or arbitrary metadata.
func (r *OpenAIRouter) buildTerminalUsageEvent(
	request *RequestContext,
	state *inferenceRequestAccess,
	requestDispatches []*inferenceDispatch,
	aggregate usageaccounting.Aggregate,
	statusCode int,
	errorCode string,
	fenceID string,
) (usageledger.TerminalEvent, error) {
	if request == nil || state == nil || state.admission == nil || state.admission.Tenant.NamespaceID == "" {
		return usageledger.TerminalEvent{}, fmt.Errorf("terminal usage event requires an admitted request")
	}
	admission := *state.admission
	tenant := admission.Tenant
	occurredAt := request.StartTime.UTC()
	if occurredAt.IsZero() {
		occurredAt = request.ProcessingStartTime.UTC()
	}
	if occurredAt.IsZero() {
		occurredAt = admission.PreparedAt.UTC()
	}
	completedAt := time.Now().UTC()
	if occurredAt.IsZero() || completedAt.Before(occurredAt) {
		occurredAt = completedAt
	}

	dispatches := make([]usageledger.Dispatch, 0, len(requestDispatches))
	known, unknown := 0, 0
	for index, dispatch := range requestDispatches {
		item, err := terminalUsageDispatch(request, dispatch, index, statusCode)
		if err != nil {
			return usageledger.TerminalEvent{}, err
		}
		dispatches = append(dispatches, item)
		if item.UsageState == usageledger.UsageUnknown {
			unknown++
		} else {
			known++
		}
	}
	evidence := usageledger.EvidenceKnown
	if unknown > 0 && known > 0 {
		evidence = usageledger.EvidenceMixed
	} else if unknown > 0 {
		evidence = usageledger.EvidenceUnknown
	}

	event := usageledger.TerminalEvent{
		Schema:              usageledger.TerminalEventSchema,
		EventID:             uuid.NewString(),
		NamespaceID:         tenant.NamespaceID,
		AdmissionID:         tenant.AdmissionID,
		FinalizationDigest:  pendingFinalizationDigest,
		EvidenceState:       evidence,
		ReplayID:            request.RouterReplayID,
		Protocol:            inferenceUsageProtocol(request),
		Path:                normalizedInferencePath(request),
		StatusCode:          canonicalTerminalStatus(statusCode),
		ErrorCode:           canonicalUsageReason(errorCode),
		OccurredAt:          occurredAt,
		CompletedAt:         completedAt,
		LatencyMilliseconds: completedAt.Sub(occurredAt).Milliseconds(),
		Stream:              request.ExpectStreamingResponse || request.IsStreamingResponse,
		ToolCall:            semanticResponseHasToolCall(request.SemanticResponse),
		Principal: usageledger.PrincipalSnapshot{
			APIKeyID: tenant.APIKeyID,
			UserID:   tenant.UserID,
			TeamID:   tenant.TeamID,
		},
		Routing:       terminalRoutingSnapshot(request, state),
		Served:        terminalServedUsage(aggregate),
		Dispatches:    dispatches,
		QuotaReceipts: terminalQuotaReceipts(admission.Rules, aggregate),
	}
	if event.ErrorCode == "request_terminated" && statusCode >= 200 && statusCode < 400 {
		event.ErrorCode = ""
	}
	if fenceID != "" {
		event.Fence = terminalUnknownFence(fenceID, admission.Rules, aggregate)
		if event.Fence == nil {
			return usageledger.TerminalEvent{}, fmt.Errorf("unknown-usage fence has no affected binding")
		}
	}

	// Hash a canonical event with a fixed digest sentinel. The resulting digest
	// is stable for the settlement plan and does not require a circular hash.
	if _, err := event.Validate(); err != nil {
		return usageledger.TerminalEvent{}, fmt.Errorf("validate terminal usage event: %w", err)
	}
	canonical, err := json.Marshal(event)
	if err != nil {
		return usageledger.TerminalEvent{}, fmt.Errorf("encode terminal usage digest input: %w", err)
	}
	digest := sha256.Sum256(canonical)
	event.FinalizationDigest = hex.EncodeToString(digest[:])
	if _, err := event.Validate(); err != nil {
		return usageledger.TerminalEvent{}, fmt.Errorf("validate finalized terminal usage event: %w", err)
	}
	return event, nil
}

func terminalUsageDispatch(
	request *RequestContext,
	dispatch *inferenceDispatch,
	index int,
	statusCode int,
) (usageledger.Dispatch, error) {
	if dispatch == nil {
		return usageledger.Dispatch{}, fmt.Errorf("dispatch %d is nil", index)
	}
	state := usageledger.UsageUnknown
	unknownReason := canonicalUsageReason(dispatch.reason)
	actual := usageaccounting.ActualUsage{}
	switch dispatch.state {
	case usageaccounting.EvidenceKnownZero:
		state = usageledger.UsageKnownZero
		unknownReason = ""
	case usageaccounting.EvidenceKnownActual:
		state = usageledger.UsageKnownActual
		unknownReason = ""
		actual = dispatch.usage
	case usageaccounting.EvidenceUnknown:
	default:
		return usageledger.Dispatch{}, fmt.Errorf("dispatch %q has unsupported evidence state %q", dispatch.id, dispatch.state)
	}
	cost, err := terminalDispatchCost(dispatch.pricing, actual, state, unknownReason)
	if err != nil {
		return usageledger.Dispatch{}, fmt.Errorf("dispatch %q cost: %w", dispatch.id, err)
	}
	completedAt := dispatch.completedAt.UTC()
	if completedAt.IsZero() {
		completedAt = time.Now().UTC()
	}
	startedAt := dispatch.startedAt.UTC()
	if startedAt.IsZero() || completedAt.Before(startedAt) {
		startedAt = completedAt
	}
	attempts := append([]usageledger.Attempt(nil), dispatch.attempts...)
	if len(attempts) == 0 {
		attemptStatus := 0
		if state == usageledger.UsageKnownActual {
			attemptStatus = canonicalTerminalStatus(statusCode)
		}
		errorCode := ""
		if state == usageledger.UsageUnknown {
			errorCode = unknownReason
		}
		attemptKind := "missing"
		if !dispatch.attemptEvidenceRequired {
			attemptKind = "local"
		}
		attempts = []usageledger.Attempt{{
			AttemptID: dispatch.id + "/attempt/" + attemptKind, Ordinal: 0,
			State: state, StatusCode: attemptStatus, ErrorCode: errorCode,
			StartedAt: startedAt, CompletedAt: completedAt,
		}}
	}
	dispatchType := dispatch.dispatchType
	if dispatchType == "" {
		dispatchType = inferenceDispatchType(index)
	}
	backendID, providerID := "", ""
	if len(attempts) > 0 {
		backendID = attempts[len(attempts)-1].BackendID
		providerID = attempts[len(attempts)-1].ProviderID
	}
	decisionID, decisionName := "", ""
	if request != nil && request.VSRSelectedDecision != nil {
		decisionID = canonicalOptionalUUID(request.VSRSelectedDecision.ID)
		decisionName = request.VSRSelectedDecision.Name
	}
	item := usageledger.Dispatch{
		DispatchID:       dispatch.id,
		Ordinal:          index,
		DispatchType:     dispatchType,
		DecisionID:       decisionID,
		DecisionName:     decisionName,
		DecisionTier:     decisionTier(request),
		ModelID:          canonicalModelIdentity(dispatch.modelID),
		ModelName:        dispatch.model,
		ModelRevision:    dispatch.modelRevision,
		PricingRevision:  dispatch.modelRevision,
		BackendID:        backendID,
		ProviderID:       providerID,
		InputTokens:      terminalQuantity(actual.InputTotal, state),
		CacheReadTokens:  terminalQuantity(actual.CacheRead, state),
		CacheWriteTokens: terminalQuantity(actual.CacheWrite, state),
		OutputTokens:     terminalQuantity(actual.Output, state),
		UsageState:       state,
		UnknownReason:    unknownReason,
		Cost:             cost,
		StartedAt:        startedAt,
		CompletedAt:      completedAt,
		Attempts:         attempts,
	}
	item.EvidenceDigest = terminalDispatchEvidenceDigest(item)
	return item, nil
}

func terminalDispatchCost(
	pricing usageaccounting.Pricing,
	actual usageaccounting.ActualUsage,
	state usageledger.UsageState,
	reason string,
) (usageledger.DispatchCost, error) {
	if state == usageledger.UsageUnknown {
		return usageledger.DispatchCost{
			Currency:  pricing.Currency,
			State:     usageledger.CostUnknown,
			Numerator: "0",
			Reason:    reason,
		}, nil
	}
	cost, err := usageaccounting.CalculateCost(pricing, actual)
	if err != nil {
		return usageledger.DispatchCost{}, err
	}
	if cost.Completeness == usageaccounting.CostUnknown {
		return usageledger.DispatchCost{
			Currency:  cost.Currency,
			State:     usageledger.CostUnknown,
			Numerator: "0",
			Reason:    canonicalUsageReason(cost.Reason),
		}, nil
	}
	return usageledger.DispatchCost{
		Currency:  cost.Currency,
		State:     usageledger.CostComplete,
		Numerator: cost.Numerator.String(),
	}, nil
}

func terminalDispatchEvidenceDigest(dispatch usageledger.Dispatch) string {
	dispatch.EvidenceDigest = ""
	payload, _ := json.Marshal(dispatch)
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

func terminalRoutingSnapshot(request *RequestContext, state *inferenceRequestAccess) usageledger.RoutingSnapshot {
	result := usageledger.RoutingSnapshot{AccessRevision: int64(state.admission.Tenant.PolicyRevision)}
	if state.entrypoint != nil {
		result.EntrypointID = canonicalOptionalUUID(state.entrypoint.ID)
		result.EntrypointName = state.entrypoint.Name
		result.RoutingRevision = state.entrypoint.Revision
	}
	if state.rule != nil {
		result.EntrypointRuleID = canonicalOptionalUUID(state.rule.ID)
		result.EntrypointRuleName = state.rule.Name
	}
	if request != nil {
		if recipe := request.Routing.SelectedRecipe(); recipe != nil {
			result.RecipeID = canonicalOptionalUUID(recipe.ID)
			result.RecipeName = string(recipe.Name)
			result.RecipeRevision = recipe.Revision
		}
	}
	return result
}

func terminalServedUsage(aggregate usageaccounting.Aggregate) usageledger.ServedUsage {
	input, output := "0", "0"
	if aggregate.ServedInput.Complete {
		input = aggregate.ServedInput.Value.String()
	}
	if aggregate.ServedOutput.Complete {
		output = aggregate.ServedOutput.Value.String()
	}
	return usageledger.ServedUsage{
		InputTokens:  input,
		InputKnown:   aggregate.ServedInput.Complete,
		OutputTokens: output,
		OutputKnown:  aggregate.ServedOutput.Complete,
	}
}

func terminalQuotaReceipts(bindings []quotaruntime.RuleBinding, aggregate usageaccounting.Aggregate) []usageledger.QuotaReceipt {
	receipts := make([]usageledger.QuotaReceipt, 0, len(bindings))
	for _, binding := range bindings {
		if binding.Rule.Algorithm == quota.AlgorithmConcurrency {
			continue
		}
		amount := "1"
		if binding.Rule.Accounting == quota.AccountingResponseActual {
			metric := aggregate.Metric(binding.Rule.Metric)
			if !metric.Complete {
				continue
			}
			amount = metric.Value.String()
		}
		receipts = append(receipts, usageledger.QuotaReceipt{
			BindingID: binding.BindingID,
			RuleID:    binding.Rule.ID,
			Metric:    string(binding.Rule.Metric),
			Amount:    amount,
		})
	}
	return receipts
}

func terminalUnknownFence(
	fenceID string,
	bindings []quotaruntime.RuleBinding,
	aggregate usageaccounting.Aggregate,
) *usageledger.UnknownFence {
	fence := &usageledger.UnknownFence{FenceID: fenceID, Reason: "authoritative_usage_missing"}
	for _, binding := range bindings {
		if binding.Rule.Accounting != quota.AccountingResponseActual {
			continue
		}
		metric := aggregate.Metric(binding.Rule.Metric)
		if metric.Complete {
			continue
		}
		if metric.Reason != "" {
			fence.Reason = canonicalUsageReason(metric.Reason)
		}
		fence.Bindings = append(fence.Bindings, usageledger.FenceBinding{
			BindingID:      binding.BindingID,
			RuleID:         binding.Rule.ID,
			AdmissionLimit: terminalRuleLimit(binding.Rule),
		})
	}
	if len(fence.Bindings) == 0 {
		return nil
	}
	return fence
}

func terminalRuleLimit(rule quota.RateLimitRule) string {
	if rule.Metric == quota.MetricCost && rule.CostLimit != nil {
		return rule.CostLimit.ScaledInteger().String()
	}
	if rule.WholeLimit != nil {
		return rule.WholeLimit.String()
	}
	return ""
}

func inferenceUsageProtocol(request *RequestContext) string {
	path := normalizedInferencePath(request)
	switch {
	case request != nil && request.SourceFormat != "":
		return string(request.SourceFormat)
	case path == "/v1/responses":
		return string(llmprotocol.OpenAIResponsesV1)
	default:
		return string(llmprotocol.OpenAIChatV1)
	}
}

func inferenceDispatchType(index int) string {
	if index == 0 {
		return "primary"
	}
	return "internal"
}

func terminalQuantity(value quota.QuotaInteger, state usageledger.UsageState) string {
	if state == usageledger.UsageUnknown {
		return "0"
	}
	return value.String()
}

func canonicalOptionalUUID(value string) string {
	parsed, err := uuid.Parse(strings.TrimSpace(value))
	if err != nil || parsed.String() != strings.ToLower(strings.TrimSpace(value)) {
		return ""
	}
	return parsed.String()
}

func canonicalModelIdentity(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > 256 {
		return ""
	}
	for index, character := range []byte(value) {
		if character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z' ||
			character >= '0' && character <= '9' || (index > 0 && strings.ContainsRune("._:/-", rune(character))) {
			continue
		}
		return ""
	}
	return value
}

func canonicalTerminalStatus(status int) int {
	if status >= 100 && status <= 599 {
		return status
	}
	return 500
}
