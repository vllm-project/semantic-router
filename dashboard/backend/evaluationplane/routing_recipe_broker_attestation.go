package evaluationplane

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

const (
	routingRecipeUnavailableReason = "not_observed"
	routingRecipeResponseError     = "invalid_router_response"
	routingRecipeRequestError      = "router_request_failed"
	routingRecipeValueError        = "invalid_signal_value"
	routingRecipeNotRecommended    = "not_recommended"
)

var routingRecipeSignalCollectionKeys = map[string]string{
	"keyword":       "keywords",
	"embedding":     "embeddings",
	"domain":        "domains",
	"fact_check":    "fact_check",
	"user_feedback": "user_feedback",
	"reask":         "reask",
	"preference":    "preferences",
	"language":      "language",
	"context":       "context",
	"structure":     "structure",
	"complexity":    "complexity",
	"modality":      "modality",
	"authz":         "authz",
	"jailbreak":     "jailbreak",
	"pii":           "pii",
	"kb":            "kb",
	"conversation":  "conversation",
	"event":         "event",
	"metadata":      "metadata",
	"classifier":    "classifier",
	"projection":    "projection",
}

// routingRecipeDecisionFromBrokerResponse creates the only decision evidence
// admitted to the execution attestation. The worker never supplies any field
// in this snapshot: identity and time come from the broker request, the plan
// comes from the immutable manifest, and observations come from the bounded
// Router response already fetched by the broker.
func routingRecipeDecisionFromBrokerResponse(
	manifest RunManifest,
	request workerBrokerRequest,
	response workerBrokerResponse,
	entry executionAttestationEntry,
) *RoutingRecipeDecisionSnapshot {
	if manifest.Mode != ModeLive || request.Operation != workerBrokerRouterEvaluate ||
		request.TrackID != "routing" || !containsTrack(manifest.TrackIDs, "routing") ||
		manifest.Target.Mixture == nil {
		return nil
	}
	plan := manifest.Target.Mixture.RoutingRecipePlan
	if ValidateRoutingRecipePlan(plan) != nil {
		return nil
	}
	snapshot := &RoutingRecipeDecisionSnapshot{
		ContractVersion: RoutingDecisionEvidenceContractVersion,
		DecisionID:      routingRecipeBrokerDecisionID(request.ID),
		PlanDigest:      plan.PlanDigest,
		CaseID:          request.CaseID,
		ObservedAt:      response.FetchedAt.UTC(),
		Signals:         make([]RoutingRecipeObservedInput, 0, len(plan.Signals)),
		Projections:     make([]RoutingRecipeObservedInput, 0, len(plan.Projections)),
		Eligibility:     unavailableRoutingRecipeEligibility(plan),
		RankedArmIDs:    []string{},
	}

	if !response.Success {
		state := "missing"
		errorCode := ""
		if entry.UpstreamAttempted {
			state = "error"
			errorCode = routingRecipeRequestError
		}
		if response.Error != nil && strings.Contains(strings.ToLower(*response.Error), "timeout") {
			state, errorCode = "timeout", ""
		}
		snapshot.Signals = unavailableRoutingRecipeInputs(plan.Signals, state, errorCode)
		snapshot.Projections = unavailableRoutingRecipeProjectionInputs(plan.Projections, state, errorCode)
		if entry.UpstreamAttempted {
			snapshot.SelectionStatus = "error"
		} else {
			snapshot.SelectionStatus = "unavailable"
		}
		return snapshot
	}

	for _, spec := range plan.Signals {
		snapshot.Signals = append(snapshot.Signals, normalizeRoutingRecipeObservedInput(response.Payload, spec, false))
	}
	for _, spec := range plan.Projections {
		snapshot.Projections = append(snapshot.Projections, normalizeRoutingRecipeObservedInput(
			response.Payload,
			RoutingRecipeInputSpec{ID: spec.ID, ValueKind: spec.ValueKind},
			true,
		))
	}

	recommendedArmIDs, recommendationsPresent, recommendationsValid := routingRecipeRecommendedArms(
		response.Payload, manifest.Target.Mixture.ModelArms,
	)
	status, selectedArmID := normalizedRoutingRecipeSelection(entry, plan)
	if !recommendationsValid {
		status, selectedArmID = "error", ""
		for index := range snapshot.Eligibility {
			snapshot.Eligibility[index] = RoutingRecipeEligibility{
				ArmID: snapshot.Eligibility[index].ArmID, State: "error", ReasonCode: routingRecipeResponseError,
			}
		}
	} else if recommendationsPresent {
		eligible := make(map[string]struct{}, len(recommendedArmIDs))
		for _, armID := range recommendedArmIDs {
			eligible[armID] = struct{}{}
		}
		for index := range snapshot.Eligibility {
			if _, present := eligible[snapshot.Eligibility[index].ArmID]; present {
				snapshot.Eligibility[index].State = "eligible"
				snapshot.Eligibility[index].ReasonCode = "none"
			} else {
				snapshot.Eligibility[index].State = "ineligible"
				snapshot.Eligibility[index].ReasonCode = routingRecipeNotRecommended
			}
		}
		snapshot.RankedArmIDs = append(snapshot.RankedArmIDs, recommendedArmIDs...)
	}
	snapshot.SelectionStatus = status
	if selectedArmID != "" {
		snapshot.SelectedArmID = selectedArmID
		// A final selection proves only the first selected arm when the Router
		// omitted recommendation evidence. It never changes eligibility.
		if !recommendationsPresent {
			snapshot.RankedArmIDs = []string{selectedArmID}
		}
	}
	return snapshot
}

func routingRecipeBrokerDecisionID(requestID uint64) string {
	return "routing-decision-" + strconv.FormatUint(requestID, 10)
}

func unavailableRoutingRecipeEligibility(plan RoutingRecipePlan) []RoutingRecipeEligibility {
	items := make([]RoutingRecipeEligibility, 0, len(plan.ArmIDs))
	for _, armID := range plan.ArmIDs {
		items = append(items, RoutingRecipeEligibility{
			ArmID: armID, State: "unavailable", ReasonCode: routingRecipeUnavailableReason,
		})
	}
	return items
}

func unavailableRoutingRecipeInputs(specs []RoutingRecipeInputSpec, state, errorCode string) []RoutingRecipeObservedInput {
	items := make([]RoutingRecipeObservedInput, 0, len(specs))
	for _, spec := range specs {
		items = append(items, RoutingRecipeObservedInput{ID: spec.ID, State: state, ErrorCode: errorCode})
	}
	return items
}

func unavailableRoutingRecipeProjectionInputs(specs []RoutingRecipeProjectionSpec, state, errorCode string) []RoutingRecipeObservedInput {
	items := make([]RoutingRecipeObservedInput, 0, len(specs))
	for _, spec := range specs {
		items = append(items, RoutingRecipeObservedInput{ID: spec.ID, State: state, ErrorCode: errorCode})
	}
	return items
}

func normalizeRoutingRecipeObservedInput(
	payload map[string]any,
	spec RoutingRecipeInputSpec,
	projection bool,
) RoutingRecipeObservedInput {
	result := RoutingRecipeObservedInput{ID: spec.ID, State: "missing"}
	if payload == nil {
		return result
	}
	if code, present := routingRecipeExactMapValue(payload, "signal_errors", spec.ID); present {
		text, valid := code.(string)
		if valid && strings.Contains(strings.ToLower(text), "timeout") {
			result.State = "timeout"
			return result
		}
		result.State = "error"
		result.ErrorCode = routingRecipeResponseError
		if valid {
			trimmed := strings.TrimSpace(text)
			if validRoutingRecipeID(trimmed) {
				result.ErrorCode = trimmed
			}
		}
		return result
	}

	value, hasValue, valueValid := routingRecipeExactNumber(payload, "signal_values", spec.ID)
	confidence, hasConfidence, confidenceValid := routingRecipeExactNumber(payload, "signal_confidences", spec.ID)
	if (hasValue && !valueValid) || (hasConfidence && !confidenceValid) {
		result.State, result.ErrorCode = "error", routingRecipeValueError
		return result
	}
	observed, present := value, hasValue
	if projection || !hasValue {
		observed, present = confidence, hasConfidence
	}
	if present {
		if spec.ValueKind == "probability" && (observed < 0 || observed > 1) {
			result.State, result.ErrorCode = "error", routingRecipeValueError
			return result
		}
		result.State, result.Value = "present", &observed
		return result
	}

	matched := routingRecipeSignalSetContains(payload, "matched_signals", spec.ID, projection)
	unmatched := routingRecipeSignalSetContains(payload, "unmatched_signals", spec.ID, projection)
	if matched && unmatched {
		result.State, result.ErrorCode = "error", routingRecipeResponseError
		return result
	}
	if spec.ValueKind == "none" && (matched || unmatched) {
		result.State = "present"
		return result
	}
	if unmatched {
		zero := float64(0)
		result.State, result.Value = "present", &zero
	}
	return result
}

func routingRecipeExactMapValue(payload map[string]any, collection, id string) (any, bool) {
	values, ok := payload[collection].(map[string]any)
	if !ok {
		return nil, false
	}
	value, present := values[id]
	return value, present
}

func routingRecipeExactNumber(payload map[string]any, collection, id string) (float64, bool, bool) {
	value, present := routingRecipeExactMapValue(payload, collection, id)
	if !present {
		return 0, false, true
	}
	number, valid := routingRecipeNumber(value)
	return number, true, valid
}

func routingRecipeNumber(value any) (float64, bool) {
	var number float64
	switch typed := value.(type) {
	case json.Number:
		parsed, err := typed.Float64()
		if err != nil {
			return 0, false
		}
		number = parsed
	case float64:
		number = typed
	case float32:
		number = float64(typed)
	case int:
		number = float64(typed)
	case int64:
		number = float64(typed)
	case int32:
		number = float64(typed)
	default:
		return 0, false
	}
	return number, !math.IsNaN(number) && !math.IsInf(number, 0)
}

func routingRecipeSignalSetContains(payload map[string]any, collection, id string, projection bool) bool {
	decision, ok := payload["decision_result"].(map[string]any)
	if !ok {
		return false
	}
	sets, ok := decision[collection].(map[string]any)
	if !ok {
		return false
	}
	parts := strings.Split(id, ":")
	if len(parts) < 2 {
		return false
	}
	signalType := parts[0]
	if projection {
		signalType = "projection"
	}
	collectionKey, supported := routingRecipeSignalCollectionKeys[signalType]
	if !supported {
		return false
	}
	want := parts[1]
	if signalType == "classifier" && len(parts) == 3 {
		want += ":" + parts[2]
	} else if len(parts) != 2 {
		return false
	}
	values, ok := sets[collectionKey].([]any)
	if !ok {
		if stringsValues, stringsOK := sets[collectionKey].([]string); stringsOK {
			for _, value := range stringsValues {
				if value == want {
					return true
				}
			}
		}
		return false
	}
	for _, value := range values {
		if text, ok := value.(string); ok && text == want {
			return true
		}
	}
	return false
}

func routingRecipeRecommendedArms(payload map[string]any, arms []ModelArm) ([]string, bool, bool) {
	if payload == nil {
		return []string{}, false, true
	}
	raw, present := payload["recommended_models"]
	if !present {
		return []string{}, false, true
	}
	values, ok := raw.([]any)
	if !ok {
		if stringsValues, stringsOK := raw.([]string); stringsOK {
			if len(stringsValues) > len(arms) {
				return nil, true, false
			}
			values = make([]any, len(stringsValues))
			for index := range stringsValues {
				values[index] = stringsValues[index]
			}
		} else {
			return nil, true, false
		}
	}
	if len(values) > len(arms) {
		return nil, true, false
	}
	result := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		identity, ok := value.(string)
		if !ok {
			return nil, true, false
		}
		armID, resolved := frozenArmID(arms, identity)
		if !resolved {
			return nil, true, false
		}
		if _, duplicate := seen[armID]; duplicate {
			return nil, true, false
		}
		seen[armID] = struct{}{}
		result = append(result, armID)
	}
	return result, true, true
}

func normalizedRoutingRecipeSelection(entry executionAttestationEntry, plan RoutingRecipePlan) (string, string) {
	if !entry.Success {
		if entry.UpstreamAttempted {
			return "error", ""
		}
		return "unavailable", ""
	}
	raw := ""
	if entry.SelectionStatus != nil {
		raw = *entry.SelectionStatus
	}
	selected := ""
	if entry.ArmID != nil {
		selected = *entry.ArmID
	}
	if entry.SelectedModel != nil && selected == "" {
		return "error", ""
	}
	switch raw {
	case "selected", "planned_final":
		if selected == "" {
			return "error", ""
		}
		return "selected", selected
	case "fallback":
		if selected == "" || plan.FallbackArmID == "" || selected != plan.FallbackArmID {
			return "error", ""
		}
		return "fallback", selected
	case "execution_required":
		if selected != "" {
			return "error", ""
		}
		return "abstained", ""
	case "unavailable":
		if selected != "" {
			return "error", ""
		}
		return "unavailable", ""
	case "failed":
		if selected != "" {
			return "error", ""
		}
		return "error", ""
	default:
		return "error", ""
	}
}

func validateBrokerRoutingRecipeDecision(mixture *ManifestMixture, entry executionAttestationEntry) error {
	if entry.Operation != workerBrokerRouterEvaluate {
		if entry.RoutingRecipeDecision != nil {
			return fmt.Errorf("non-routing broker operation contains routing recipe decision evidence")
		}
		return nil
	}
	if mixture == nil || entry.RoutingRecipeDecision == nil || entry.FetchedAt == nil {
		return fmt.Errorf("routing broker operation omits its server-owned decision snapshot")
	}
	snapshot := *entry.RoutingRecipeDecision
	if err := ValidateRoutingRecipeDecisionSnapshot(mixture.RoutingRecipePlan, snapshot); err != nil {
		return fmt.Errorf("routing broker decision snapshot: %w", err)
	}
	if snapshot.DecisionID != routingRecipeBrokerDecisionID(entry.RequestID) ||
		snapshot.CaseID != entry.CaseID || !snapshot.ObservedAt.Equal(entry.FetchedAt.UTC()) {
		return fmt.Errorf("routing broker decision identity differs from its server request")
	}
	wantStatus, wantArmID := normalizedRoutingRecipeSelection(entry, mixture.RoutingRecipePlan)
	if snapshot.SelectionStatus != wantStatus || snapshot.SelectedArmID != wantArmID {
		return fmt.Errorf("routing broker decision selection differs from its Router response")
	}
	if wantArmID == "" {
		if entry.ArmID != nil || entry.SelectedModel != nil {
			return fmt.Errorf("non-final routing decision contains a selected model")
		}
		return nil
	}
	if entry.ArmID == nil || *entry.ArmID != wantArmID || entry.SelectedModel == nil {
		return fmt.Errorf("final routing decision does not bind its selected frozen arm")
	}
	resolved, ok := frozenArmID(mixture.ModelArms, *entry.SelectedModel)
	if !ok || resolved != wantArmID {
		return fmt.Errorf("final routing decision selected outside the frozen model pool")
	}
	return nil
}
