package evaluationplane

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// retainMethodLedgerPayload admits only one of the four typed ledger
// contracts and bounds every reducer collection before the response can be
// retained in memory. Unknown fields never survive as an untyped payload.
func retainMethodLedgerPayload(
	operation string,
	raw []byte,
	payload map[string]any,
) (map[string]any, error) {
	if payload == nil {
		return nil, fmt.Errorf("method ledger response is empty")
	}
	switch operation {
	case workerBrokerAgentTaskLedger:
		var ledger agentTaskLedgerPayload
		if err := decodeExactJSON(raw, &ledger); err != nil || len(ledger.Attempts) > maxRecordsPerRun {
			return nil, fmt.Errorf("agent-task ledger response violates its bounded contract")
		}
	case workerBrokerFaultRecoveryLedger:
		var ledger faultRecoveryLedgerPayload
		if err := decodeExactJSON(raw, &ledger); err != nil || len(ledger.Pairs) > maxRecordsPerRun {
			return nil, fmt.Errorf("fault-recovery ledger response violates its bounded contract")
		}
	case workerBrokerHardPolicyLedger:
		var ledger hardPolicyLedgerPayload
		if err := decodeExactJSON(raw, &ledger); err != nil || len(ledger.Observations) > maxRecordsPerRun {
			return nil, fmt.Errorf("hard-policy ledger response violates its bounded contract")
		}
	case workerBrokerProductionExperimentLedger:
		var ledger productionExperimentLedgerPayload
		if err := decodeExactJSON(raw, &ledger); err != nil || len(ledger.Assignments) > maxRecordsPerRun ||
			len(ledger.PreferenceOutcomes) > maxRecordsPerRun {
			return nil, fmt.Errorf("production experiment ledger response violates its bounded contract")
		}
	default:
		return nil, fmt.Errorf("response operation is not a method ledger")
	}
	return payload, nil
}

func (broker *workerHTTPBroker) attestResponse(
	request workerBrokerRequest,
	requestPayload []byte,
	responsePayload []byte,
	response workerBrokerResponse,
	upstreamAttempted bool,
	latencyMicroseconds int64,
	pairing *controlledPairObservation,
) executionAttestationEntry {
	requestDigest, requestDigestErr := brokerRequestSemanticDigest(request.Operation, requestPayload)
	if requestDigestErr != nil {
		// Invalid requests are still journaled fail-closed. They cannot pass the
		// case-bound record validator, but retaining a bounded digest preserves
		// the broker transcript for diagnosis.
		requestDigest = digestBytes(requestPayload)
	}
	entry := executionAttestationEntry{
		RequestID: request.ID, Operation: request.Operation, TrackID: request.TrackID,
		CaseID: request.CaseID, AttemptID: request.AttemptID,
		RequestDigest: requestDigest, ResponseDigest: digestBytes(responsePayload),
		UpstreamAttempted: upstreamAttempted, Success: response.Success,
		StatusCode: copyInt(response.StatusCode), LatencyMicroseconds: latencyMicroseconds,
		Headers: copyStringMap(response.Headers), responsePayload: response.retainedMethodLedgerPayload,
		ControlledPair: pairing, FetchedAt: copyTime(&response.FetchedAt),
	}
	entry.RequestedModel = brokerRequestedModel(request.Operation, requestPayload)
	entry.LedgerSealedAt = brokerLedgerSealedAt(request.Operation, response.Payload)
	populateBrokerObservedFields(&entry, response.Payload)
	// Router diagnostics expose both the configured decision algorithm and the
	// method that actually selected this request. Execution evidence is bound to
	// the realized method; the configured algorithm remains in routing traces.
	if request.Operation == workerBrokerRouterEvaluate {
		entry.Algorithm = copyString(entry.SelectionMethod)
	}
	if entry.SelectedModel == nil && (request.Operation == workerBrokerRoutedChatCompletion ||
		request.Operation == workerBrokerArmChatCompletion) {
		entry.SelectedModel = nonEmptyStringPointer(response.Headers["x-vsr-selected-model"])
	}
	if response.Success && request.Operation == workerBrokerRoutedChatCompletion {
		method := nonEmptyStringPointer(response.Headers["x-vsr-selected-algorithm"])
		if entry.Algorithm == nil {
			entry.Algorithm = method
		}
		if entry.SelectionMethod == nil {
			entry.SelectionMethod = method
		}
		if entry.SelectionStatus == nil {
			entry.SelectionStatus = nonEmptyStringPointer("selected")
		}
	} else if request.Operation == workerBrokerRoutedChatCompletion {
		entry.SelectionStatus = nil
		entry.SelectionMethod = nil
		entry.Algorithm = nil
	}
	if entry.Recipe == nil && request.Operation == workerBrokerRoutedChatCompletion {
		entry.Recipe = nonEmptyStringPointer(response.Headers["x-vsr-selected-recipe"])
	}
	if entry.DecisionName == nil && request.Operation == workerBrokerRoutedChatCompletion {
		entry.DecisionName = nonEmptyStringPointer(response.Headers["x-vsr-selected-decision"])
	}
	if entry.Recipe == nil && (request.Operation == workerBrokerRoutedChatCompletion ||
		request.Operation == workerBrokerRouterEvaluate) && broker.manifest.Target.Mixture != nil {
		recipe := broker.manifest.Target.Mixture.RecipeName
		entry.Recipe = &recipe
	}
	entry.ArmID = broker.resolveAttestedArmID(entry)
	entry.RoutingRecipeDecision = routingRecipeDecisionFromBrokerResponse(
		broker.manifest, request, response, entry,
	)
	if content := brokerResponseContent(response.Payload); content != nil {
		digest := digestString(normalizedAnswer(*content))
		entry.ResponseContentDigest = &digest
	}
	receipt, err := brokerEntryReceipt(entry)
	if err == nil {
		entry.BrokerReceipt = receipt
	}
	broker.entriesMu.Lock()
	broker.entries[request.ID] = entry
	broker.entriesMu.Unlock()
	return entry
}

func brokerLedgerSealedAt(operation string, payload map[string]any) *time.Time {
	if !isMethodLedgerOperation(operation) || payload == nil {
		return nil
	}
	raw, ok := payload["sealed_at"].(string)
	if !ok {
		return nil
	}
	sealedAt, err := time.Parse(time.RFC3339Nano, raw)
	if err != nil {
		return nil
	}
	sealedAt = sealedAt.UTC()
	return &sealedAt
}

func brokerRequestedModel(operation string, payload []byte) *string {
	switch operation {
	case workerBrokerRoutedChatCompletion, workerBrokerArmChatCompletion, workerBrokerRouterEvaluate:
	default:
		return nil
	}
	var envelope struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil || envelope.Model == "" {
		return nil
	}
	return &envelope.Model
}

func (broker *workerHTTPBroker) resolveAttestedArmID(entry executionAttestationEntry) *string {
	if broker.manifest.Target.Mixture == nil {
		return nil
	}
	candidate := entry.SelectedModel
	if entry.Operation == workerBrokerArmChatCompletion {
		candidate = entry.RequestedModel
	}
	if candidate == nil {
		return nil
	}
	for _, arm := range broker.manifest.Target.Mixture.ModelArms {
		if *candidate == arm.ID || *candidate == arm.Model {
			armID := arm.ID
			return &armID
		}
	}
	return nil
}

func nonEmptyStringPointer(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func copyString(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func populateBrokerObservedFields(entry *executionAttestationEntry, payload map[string]any) {
	if payload == nil {
		return
	}
	entry.SelectedModel = mapStringPointer(payload, "selected_model")
	entry.SelectionStatus = mapStringPointer(payload, "selection_status")
	entry.SelectionMethod = mapStringPointer(payload, "selection_method")
	entry.Recipe = mapStringPointer(payload, "recipe")
	if decision, ok := payload["decision_result"].(map[string]any); ok {
		entry.DecisionName = mapStringPointer(decision, "decision_name")
		entry.Algorithm = mapStringPointer(decision, "algorithm")
	}
	if usage, ok := payload["usage"].(map[string]any); ok {
		entry.InputTokens = mapNonNegativeIntegerPointer(usage, "prompt_tokens")
		entry.OutputTokens = mapNonNegativeIntegerPointer(usage, "completion_tokens")
	}
}

func mapStringPointer(value map[string]any, key string) *string {
	text, ok := value[key].(string)
	if !ok || text == "" || text != strings.TrimSpace(text) ||
		len(text) > maxBrokerObservedFieldBytes || strings.ContainsAny(text, "\x00\r\n") {
		return nil
	}
	return &text
}

func mapNonNegativeIntegerPointer(value map[string]any, key string) *int64 {
	raw, ok := value[key]
	if !ok {
		return nil
	}
	var parsed int64
	switch number := raw.(type) {
	case json.Number:
		converted, err := number.Int64()
		if err != nil {
			return nil
		}
		parsed = converted
	case int64:
		parsed = number
	case int:
		parsed = int64(number)
	default:
		return nil
	}
	if parsed < 0 {
		return nil
	}
	return &parsed
}

func copyInt(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func copyStringMap(value map[string]string) map[string]string {
	copy := make(map[string]string, len(value))
	for key, item := range value {
		copy[key] = item
	}
	return copy
}

func (broker *workerHTTPBroker) transcript(completedAt time.Time) brokerExecutionTranscript {
	broker.entriesMu.Lock()
	entries := orderedExecutionEntries(broker.entries)
	broker.entriesMu.Unlock()
	return brokerExecutionTranscript{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: broker.manifest.RunID, ManifestDigest: broker.manifest.ManifestDigest,
		TargetID: broker.manifest.Target.ID, Mode: broker.manifest.Mode,
		PolicySnapshotDigest:  broker.manifest.PolicySnapshotDigest,
		BackendTopologyDigest: broker.manifest.Target.BackendTopologyDigest,
		StartedAt:             broker.startedAt, CompletedAt: completedAt.UTC(), Entries: entries,
	}
}
