package evaluationplane

import (
	"fmt"
	"sort"
	"time"
)

const executionAttestationContractVersion = "evaluation-execution-attestation.v1"

type executionAttestationEntry struct {
	RequestID             uint64                         `json:"request_id"`
	Operation             string                         `json:"operation"`
	TrackID               TrackID                        `json:"track_id,omitempty"`
	CaseID                string                         `json:"case_id,omitempty"`
	AttemptID             string                         `json:"attempt_id,omitempty"`
	RequestDigest         string                         `json:"request_digest"`
	ResponseDigest        string                         `json:"response_digest"`
	BrokerReceipt         string                         `json:"broker_receipt"`
	UpstreamAttempted     bool                           `json:"upstream_attempted"`
	Success               bool                           `json:"success"`
	StatusCode            *int                           `json:"status_code"`
	LatencyMicroseconds   int64                          `json:"latency_microseconds"`
	FetchedAt             *time.Time                     `json:"fetched_at,omitempty"`
	LedgerSealedAt        *time.Time                     `json:"ledger_sealed_at,omitempty"`
	Headers               map[string]string              `json:"headers"`
	RequestedModel        *string                        `json:"requested_model,omitempty"`
	ArmID                 *string                        `json:"arm_id,omitempty"`
	SelectedModel         *string                        `json:"selected_model,omitempty"`
	SelectionStatus       *string                        `json:"selection_status,omitempty"`
	SelectionMethod       *string                        `json:"selection_method,omitempty"`
	Recipe                *string                        `json:"recipe,omitempty"`
	DecisionName          *string                        `json:"decision_name,omitempty"`
	Algorithm             *string                        `json:"algorithm,omitempty"`
	InputTokens           *int64                         `json:"input_tokens,omitempty"`
	OutputTokens          *int64                         `json:"output_tokens,omitempty"`
	ResponseContentDigest *string                        `json:"response_content_digest,omitempty"`
	Quality               *float64                       `json:"quality,omitempty"`
	ControlledPair        *controlledPairObservation     `json:"controlled_pair,omitempty"`
	RoutingRecipeDecision *RoutingRecipeDecisionSnapshot `json:"routing_recipe_decision,omitempty"`

	// responsePayload exists only while binding one strictly decoded method
	// ledger to the worker-emitted rows. Ordinary upstream payloads are never
	// retained in the transcript.
	responsePayload map[string]any `json:"-"`
}

type brokerExecutionTranscript struct {
	SchemaVersion         string                      `json:"schema_version"`
	ContractVersion       string                      `json:"contract_version"`
	RunID                 string                      `json:"run_id"`
	ManifestDigest        string                      `json:"manifest_digest"`
	TargetID              string                      `json:"target_id"`
	Mode                  Mode                        `json:"mode"`
	PolicySnapshotDigest  string                      `json:"policy_snapshot_digest"`
	BackendTopologyDigest string                      `json:"backend_topology_digest"`
	StartedAt             time.Time                   `json:"started_at"`
	CompletedAt           time.Time                   `json:"completed_at"`
	Entries               []executionAttestationEntry `json:"entries"`
}

type executionAttestation struct {
	SchemaVersion         string                      `json:"schema_version"`
	ContractVersion       string                      `json:"contract_version"`
	RunID                 string                      `json:"run_id"`
	ManifestDigest        string                      `json:"manifest_digest"`
	TargetID              string                      `json:"target_id"`
	Mode                  Mode                        `json:"mode"`
	PolicySnapshotDigest  string                      `json:"policy_snapshot_digest"`
	BackendTopologyDigest string                      `json:"backend_topology_digest"`
	StartedAt             time.Time                   `json:"started_at"`
	CompletedAt           time.Time                   `json:"completed_at"`
	Entries               []executionAttestationEntry `json:"entries"`
	Digest                string                      `json:"digest"`
}

func brokerEntryReceipt(entry executionAttestationEntry) (string, error) {
	subject := map[string]any{
		"request_id": entry.RequestID, "operation": entry.Operation,
		"track_id": entry.TrackID, "case_id": entry.CaseID, "attempt_id": entry.AttemptID,
		"request_digest": entry.RequestDigest, "response_digest": entry.ResponseDigest,
		"upstream_attempted": entry.UpstreamAttempted, "success": entry.Success,
		"status_code": entry.StatusCode, "latency_microseconds": entry.LatencyMicroseconds,
		"fetched_at":       entry.FetchedAt,
		"ledger_sealed_at": entry.LedgerSealedAt,
		"headers":          entry.Headers, "requested_model": entry.RequestedModel,
		"arm_id": entry.ArmID, "selected_model": entry.SelectedModel,
		"selection_status": entry.SelectionStatus, "selection_method": entry.SelectionMethod,
		"recipe": entry.Recipe, "decision_name": entry.DecisionName, "algorithm": entry.Algorithm,
		"input_tokens": entry.InputTokens, "output_tokens": entry.OutputTokens,
		"response_content_digest": entry.ResponseContentDigest,
		"controlled_pair":         entry.ControlledPair,
		"routing_recipe_decision": entry.RoutingRecipeDecision,
	}
	digest, err := canonicalValueDigest(subject)
	if err != nil {
		return "", fmt.Errorf("digest evaluation broker receipt: %w", err)
	}
	return digest, nil
}

func executionAttestationDigest(attestation executionAttestation) (string, error) {
	attestation.Digest = ""
	digest, err := canonicalValueDigest(attestation)
	if err != nil {
		return "", fmt.Errorf("digest evaluation execution attestation: %w", err)
	}
	return digest, nil
}

func orderedExecutionEntries(entries map[uint64]executionAttestationEntry) []executionAttestationEntry {
	ids := make([]uint64, 0, len(entries))
	for id := range entries {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(left, right int) bool { return ids[left] < ids[right] })
	ordered := make([]executionAttestationEntry, 0, len(ids))
	for _, id := range ids {
		ordered = append(ordered, entries[id])
	}
	return ordered
}
