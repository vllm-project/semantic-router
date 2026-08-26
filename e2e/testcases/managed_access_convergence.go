package testcases

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type managedAccessRuntimeDiagnostics struct {
	Status    string `json:"status"`
	Namespace *struct {
		NamespaceID string                   `json:"namespaceId"`
		Publication managedAccessPublication `json:"publication"`
	} `json:"namespace"`
}

type managedAccessPublication struct {
	Readiness                       managedAccessPublicationReadiness `json:"readiness"`
	ActiveReplicas                  []string                          `json:"activeReplicas"`
	RecordedRequiredReplicas        []string                          `json:"recordedRequiredReplicas"`
	BarrierAcknowledgementsRequired bool                              `json:"barrierAcknowledgementsRequired"`
	BarrierAcknowledgements         []string                          `json:"barrierAcknowledgements"`
	RoutingAcknowledgements         []string                          `json:"routingAcknowledgements"`
	MissingBarrierAcks              []string                          `json:"missingBarrierAcks"`
	MissingRoutingAcks              []string                          `json:"missingRoutingAcks"`
}

type managedAccessPublicationReadiness struct {
	Ready           bool   `json:"ready"`
	RuntimeEpoch    uint64 `json:"runtimeEpoch"`
	DesiredRevision uint64 `json:"desiredRevision"`
	AppliedRevision uint64 `json:"appliedRevision"`
}

func waitManagedAccessReplicaConvergence(
	ctx context.Context,
	client *managedAccessClient,
	namespaceID string,
	afterRevision uint64,
) (uint64, error) {
	deadline := time.Now().Add(90 * time.Second)
	lastReason := "runtime diagnostics are not visible"
	for time.Now().Before(deadline) {
		revision, reason, err := readManagedAccessReplicaConvergence(
			ctx, client, namespaceID, afterRevision,
		)
		if err != nil {
			return 0, err
		}
		if reason == "" {
			return revision, nil
		}
		lastReason = reason
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return 0, ctx.Err()
		case <-timer.C:
		}
	}
	return 0, fmt.Errorf("two-replica publication did not converge: %s", lastReason)
}

func readManagedAccessReplicaConvergence(
	ctx context.Context,
	client *managedAccessClient,
	namespaceID string,
	afterRevision uint64,
) (uint64, string, error) {
	query := url.Values{"namespaceId": []string{namespaceID}}
	var diagnostics managedAccessRuntimeDiagnostics
	status, _, err := client.request(
		ctx, "", http.MethodGet, "/runtime-diagnostics?"+query.Encode(), "", nil, nil,
		[]int{http.StatusOK}, &diagnostics,
	)
	if managedAccessRuntimeDiagnosticsPending(status, err) {
		return 0, "runtime diagnostics Namespace is not visible yet", nil
	}
	if err != nil {
		return 0, "", fmt.Errorf("read public runtime diagnostics: %w", err)
	}
	revision, reason := managedAccessReplicaConvergenceReason(diagnostics, namespaceID, afterRevision)
	return revision, reason, nil
}

func managedAccessRuntimeDiagnosticsPending(status int, err error) bool {
	if status != http.StatusNotFound || err == nil {
		return false
	}
	var responseErr *managedAccessResponseError
	return errors.As(err, &responseErr) && responseErr.code == "not_found"
}

func managedAccessReplicaConvergenceReason(
	diagnostics managedAccessRuntimeDiagnostics,
	namespaceID string,
	afterRevision uint64,
) (uint64, string) {
	if diagnostics.Status != "ready" || diagnostics.Namespace == nil {
		return 0, "runtime diagnostics are not ready"
	}
	if diagnostics.Namespace.NamespaceID != namespaceID {
		return 0, "runtime diagnostics selected a different Namespace"
	}
	publication := diagnostics.Namespace.Publication
	readiness := publication.Readiness
	if reason := managedAccessReadinessReason(readiness, afterRevision); reason != "" {
		return readiness.AppliedRevision, reason
	}
	if len(publication.ActiveReplicas) != 2 {
		return readiness.AppliedRevision, fmt.Sprintf(
			"active Router replica count is %d, want 2", len(publication.ActiveReplicas),
		)
	}
	if !managedAccessSameReplicaSet(publication.ActiveReplicas, publication.RecordedRequiredReplicas) {
		return readiness.AppliedRevision, "required Router replicas do not match the active replica set"
	}
	if reason := managedAccessBarrierReason(publication); reason != "" {
		return readiness.AppliedRevision, reason
	}
	if !managedAccessSameReplicaSet(publication.ActiveReplicas, publication.RoutingAcknowledgements) {
		return readiness.AppliedRevision, "routing acknowledgements do not cover both Router replicas"
	}
	if len(publication.MissingRoutingAcks) != 0 {
		return readiness.AppliedRevision, "runtime diagnostics still report missing routing acknowledgements"
	}
	return readiness.AppliedRevision, ""
}

func managedAccessReadinessReason(
	readiness managedAccessPublicationReadiness,
	afterRevision uint64,
) string {
	ready := readiness.Ready && readiness.RuntimeEpoch > 0 && readiness.DesiredRevision > 0 &&
		readiness.DesiredRevision == readiness.AppliedRevision && readiness.AppliedRevision > afterRevision
	if ready {
		return ""
	}
	return fmt.Sprintf(
		"publication readiness is ready=%t epoch=%d desired=%d applied=%d after=%d",
		readiness.Ready, readiness.RuntimeEpoch, readiness.DesiredRevision,
		readiness.AppliedRevision, afterRevision,
	)
}

func managedAccessBarrierReason(publication managedAccessPublication) string {
	if !publication.BarrierAcknowledgementsRequired {
		return managedAccessNonRestrictiveBarrierReason(publication)
	}
	if !managedAccessSameReplicaSet(publication.ActiveReplicas, publication.BarrierAcknowledgements) {
		return "access barrier acknowledgements do not cover both Router replicas"
	}
	if len(publication.MissingBarrierAcks) != 0 {
		return "runtime diagnostics still report missing barrier acknowledgements"
	}
	return ""
}

func managedAccessNonRestrictiveBarrierReason(publication managedAccessPublication) string {
	if len(publication.BarrierAcknowledgements) != 0 || len(publication.MissingBarrierAcks) != 0 {
		return "non-restrictive publication reported barrier acknowledgement state"
	}
	return ""
}

func managedAccessSameReplicaSet(left []string, right []string) bool {
	if len(left) != len(right) || len(left) == 0 {
		return false
	}
	members := make(map[string]struct{}, len(left))
	for _, replicaID := range left {
		if strings.TrimSpace(replicaID) == "" {
			return false
		}
		members[replicaID] = struct{}{}
	}
	if len(members) != len(left) {
		return false
	}
	for _, replicaID := range right {
		if _, found := members[replicaID]; !found {
			return false
		}
	}
	return true
}
