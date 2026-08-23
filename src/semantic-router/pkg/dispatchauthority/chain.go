package dispatchauthority

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
)

// IssueChain authorizes one ordered sequence of independently journaled Model
// dispatches. Every candidate is rebound to the same opaque Admission before
// the request-exact capability is signed.
func (authority *MeteredAuthority) IssueChain(request MeteredChainIssueRequest) (string, error) {
	if authority == nil {
		return "", fmt.Errorf("metered dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.preparer == nil || authority.issuer == nil {
		return "", fmt.Errorf("metered dispatch authority is closed")
	}
	prepared, candidates, err := authority.prepareChain(request.Admission, request.Candidates, primaryDispatchType)
	if err != nil {
		return "", err
	}
	prepared.requestID = request.RequestID
	return authority.issuer.Issue(chainCapabilityRequest(
		prepared, candidates, request.Fallback, request.Final,
	))
}

func (authority *MeteredAuthority) prepareChain(
	admission accessruntime.Admission,
	issues []CandidateIssue,
	dispatchType string,
) (preparedIdentity, []backendinvoker.DispatchCandidate, error) {
	if len(issues) == 0 {
		return preparedIdentity{}, nil, fmt.Errorf("metered fallback chain is empty")
	}
	candidates := make([]backendinvoker.DispatchCandidate, 0, len(issues))
	var common preparedIdentity
	for index, issue := range issues {
		prepared, err := authority.preparer.prepare(admission, issue.Dispatch)
		if err != nil {
			return preparedIdentity{}, nil, fmt.Errorf("prepare metered fallback candidate %d: %w", index, err)
		}
		if index == 0 {
			common = prepared
		} else if !samePreparedRequest(common, prepared) {
			return preparedIdentity{}, nil, fmt.Errorf("metered fallback candidates do not share one admission")
		}
		candidates = append(candidates, dispatchCandidate(prepared, issue, dispatchType))
	}
	return common, candidates, nil
}

func (authority *RoutingOnlyAuthority) IssueChain(
	ctx context.Context,
	request RoutingOnlyChainIssueRequest,
) (string, error) {
	if authority == nil {
		return "", fmt.Errorf("routing-only dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.issuer == nil {
		return "", fmt.Errorf("routing-only dispatch authority is closed")
	}
	if len(request.Candidates) == 0 {
		return "", fmt.Errorf("routing-only fallback chain is empty")
	}
	candidates := make([]backendinvoker.DispatchCandidate, 0, len(request.Candidates))
	var common preparedIdentity
	for index, issue := range request.Candidates {
		prepared, err := authority.prepareLocked(
			ctx, request.Generation, request.RequestID, issue.Dispatch, issue.Model,
		)
		if err != nil {
			return "", fmt.Errorf("prepare routing-only fallback candidate %d: %w", index, err)
		}
		if index == 0 {
			common = prepared
		} else if !samePreparedRequest(common, prepared) {
			return "", fmt.Errorf("routing-only fallback candidates do not share one generation")
		}
		candidates = append(candidates, dispatchCandidate(prepared, issue, primaryDispatchType))
	}
	return authority.issuer.Issue(chainCapabilityRequest(
		common, candidates, request.Fallback, request.Final,
	))
}

func chainCapabilityRequest(
	prepared preparedIdentity,
	candidates []backendinvoker.DispatchCandidate,
	fallback backendinvoker.FallbackPolicy,
	request ChainFinalRequest,
) backendinvoker.CapabilityIssueRequest {
	return backendinvoker.CapabilityIssueRequest{
		NamespaceID: prepared.namespaceID, QuotaPartition: prepared.quotaPartition,
		PublicationID: prepared.publicationID, RuntimeEpoch: prepared.runtimeEpoch,
		RoutingRevision: prepared.routingRevision, RoutingDigest: prepared.routingDigest,
		AdmissionID: prepared.admissionID, AdmissionDigest: prepared.admissionDigest,
		RequestID: prepared.requestID, Candidates: candidates, Fallback: fallback,
		Method: request.Method, Path: request.Path, Query: request.Query, WireFormat: request.WireFormat, Body: request.Body,
	}
}

func dispatchCandidate(
	prepared preparedIdentity,
	issue CandidateIssue,
	dispatchType string,
) backendinvoker.DispatchCandidate {
	return backendinvoker.DispatchCandidate{
		DispatchID: prepared.dispatchID, DispatchType: dispatchType,
		Ordinal: int(prepared.ordinal), DispatchPlanDigest: prepared.dispatchPlanDigest,
		ModelID: issue.Model.ID, ModelRevision: issue.Model.Revision, Priority: issue.Priority,
	}
}

func samePreparedRequest(left, right preparedIdentity) bool {
	return left.namespaceID == right.namespaceID && left.quotaPartition == right.quotaPartition &&
		left.publicationID == right.publicationID && left.runtimeEpoch == right.runtimeEpoch &&
		left.routingRevision == right.routingRevision && left.routingDigest == right.routingDigest &&
		left.admissionID == right.admissionID && left.admissionDigest == right.admissionDigest &&
		left.requestID == right.requestID
}
