package backendinvoker

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// RoutingSnapshotSource returns the exact immutable routing revision pinned by
// a dispatch capability. Implementations may read a standalone in-memory
// snapshot or a managed active-snapshot cache, but must verify the complete pin
// and never silently fall back to another revision or digest.
type RoutingSnapshotSource interface {
	Snapshot(context.Context, routingcontext.Generation) (*routingsnapshot.Snapshot, error)
}

// SnapshotPlanResolver converts every immutable candidate Model revision into
// one ordered plan chain from the exact snapshot pinned by the capability.
type SnapshotPlanResolver struct {
	Source RoutingSnapshotSource
}

var _ PlanResolver = (*SnapshotPlanResolver)(nil)

func (r *SnapshotPlanResolver) ResolvePlans(ctx context.Context, capability DispatchCapability) (PlanChain, error) {
	if r == nil || r.Source == nil {
		return PlanChain{}, fmt.Errorf("routing snapshot source is required")
	}
	pin := routingcontext.Generation{
		NamespaceID: capability.NamespaceID, QuotaPartition: capability.QuotaPartition,
		PublicationID: capability.PublicationID, RuntimeEpoch: capability.RuntimeEpoch,
		SnapshotRevision: capability.RoutingRevision,
		RoutingDigest:    capability.RoutingDigest,
	}
	snapshot, err := r.Source.Snapshot(ctx, pin)
	if err != nil {
		return PlanChain{}, fmt.Errorf("load routing snapshot: %w", err)
	}
	if snapshot == nil || snapshot.NamespaceID != capability.NamespaceID ||
		snapshot.Revision != capability.RoutingRevision {
		return PlanChain{}, fmt.Errorf("routing snapshot identity mismatch")
	}
	resolved := PlanChain{
		Fallback:   cloneFallbackPolicy(capability.Fallback),
		Candidates: make([]Plan, 0, len(capability.Candidates)),
	}
	for index, candidate := range capability.Candidates {
		plan, resolveErr := resolveSnapshotCandidate(snapshot, capability, candidate)
		if resolveErr != nil {
			return PlanChain{}, fmt.Errorf("resolve dispatch candidate %d: %w", index, resolveErr)
		}
		resolved.Candidates = append(resolved.Candidates, plan)
	}
	return resolved, nil
}

func resolveSnapshotCandidate(
	snapshot *routingsnapshot.Snapshot,
	capability DispatchCapability,
	candidate DispatchCandidate,
) (Plan, error) {
	model, found := snapshot.Model(candidate.ModelID)
	if !found || model.Revision != candidate.ModelRevision {
		return Plan{}, fmt.Errorf("model revision is not present in the pinned snapshot")
	}
	requestTimeout, err := time.ParseDuration(model.Execution.RequestTimeout)
	if err != nil {
		return Plan{}, fmt.Errorf("invalid compiled request timeout: %w", err)
	}
	streamTimeout, err := time.ParseDuration(model.Execution.StreamTimeout)
	if err != nil {
		return Plan{}, fmt.Errorf("invalid compiled stream timeout: %w", err)
	}
	backends := make([]Backend, 0, len(model.Backends))
	var totalWeight uint64
	for _, backend := range model.Backends {
		weight, err := runtimeWeight(backend.Weight)
		if err != nil {
			return Plan{}, fmt.Errorf("backend %s weight: %w", backend.ID, err)
		}
		if ^uint64(0)-totalWeight < weight {
			return Plan{}, fmt.Errorf("combined backend weight exceeds the runtime range")
		}
		totalWeight += weight
		backends = append(backends, Backend{
			ID: backend.ID, Origin: backend.Origin, ProviderID: backend.ProviderID,
			WireFormat:           backend.WireFormat,
			ProviderModelID:      backend.ProviderModelID,
			ProviderCredentialID: backend.ProviderCredentialID,
			Connection:           runtimeConnection(backend.Connection), Weight: weight,
		})
	}
	return Plan{
		NamespaceID: capability.NamespaceID, QuotaPartition: capability.QuotaPartition,
		PublicationID: capability.PublicationID, RuntimeEpoch: capability.RuntimeEpoch,
		RoutingRevision: capability.RoutingRevision, RoutingDigest: capability.RoutingDigest,
		AdmissionID:     capability.AdmissionID,
		AdmissionDigest: capability.AdmissionDigest,
		RequestID:       capability.RequestID,
		DispatchID:      candidate.DispatchID, DispatchType: candidate.DispatchType,
		Ordinal: candidate.Ordinal, Priority: candidate.Priority,
		DispatchPlanDigest: candidate.DispatchPlanDigest,
		ModelID:            model.ID, ModelRevision: model.Revision,
		Execution: Execution{
			MaxRetries: model.Execution.MaxRetries, RetryOn: runtimeRetryTriggers(model.Execution.RetryOn),
			RequestTimeout: requestTimeout, StreamTimeout: streamTimeout,
		},
		Backends: backends, RequestDigest: capability.RequestDigest, SourceFormat: capability.WireFormat,
	}, nil
}

func runtimeRetryTriggers(source []string) []FallbackTrigger {
	result := make([]FallbackTrigger, len(source))
	for index, trigger := range source {
		result[index] = FallbackTrigger(trigger)
	}
	return result
}

func runtimeConnection(source routingsnapshot.BackendConnection) Connection {
	connection := Connection{
		Path: source.Path, Headers: make(http.Header, len(source.Headers)),
	}
	for name, value := range source.Headers {
		connection.Headers.Set(name, value)
	}
	return connection
}

// runtimeWeight maps a canonical decimal with at most nine fractional digits
// to an exact positive integer weight. A common 10^9 scale preserves every
// ratio without floating-point rounding.
func runtimeWeight(value string) (uint64, error) {
	whole, fraction, found := strings.Cut(value, ".")
	if whole == "" || len(fraction) > 9 {
		return 0, fmt.Errorf("weight is not a canonical decimal")
	}
	for len(fraction) < 9 {
		fraction += "0"
	}
	wholeValue, err := strconv.ParseUint(whole, 10, 64)
	if err != nil || wholeValue > (^uint64(0)-999_999_999)/1_000_000_000 {
		return 0, fmt.Errorf("weight exceeds the runtime range")
	}
	fractionValue := uint64(0)
	if found {
		parsed, parseErr := strconv.ParseUint(fraction, 10, 64)
		if parseErr != nil {
			return 0, fmt.Errorf("weight is not a canonical decimal")
		}
		fractionValue = parsed
	}
	weight := wholeValue*1_000_000_000 + fractionValue
	if weight == 0 {
		return 0, fmt.Errorf("weight must be positive")
	}
	return weight, nil
}
