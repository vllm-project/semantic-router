package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// contextRecoveryContract is one request's effective recovery contract: the
// store every context action writes to and the bounds all of them share. It is
// resolved once, when the decision is bound, and every writer and the
// retrieval path read this value rather than their own plugin configuration.
type contextRecoveryContract struct {
	Store              string
	TTL                time.Duration
	MaxBytesPerRequest int
	MaxTotalBytes      int
	MaxRetrievals      int
}

// effectiveRecoveryContract resolves one action's settings to the values it
// would actually run with. An omitted bound means that action's default, never
// "unbounded", so merging cannot let an omitted value widen a configured one.
func effectiveRecoveryContract(
	recovery *config.ContextCompressionRecoveryConfig,
	bytesPerRequestDefault int,
) contextRecoveryContract {
	contract := contextRecoveryContract{
		Store:              strings.TrimSpace(recovery.Store),
		TTL:                contextcompression.DefaultRecoveryTTL,
		MaxBytesPerRequest: bytesPerRequestDefault,
		MaxTotalBytes:      contextcompression.DefaultRecoveryMaxTotalBytes,
		MaxRetrievals:      contextcompression.DefaultRecoveryMaxRetrievals,
	}
	if recovery.TTLSeconds > 0 {
		contract.TTL = time.Duration(recovery.TTLSeconds) * time.Second
	}
	if recovery.MaxBytesPerRequest > 0 {
		contract.MaxBytesPerRequest = recovery.MaxBytesPerRequest
	}
	if recovery.MaxTotalBytes > 0 {
		contract.MaxTotalBytes = recovery.MaxTotalBytes
	}
	if recovery.MaxRetrievals > 0 {
		contract.MaxRetrievals = recovery.MaxRetrievals
	}
	return contract
}

func activeRecoveryContracts(decision *config.Decision) []contextRecoveryContract {
	var active []contextRecoveryContract
	if compression := decision.GetContextCompressionConfig(); compression != nil &&
		compression.Enabled && compression.Recovery != nil && compression.Recovery.Enabled {
		active = append(active, effectiveRecoveryContract(
			compression.Recovery, contextcompression.DefaultCompressionRecoveryBytesPerRequest))
	}
	// A disabled action takes no part, even with a recovery block configured.
	if reset := decision.GetHistoryResetConfig(); reset.IsEnabled() && reset.RequiresRecovery() {
		active = append(active, effectiveRecoveryContract(
			reset.Recovery, config.DefaultHistoryResetRecoveryBytesPerRequest))
	}
	return active
}

// mergeRecoveryContracts merges the actions' effective contracts. Each action
// keeps its own per-request default; every bound then resolves to the
// stricter effective value, so neither action can widen the other's. The
// actions' stores must agree, which configuration validation and router
// construction both enforce before a decision can be bound.
func mergeRecoveryContracts(active []contextRecoveryContract) *contextRecoveryContract {
	if len(active) == 0 {
		return nil
	}
	merged := active[0]
	for _, next := range active[1:] {
		if next.TTL < merged.TTL {
			merged.TTL = next.TTL
		}
		merged.MaxBytesPerRequest = min(merged.MaxBytesPerRequest, next.MaxBytesPerRequest)
		merged.MaxTotalBytes = min(merged.MaxTotalBytes, next.MaxTotalBytes)
		merged.MaxRetrievals = min(merged.MaxRetrievals, next.MaxRetrievals)
	}
	return &merged
}

// bindContextPolicy binds the selected decision's context policies once the
// decision is known, on every path that runs the context stage. It binds the
// reset policy and the request's recovery contract with its request-scoped
// store, which carries the one byte budget all recovery writers share.
func (r *OpenAIRouter) bindContextPolicy(ctx *RequestContext) {
	if ctx.VSRSelectedDecision == nil {
		return
	}
	bindHistoryResetPolicy(ctx)
	contract := mergeRecoveryContracts(activeRecoveryContracts(ctx.VSRSelectedDecision))
	if contract == nil {
		return
	}
	ctx.ContextRecovery = contract
	if store := r.contextRecoveryStore(*contract); store != nil {
		ctx.ContextRecoveryStore = contextcompression.NewRequestRecoveryStore(store, contract.MaxBytesPerRequest)
	}
}

// applyContextRecoveryContract projects the request's contract onto a
// compression policy, so a compression write uses the same bounds as reset.
func applyContextRecoveryContract(policy *contextcompression.RecoveryPolicy, contract *contextRecoveryContract) {
	if contract == nil || !policy.Enabled {
		return
	}
	policy.TTL = contract.TTL
	policy.MaxBytesPerRequest = contract.MaxBytesPerRequest
	policy.MaxTotalBytes = contract.MaxTotalBytes
	policy.MaxRetrievals = contract.MaxRetrievals
}
