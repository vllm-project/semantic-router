package kvtransfer

import (
	"context"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// SourceLookup returns a last-known candidate in the authenticated namespace.
// Records must name the actual serving endpoint and immutable model identity.
type SourceLookup interface {
	LookupSource(context.Context, string, string) (*SourceCache, error)
}

// PairPolicy permits one directional mapper up to an inclusive turn bound.
// Its zero value disables transfer. Turn zero is the initial prompt.
type PairPolicy struct {
	Mapper          Mapper
	Enabled         bool
	MaxTransferTurn int
}

// Coordinator combines explicit policy, source discovery, and B2 eligibility.
// Initialize its lookup and policies before serving requests.
type Coordinator struct {
	Lookup   SourceLookup
	Policies []PairPolicy
	// CanExport resolves the discovered source endpoint, separately from target capabilities.
	CanExport func(SourceCache) bool
}

const (
	ReasonDisabled          Reason = "disabled"
	ReasonTurnLimit         Reason = "turn_limit"
	ReasonSourceUnavailable Reason = "source_unavailable"
	ReasonPairNotAllowed    Reason = "pair_not_allowed"
)

// Plan builds a candidate only for a permitted switch. A failed lookup is a
// normal-prefill result, never a request failure or evidence of a cache hit.
func (c *Coordinator) Plan(ctx context.Context, request Request, turn int) (*Hint, Reason) {
	if c == nil || c.Lookup == nil {
		return nil, ReasonDisabled
	}
	if !present(request.AuthenticatedPrincipal, request.SessionID) ||
		(request.SessionProvenance != "header" && request.SessionProvenance != "response_api") ||
		!cache.UserScopeSecretConfigured() {
		return nil, ReasonUntrustedSession
	}
	if turn < 0 {
		return nil, ReasonTurnLimit
	}
	enabled, bounded := false, false
	for _, p := range c.Policies {
		if p.Enabled && p.Mapper.Target == request.Target {
			enabled = true
			if p.MaxTransferTurn >= turn {
				bounded = true
			}
		}
	}
	if !enabled {
		return nil, ReasonPairNotAllowed
	}
	if !bounded {
		return nil, ReasonTurnLimit
	}
	source, err := c.Lookup.LookupSource(ctx, cache.UserScopeNamespace(request.AuthenticatedPrincipal), request.SessionID)
	if err != nil || source == nil {
		return nil, ReasonSourceUnavailable
	}
	if c.CanExport != nil {
		request.SourceCanExport = c.CanExport(*source)
	}
	for _, p := range c.Policies {
		if p.Enabled && p.MaxTransferTurn >= turn && p.Mapper.Source == source.Model && p.Mapper.Target == request.Target {
			return PlanHandoff(request, *source, p.Mapper)
		}
	}
	return nil, ReasonPairNotAllowed
}

// Headers projects an approved candidate onto the established transfer wire.
func (h Hint) Headers() map[string]string {
	return map[string]string{headers.VSRKVSourcePod: h.SourceEndpoint, headers.VSRKVCacheID: h.CacheID, headers.VSRKVMapperID: h.MapperID}
}
