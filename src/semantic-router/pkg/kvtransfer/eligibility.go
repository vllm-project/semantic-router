// Package kvtransfer defines the router-side contract for cross-model KV handoff.
package kvtransfer

import (
	"encoding/hex"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ModelIdentity names the exact serving configuration of a model. A routing
// alias is not a model identity; revisions must identify immutable weights and
// tokenizer files.
type ModelIdentity = config.KVServingIdentity

// Mapper identifies an artifact fitted for one source and target configuration.
type Mapper struct {
	ID     string
	Source ModelIdentity
	Target ModelIdentity
}

// SourceCache is a last-known cache location for one tenant-scoped session.
// Endpoint must be the selected upstream endpoint that served the source turn,
// not a configured load-balancer address. A record is only a candidate: the
// target connector must verify the token prefix and cache contents.
type SourceCache struct {
	Namespace string
	SessionID string
	CacheID   string
	Endpoint  string
	Model     ModelIdentity
	ExpiresAt time.Time
}

// Request describes the target turn and the backend capabilities known to the
// router. The source record and mapper are supplied by later integration steps.
type Request struct {
	// Principal must come from the configured authentication gateway. A
	// client-provided session ID alone cannot authorize cache reuse.
	AuthenticatedPrincipal string
	// SessionProvenance uses the RequestContext provenance values. Only an
	// explicit session or retained Response API lineage may reuse state.
	SessionProvenance string
	SessionID         string
	Target            ModelIdentity
	SourceCanExport   bool
	TargetCanLoad     bool
	Now               time.Time
}

// Hint is an eligible transfer candidate. It is not evidence of a cache hit.
// The endpoint is kept here for a later transport step; the same-host vLLM
// connector currently consumes Namespace, CacheID, and MapperID.
type Hint struct {
	Namespace      string
	CacheID        string
	MapperID       string
	SourceEndpoint string
}

// Reason lets the caller distinguish an eligible candidate from a safe no-op.
type Reason string

const (
	ReasonEligible           Reason = "eligible"
	ReasonMissingInput       Reason = "missing_input"
	ReasonUntrustedSession   Reason = "untrusted_session"
	ReasonUnpinnedIdentity   Reason = "unpinned_identity"
	ReasonUnavailableBackend Reason = "unavailable_backend"
	ReasonScopeMismatch      Reason = "scope_mismatch"
	ReasonStaleSource        Reason = "stale_source"
	ReasonSameModel          Reason = "same_model"
	ReasonIdentityMismatch   Reason = "identity_mismatch"
	ReasonVocabularyMismatch Reason = "vocabulary_mismatch"
	ReasonUnsupportedServing Reason = "unsupported_serving"
)

// PlanHandoff returns a candidate only when every router-visible condition is
// satisfied. A nil hint means the target request must use normal prefill.
func PlanHandoff(request Request, source SourceCache, mapper Mapper) (*Hint, Reason) {
	if !present(request.SessionID, source.Namespace,
		source.SessionID, source.CacheID, source.Endpoint, mapper.ID) ||
		request.Now.IsZero() {
		return nil, ReasonMissingInput
	}
	if !present(request.AuthenticatedPrincipal) ||
		(request.SessionProvenance != "header" && request.SessionProvenance != "response_api") ||
		!cache.UserScopeSecretConfigured() {
		return nil, ReasonUntrustedSession
	}
	namespace := cache.UserScopeNamespace(request.AuthenticatedPrincipal)
	if !validIdentity(request.Target) || !validIdentity(source.Model) ||
		!validIdentity(mapper.Source) || !validIdentity(mapper.Target) {
		return nil, ReasonUnpinnedIdentity
	}
	if !request.SourceCanExport || !request.TargetCanLoad {
		return nil, ReasonUnavailableBackend
	}
	if namespace != source.Namespace || request.SessionID != source.SessionID {
		return nil, ReasonScopeMismatch
	}
	if source.ExpiresAt.IsZero() || !request.Now.Before(source.ExpiresAt) {
		return nil, ReasonStaleSource
	}
	if source.Model == request.Target {
		return nil, ReasonSameModel
	}
	if source.Model != mapper.Source || request.Target != mapper.Target {
		return nil, ReasonIdentityMismatch
	}
	if source.Model.Tokenizer != request.Target.Tokenizer ||
		source.Model.TokenizerRevision != request.Target.TokenizerRevision {
		return nil, ReasonVocabularyMismatch
	}
	if !supportedServing(source.Model) || !supportedServing(request.Target) {
		return nil, ReasonUnsupportedServing
	}
	return &Hint{
		Namespace:      namespace,
		CacheID:        source.CacheID,
		MapperID:       mapper.ID,
		SourceEndpoint: source.Endpoint,
	}, ReasonEligible
}

func present(values ...string) bool {
	for _, value := range values {
		if value == "" || strings.TrimSpace(value) != value {
			return false
		}
	}
	return true
}

func validIdentity(identity ModelIdentity) bool {
	return present(identity.Model, identity.Tokenizer, identity.Precision, identity.HeadOrder) &&
		pinnedCommit(identity.WeightRevision) && pinnedCommit(identity.TokenizerRevision) &&
		identity.TensorParallel > 0 && identity.KVHeads > 0 && identity.HeadDim > 0
}

func pinnedCommit(revision string) bool {
	if len(revision) != 40 {
		return false
	}
	_, err := hex.DecodeString(revision)
	return err == nil
}

func supportedServing(identity ModelIdentity) bool {
	return identity.AdapterID == "" && identity.Precision == "bf16" &&
		identity.TensorParallel == 1 && identity.HeadOrder == "contiguous"
}
