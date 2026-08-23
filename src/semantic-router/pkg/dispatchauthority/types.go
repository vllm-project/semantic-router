// Package dispatchauthority binds admitted inference requests to short-lived,
// request-exact backend dispatch capabilities.
package dispatchauthority

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

const (
	primaryDispatchType = "primary"
	grantedDispatchType = "looper"
)

// MeteredAuthorityOptions contains the only dependencies accepted by the
// metered production authority. Issuer key material is copied by the authority
// and is zeroed when Close returns.
type MeteredAuthorityOptions struct {
	Access *accessruntime.Runtime
	Issuer backendinvoker.CapabilityIssuerOptions
}

// ModelIdentity pins a model from one immutable routing snapshot.
type ModelIdentity struct {
	ID       string
	Revision int64
}

// FinalRequest is the exact provider-neutral HTTP request that will cross the
// private backend-dispatch boundary. Provider credentials and wire adaptation
// are deliberately applied later by the backend invoker.
type FinalRequest struct {
	Model      ModelIdentity
	Method     string
	Path       string
	Query      string
	WireFormat llmprotocol.WireFormat
	Body       []byte
}

// PrimaryIssueRequest authorizes one direct model dispatch. It intentionally
// contains no namespace, partition, routing revision, admission identity, or
// dispatch type: those fields are derived from the opaque admitted request.
type PrimaryIssueRequest struct {
	Admission accessruntime.Admission
	Dispatch  accessruntime.DispatchFacts
	RequestID string
	Final     FinalRequest
}

// GrantIssueRequest authorizes one internal model hop to mint its exact
// capability only after the final nested request body is known.
type GrantIssueRequest struct {
	Admission accessruntime.Admission
	Dispatch  accessruntime.DispatchFacts
	RequestID string
	Model     ModelIdentity
}

// RoutingPublicationReader exposes only the process-local active publication
// lease used by routing-only inference. It must never perform a request-time
// database lookup.
type RoutingPublicationReader interface {
	CurrentRoutingPublication(string) (accesspublisher.RuntimePublicationIdentity, bool)
}

// RoutingOnlyAuthorityOptions composes the explicitly public, unmetered data
// plane. It still pins every dispatch to one active immutable generation and
// never accepts caller-supplied admission identities.
type RoutingOnlyAuthorityOptions struct {
	NamespaceID  string
	Publications RoutingPublicationReader
	Issuer       backendinvoker.CapabilityIssuerOptions
}

type RoutingOnlyIssueRequest struct {
	Generation routingcontext.Generation
	Dispatch   accessruntime.DispatchFacts
	RequestID  string
	Final      FinalRequest
}

type RoutingOnlyGrantIssueRequest struct {
	Generation routingcontext.Generation
	Dispatch   accessruntime.DispatchFacts
	RequestID  string
	Model      ModelIdentity
}

// CandidateIssue contains one independently journaled Model dispatch in an
// ordered fallback chain. Priority is the contiguous route tier beginning at
// zero; each candidate retains its request-wide dispatch ordinal.
type CandidateIssue struct {
	Dispatch accessruntime.DispatchFacts
	Model    ModelIdentity
	Priority int
}

// ChainFinalRequest is one exact provider-neutral request shared immutably by
// every candidate in a fallback chain.
type ChainFinalRequest struct {
	Method     string
	Path       string
	Query      string
	WireFormat llmprotocol.WireFormat
	Body       []byte
}

type MeteredChainIssueRequest struct {
	Admission  accessruntime.Admission
	Candidates []CandidateIssue
	Fallback   backendinvoker.FallbackPolicy
	RequestID  string
	Final      ChainFinalRequest
}

type RoutingOnlyChainIssueRequest struct {
	Generation routingcontext.Generation
	Candidates []CandidateIssue
	Fallback   backendinvoker.FallbackPolicy
	RequestID  string
	Final      ChainFinalRequest
}

// FallbackCapabilityRuntime is the chain-native authority seam. It is kept
// separate from CapabilityRuntime until the ExtProc selection layer supplies
// complete, independently journaled candidate dispatches.
type FallbackCapabilityRuntime interface {
	IssueMeteredChain(MeteredChainIssueRequest) (string, error)
	IssueRoutingOnlyChain(context.Context, RoutingOnlyChainIssueRequest) (string, error)
}

// OutcomeVerificationRequest pins a signed private-dispatch outcome to the
// exact Router generation and request that will consume it.
type OutcomeVerificationRequest struct {
	Generation routingcontext.Generation
	RequestID  string
}

// OutcomeRuntime authenticates private response evidence with the authority's
// existing capability keyring. Request-local candidate matching remains an
// ExtProc responsibility because only ExtProc owns the dispatch journal.
type OutcomeRuntime interface {
	VerifyDispatchOutcome(context.Context, string, OutcomeVerificationRequest) (backendinvoker.DispatchOutcome, error)
}

type GrantVerificationRequest struct {
	Generation routingcontext.Generation
	RequestID  string
}

// RoutingSnapshotAttacher binds the exact process-local snapshot registry once
// it has been constructed. Issuance is unavailable until attachment succeeds.
type RoutingSnapshotAttacher interface {
	AttachRoutingSnapshots(backendinvoker.RoutingSnapshotSource) error
}

// CapabilityRuntime is the only authority exposed to ExtProc. Mode-specific
// methods keep metered and routing-only authorization impossible to confuse.
type CapabilityRuntime interface {
	Metered() bool
	IssueMeteredPrimary(PrimaryIssueRequest) (string, error)
	IssueMeteredGrant(GrantIssueRequest) (string, error)
	IssueRoutingOnlyPrimary(context.Context, RoutingOnlyIssueRequest) (string, error)
	IssueRoutingOnlyGrant(context.Context, RoutingOnlyGrantIssueRequest) (string, error)
	VerifyGrant(context.Context, string, GrantVerificationRequest) (VerifiedGrant, error)
	IssueFromGrant(context.Context, VerifiedGrant, FinalRequest) (string, error)
}

// VerifiedGrant is process-local proof that a serialized internal dispatch
// grant was verified by this exact authority. Its zero value has no authority.
type VerifiedGrant struct {
	grant      backendinvoker.VerifiedDispatchGrant
	owner      *authorityIdentity
	generation routingcontext.Generation
	requestID  string
}

type authorityIdentity struct {
	marker byte
}

type preparedIdentity struct {
	namespaceID        string
	quotaPartition     string
	publicationID      string
	runtimeEpoch       uint64
	routingRevision    int64
	routingDigest      string
	admissionID        string
	admissionDigest    string
	requestID          string
	dispatchID         string
	ordinal            uint32
	dispatchPlanDigest string
}
