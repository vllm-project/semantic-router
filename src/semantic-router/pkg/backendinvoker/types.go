// Package backendinvoker owns every physical model attempt. Envoy and callers
// select neither backend destinations nor provider credentials.
package backendinvoker

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type UsageState string

const (
	UsageKnownZero   UsageState = "known_zero"
	UsageKnownActual UsageState = "known_actual"
	UsageUnknown     UsageState = "unknown"
)

type AttemptState string

const (
	AttemptKnownZero       AttemptState = "known_zero"
	AttemptResponseStarted AttemptState = "response_started"
	AttemptUnknown         AttemptState = "unknown"
)

// FallbackTrigger is the closed set of failure evidence that a published
// route may authorize for cross-Model fallback. A trigger is only actionable
// when the attempt is also proven known-zero before any request or response
// bytes crossed the transport boundary.
type FallbackTrigger string

const (
	FallbackUnavailable FallbackTrigger = "unavailable"
	FallbackOverloaded  FallbackTrigger = "overloaded"
	FallbackTimeout     FallbackTrigger = "timeout"
)

const maximumDispatchCandidates = 32

// FallbackPolicy is immutable capability input. On is canonical-order,
// duplicate-free, and bounded by the closed FallbackTrigger vocabulary.
type FallbackPolicy struct {
	On []FallbackTrigger `json:"on"`
}

type Backend struct {
	ID                        string
	Origin                    string
	ProviderID                string
	WireFormat                llmprotocol.WireFormat
	ProviderModelID           string
	ProviderCredentialID      string
	ProviderCredentialVersion string
	Connection                Connection
	Weight                    uint64
}

// Connection contains only non-secret, compiled wire values. Product identity
// and credential material are deliberately separate from this protocol input.
type Connection struct {
	Path    string
	Headers http.Header
}

type Execution struct {
	MaxRetries     int
	RequestTimeout time.Duration
	StreamTimeout  time.Duration
}

type Plan struct {
	NamespaceID        string
	QuotaPartition     string
	PublicationID      string
	RuntimeEpoch       uint64
	RoutingRevision    int64
	RoutingDigest      string
	AdmissionID        string
	AdmissionDigest    string
	RequestID          string
	DispatchID         string
	DispatchType       string
	Ordinal            int
	Priority           int
	DispatchPlanDigest string
	ModelID            string
	ModelRevision      int64
	Method             string
	Path               string
	Query              string
	Headers            http.Header
	Body               []byte
	Streaming          bool
	SourceFormat       llmprotocol.WireFormat
	Execution          Execution
	Backends           []Backend
	RequestDigest      string
}

// PlanChain is the exact ordered cross-Model execution plan resolved from one
// signed capability and one pinned routing snapshot. Candidates contain the
// same request bytes and common routing identity but distinct dispatch and
// Model identities.
type PlanChain struct {
	Fallback   FallbackPolicy
	Candidates []Plan
}

type Credential struct {
	Header  string
	Prefix  string
	Secret  string
	Extra   http.Header
	Version string
}

// CredentialPublication pins ProviderCredential resolution to the same
// namespace publication as the routing plan. A resolver must never follow a
// mutable active pointer or substitute another publication.
type CredentialPublication struct {
	NamespaceID    string
	QuotaPartition string
	PublicationID  string
}

func (identity CredentialPublication) Validate() error {
	if strings.TrimSpace(identity.NamespaceID) == "" || strings.TrimSpace(identity.QuotaPartition) == "" ||
		strings.TrimSpace(identity.PublicationID) == "" {
		return fmt.Errorf("credential publication identity is required")
	}
	if len(identity.NamespaceID) > 256 || len(identity.QuotaPartition) > 256 || len(identity.PublicationID) > 256 ||
		strings.ContainsRune(identity.NamespaceID, 0) || strings.ContainsRune(identity.QuotaPartition, 0) ||
		strings.ContainsRune(identity.PublicationID, 0) {
		return fmt.Errorf("credential publication identity is invalid")
	}
	return nil
}

type CredentialResolver interface {
	Pin(context.Context, CredentialPublication, string, string, string) (string, error)
	ResolvePinned(context.Context, CredentialPublication, string, string, string, string) (Credential, error)
}

type Journal interface {
	BeginDispatch(context.Context, Plan, time.Time) error
	BeginAttempt(context.Context, Plan, Attempt) error
	FinishAttempt(context.Context, Plan, AttemptResult) error
}

type Attempt struct {
	ID        string
	Number    int
	BackendID string
	StartedAt time.Time
}

type AttemptResult struct {
	Attempt
	State           AttemptState
	StatusCode      int
	CompletedAt     time.Time
	ErrorCode       string
	FallbackTrigger FallbackTrigger
}

type CandidateOutcome struct {
	DispatchID         string
	DispatchType       string
	Ordinal            int
	DispatchPlanDigest string
	ModelID            string
	ModelRevision      int64
	Priority           int
	State              AttemptState
	FallbackTrigger    FallbackTrigger
	Attempts           []AttemptResult
}

type Result struct {
	Response *http.Response
	Selected *Plan
	Attempt  AttemptResult
	Outcomes []CandidateOutcome
}

type Transport interface {
	RoundTrip(*http.Request) (*http.Response, error)
}

// ResponseTerminal is the one protocol-neutral completion record emitted for a
// physical attempt. UsageUnavailable is evidence, not zero. Error is populated
// only for a failed terminal and is already safe for public reporting.
type ResponseTerminal struct {
	Usage      llmprotocol.Usage
	StopReason llmprotocol.StopReason
	Error      *llmprotocol.ProtocolError
}

// ResponseFinalizer consumes the semantic terminal produced by the same codec
// engine that decoded the backend response. Accounting must never reparse the
// client-encoded response body to reconstruct this evidence.
type ResponseFinalizer interface {
	Finalize(context.Context, Plan, AttemptResult, ResponseTerminal) error
}

type StreamingResult struct {
	Header     http.Header
	StatusCode int
	Body       io.ReadCloser
	Attempt    AttemptResult
}
