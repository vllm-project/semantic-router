// Package systemonecodec converts System One requests and answers between the
// schema-neutral IR (systemoneprotocol) and the schemas the Router speaks:
// systemone/0.2, the published standard; vllm-sr/5.0, the standard plus the
// vllm-sr extension set that vllm-sr serve speaks; the legacy vllm-sr/4.0;
// and the internal form, the IR's encoding on the hops inside a deployment.
//
// On the client side a SchemaCodec plans the wire calls that carry a batch
// to a target and reads their replies, and a deviation rewrites them for a
// target that claims a schema but does not conform to it. On the server side
// a ListenerCodec reads a client's request into the IR, plans how it goes on
// (its bytes passed through, or the IR translated to the internal form) and
// writes the answers in the client's schema.
//
// It mirrors the Router's LLM protocol codecs (llmprotocol and
// protocolcodec): an immutable registry of stateless codecs that callers
// resolve by ID, so adding a schema or a deviation never touches callers.
package systemonecodec

import (
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// SchemaID names an upstream schema: "<family>/<major>.<minor>".
type SchemaID string

// The schemas this build speaks.
const (
	// SchemaSystemOne02 is the published standard, systemone/0.2.
	SchemaSystemOne02 SchemaID = systemoneprotocol.StandardSchemaID
	// SchemaVLLMSR50 is the standard plus the vllm-sr extension set 5.0, which
	// vllm-sr serve speaks.
	SchemaVLLMSR50 SchemaID = systemoneprotocol.ExtensionSchemaID
	// SchemaVLLMSR40 is the legacy model runtime contract 4.0.0.
	SchemaVLLMSR40 SchemaID = "vllm-sr/4.0"
	// DefaultSchema is the schema of a deployment that names none.
	DefaultSchema = SchemaSystemOne02
)

// DeviationStateText sends an object or array state as its canonical text.
const DeviationStateText = "state_text"

// DefaultMaxConcurrency bounds the calls a split batch has in flight.
const DefaultMaxConcurrency = 8

// SpecRef is the specification a schema follows: its file in the repository,
// the sha256 of that file, where it came from, and its version.
type SpecRef struct {
	ID      SchemaID
	Path    string
	SHA256  string
	Source  string
	Version string
}

// Capability is something a schema carries.
type Capability uint32

const (
	// CapabilityBatch sends several requests in one call.
	CapabilityBatch Capability = 1 << iota
	// CapabilityObjectState sends an object or array state as JSON.
	CapabilityObjectState
	// CapabilityDeadline sends the caller's deadline.
	CapabilityDeadline
	// CapabilityPresets sends a model preset by name.
	CapabilityPresets
	// CapabilitySetQuestions and CapabilitySpan send set and span questions.
	CapabilitySetQuestions
	CapabilitySpan
	// CapabilityFullInput asks for, and reads, proof of full input coverage.
	CapabilityFullInput
	// CapabilityImages and CapabilityVideos send media.
	CapabilityImages
	CapabilityVideos
	// CapabilityCard reads a model card from the listing.
	CapabilityCard
	// CapabilityQuestionErrors names the failed question of a failed call.
	CapabilityQuestionErrors
)

var capabilityNames = []struct {
	capability Capability
	name       string
}{
	{CapabilityBatch, "batch"},
	{CapabilityObjectState, "object_state"},
	{CapabilityDeadline, "deadline"},
	{CapabilityPresets, "presets"},
	{CapabilitySetQuestions, "set"},
	{CapabilitySpan, "span"},
	{CapabilityFullInput, "full_input"},
	{CapabilityImages, "images"},
	{CapabilityVideos, "videos"},
	{CapabilityCard, "card"},
	{CapabilityQuestionErrors, "question_errors"},
}

// CapabilitySet is an immutable set of capabilities.
type CapabilitySet struct{ bits Capability }

// Capabilities is the set of the given capabilities.
func Capabilities(values ...Capability) CapabilitySet {
	var bits Capability
	for _, value := range values {
		bits |= value
	}
	return CapabilitySet{bits: bits}
}

// Has reports whether the set holds every given capability.
func (s CapabilitySet) Has(values ...Capability) bool {
	for _, value := range values {
		if s.bits&value != value {
			return false
		}
	}
	return true
}

// Names are the set's capabilities by name.
func (s CapabilitySet) Names() []string {
	var names []string
	for _, entry := range capabilityNames {
		if s.Has(entry.capability) {
			names = append(names, entry.name)
		}
	}
	return names
}

func (s CapabilitySet) String() string { return strings.Join(s.Names(), ",") }

// Target is the deployment a plan sends to.
type Target struct {
	Deployment string
	Model      string
}

// Policy is how a codec treats what a schema cannot carry.
type Policy struct {
	// StripUnsupported drops an extension field the target's schema strips
	// (with a Diagnostic) instead of refusing the request.
	StripUnsupported bool
	// Presets, when set, gives the canonical question of a model preset, for
	// a schema whose preset questions carry their instructions and criteria.
	Presets func(name string) (systemoneprotocol.Question, bool)
}

// DefaultPolicy strips what the target's schema strips and refuses the rest.
var DefaultPolicy = Policy{StripUnsupported: true}

// DiagnosticAction is what a codec did to a field.
type DiagnosticAction string

const (
	DiagnosticStripped  DiagnosticAction = "stripped"
	DiagnosticConverted DiagnosticAction = "converted"
	DiagnosticSplit     DiagnosticAction = "split"
	DiagnosticRewritten DiagnosticAction = "rewritten"
	// DiagnosticLegacy marks a vllm-sr/4.0 form a listener's client used.
	DiagnosticLegacy DiagnosticAction = "legacy_form"
	// DiagnosticPassedThrough and DiagnosticTranslated are a listener's Plan
	// for a request: its bytes forwarded as the client sent them, or the IR
	// written in the internal form.
	DiagnosticPassedThrough DiagnosticAction = "passed_through"
	DiagnosticTranslated    DiagnosticAction = "translated"
)

// Diagnostic is one change a codec or deviation made to a message.
type Diagnostic struct {
	Field  string
	Action DiagnosticAction
	Reason string
}

// Diagnostics are a plan's changes, for metrics and logs.
type Diagnostics []Diagnostic

// WireCall is one HTTP exchange of a plan: a POST of Body to the
// deployment's endpoint (Path, when set, follows the endpoint's path, as the
// batch endpoint does), carrying the batch items Items.
type WireCall struct {
	Path   string
	Header http.Header
	Body   systemoneprotocol.Value
	Items  []int
	// Batch says the body is a batch whose entries are the items' names.
	Batch bool
}

// Plan is how a codec sends a batch: its calls, which together carry every
// item once, and the items' names.
type Plan struct {
	Calls []WireCall
	Names []string
}

// WireReply is one call's reply.
type WireReply struct {
	Status int
	Header http.Header
	Body   []byte
}

// ItemResult is one batch item's answer: its response, or the error that
// answers it, a *systemoneprotocol.Problem when the server refused it.
type ItemResult struct {
	Response systemoneprotocol.Response
	Err      error
}

// SchemaCodec speaks one upstream schema. A codec is stateless: one value
// serves every request concurrently.
type SchemaCodec interface {
	ID() SchemaID
	Spec() SpecRef
	Capabilities() CapabilitySet
	Stateless() bool
	// Plan sends a batch to a target: one call for the whole batch, or one
	// per item for a schema without batches.
	Plan(systemoneprotocol.Batch, Target, Policy) (Plan, Diagnostics, error)
	// DecodeReply reads one call's reply as the results of the items it
	// carried, in the call's item order.
	DecodeReply(Plan, WireCall, WireReply) ([]ItemResult, error)
	// ListingPath is the path, relative to the endpoint, a GET of which
	// lists the server's models.
	ListingPath() string
	// DecodeListing reads a listing reply.
	DecodeListing(status int, body []byte) (systemoneprotocol.Listing, error)
}

// Deviation rewrites the calls and replies of a target that claims a schema
// but does not conform to it in one declared way. A deviation is stateless.
type Deviation interface {
	ID() string
	AppliesTo(SchemaID) bool
	RewriteRequest(*WireCall) (Diagnostics, error)
	RewriteReply(*WireReply) error
}

// RetryAfter reads a Retry-After header: seconds, or an HTTP date. ok is
// false when the header is absent or unreadable.
func RetryAfter(header http.Header, now time.Time) (time.Duration, bool) {
	value := strings.TrimSpace(header.Get("Retry-After"))
	if value == "" {
		return 0, false
	}
	if seconds, ok := parseSeconds(value); ok {
		return seconds, true
	}
	if when, err := http.ParseTime(value); err == nil {
		if wait := when.Sub(now); wait > 0 {
			return wait, true
		}
		return 0, true
	}
	return 0, false
}

func parseSeconds(value string) (time.Duration, bool) {
	var seconds int64
	for _, c := range value {
		if c < '0' || c > '9' {
			return 0, false
		}
		seconds = seconds*10 + int64(c-'0')
		if seconds > 1<<31 {
			return 0, false
		}
	}
	return time.Duration(seconds) * time.Second, true
}
