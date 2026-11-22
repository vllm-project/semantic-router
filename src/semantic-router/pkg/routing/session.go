package routing

import (
	"context"
	"fmt"
)

// Processor opens routing sessions. The routing core implements it.
type Processor interface {
	// Open starts one request. ctx carries the request's trace context,
	// deadline and cancellation for the whole session.
	Open(ctx context.Context) (Session, error)
}

// Session processes the phases of one request in order: request headers,
// the request body when there is one, response headers, and the response
// body (once when buffered, once per chunk when streamed). A phase that
// returns an error ends the session. Close must be called exactly once.
type Session interface {
	RequestHeaders(header Header, endOfStream bool) (*Effect, error)
	RequestBody(body []byte, endOfStream bool) (*Effect, error)
	// ResponseHeaders receives the upstream status as the ":status" header.
	ResponseHeaders(header Header, endOfStream bool) (*Effect, error)
	ResponseBody(body []byte, endOfStream bool) (*Effect, error)
	// Evidence reports the routing facts known so far.
	Evidence() Evidence
	// Close ends the request lifecycle. err is nil when the response was
	// delivered, a *LocalReply when the gateway answered for a backend that
	// did not, and another error when the transport ended the request early
	// (client disconnect, deadline).
	Close(err error)
}

// CutBodySession is a Session that answers for a buffered upstream body the
// gateway could not read whole, such as one whose backend closed before the
// declared length. Envoy answers such a response with its own local reply,
// which no phase sees; the engine lets this session answer it instead.
type CutBodySession interface {
	Session
	ResponseBodyCut(err error) (*Effect, error)
}

// LocalReply is the Close error of a request no backend answered: the client
// got the gateway's local reply, which the response phases never see.
type LocalReply struct {
	// Status is the status the client received.
	Status int
	// Failure classifies why no backend answered (for example
	// connect_failure, reset, timeout); empty when unknown.
	Failure string
}

func (e *LocalReply) Error() string {
	return fmt.Sprintf("the gateway answered %d for a failed upstream call (%s)", e.Status, e.Failure)
}

// Evidence is what the routing core decided for a request, for response
// headers, traces and parity records.
type Evidence struct {
	Recipe          string              `json:"recipe,omitempty"`
	Decision        string              `json:"decision,omitempty"`
	Confidence      float64             `json:"confidence,omitempty"`
	Category        string              `json:"category,omitempty"`
	Model           string              `json:"model,omitempty"`
	SelectionMethod string              `json:"selection_method,omitempty"`
	ReasoningMode   string              `json:"reasoning_mode,omitempty"`
	CacheHit        bool                `json:"cache_hit,omitempty"`
	Signals         map[string][]string `json:"signals,omitempty"`
}
