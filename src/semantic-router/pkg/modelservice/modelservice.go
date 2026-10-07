// Package modelservice connects the Router to the built-in model runtime
// (src/model-runtime): the client generated from the runtime's OpenAPI
// contract, transports over Unix sockets and TCP, the supervisor of
// Router-managed runtime processes, and per-deployment readiness so callers
// fail open without waiting on a runtime that is not ready.
package modelservice

import (
	"context"
	"errors"
	"os"
	"time"
)

// ReadyTimeoutEnv bounds, as a Go duration, how long preparing a binding waits
// for its deployment to become ready when the caller sets no deadline.
const ReadyTimeoutEnv = "VLLM_SRUN_READY_TIMEOUT"

const defaultReadyTimeout = 10 * time.Minute

// ReadyTimeout is the preparation wait from ReadyTimeoutEnv (default 10 minutes:
// a first start may download and verify the models).
func ReadyTimeout() time.Duration {
	if value, err := time.ParseDuration(os.Getenv(ReadyTimeoutEnv)); err == nil && value > 0 {
		return value
	}
	return defaultReadyTimeout
}

// Question is one typed question; Choices keep Choice / Noul option order,
// Levels keep Score level order and Labels keep Set / Span label order.
// Threshold and Head apply to Set and Span questions. Preset names a question
// the model defines, which replaces the type, instructions and options.
type Question struct {
	ID           string
	Type         string
	Instructions string
	Choices      []Choice
	Levels       []string
	Labels       []Choice
	Threshold    *float64
	Head         string
	Preset       string
}

// Choice is one Choice option, a Noul description or a Set / Span label.
type Choice struct {
	Key         string
	Description string
}

// Request asks one deployment several questions about one state. Model is
// the served model on the runtime (empty while it serves one model). State
// is the text the questions are about; Parts, when set, replace it with
// named parts (request, context, answer) for models that read typed parts.
type Request struct {
	Model     string
	State     string
	Parts     map[string]string
	Questions []Question
}

// Answer is one runtime answer. Error is set instead of the values when the
// runtime could not answer this question. A Set answer holds each label's
// probability in Probabilities and the labels above Threshold in Selected. A
// Span answer holds the labelled Spans above Threshold and, in Noul, the
// highest probability any word of the text reached; Head names the span head
// that answered when the model reports it.
type Answer struct {
	Type          string
	Choice        string
	Noul          float64
	Score         float64
	Probabilities map[string]float64
	Confidence    float64
	Selected      []string
	Spans         []Span
	Threshold     float64
	Head          string
	Error         string
}

// Response holds the answers by question ID.
type Response struct {
	Model       string
	Answers     map[string]Answer
	InputTokens int
}

// Decider answers questions through a named model_runtime deployment.
type Decider interface {
	Decide(ctx context.Context, deployment string, request Request) (Response, error)
}

var (
	// ErrUnknownDeployment means the configuration declares no such model_runtime deployment.
	ErrUnknownDeployment = errors.New("model runtime deployment is not configured")
	// ErrUnavailable means the runtime is starting, restarting or failed; callers fail open at once.
	ErrUnavailable = errors.New("model runtime is not ready")
	// ErrOverloaded means the runtime refused the request at admission.
	ErrOverloaded = errors.New("model runtime is overloaded")
	// ErrRejected means the runtime rejected the request as invalid.
	ErrRejected = errors.New("model runtime rejected the request")
	// ErrFailed covers transport failures and unexpected responses.
	ErrFailed = errors.New("model runtime call failed")
)

// ErrorReason is a stable, low-cardinality label for an error.
func ErrorReason(err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, ErrUnknownDeployment):
		return "unknown_deployment"
	case errors.Is(err, ErrUnavailable):
		return "unavailable"
	case errors.Is(err, ErrOverloaded):
		return "overloaded"
	case errors.Is(err, ErrRejected):
		return "rejected"
	default:
		return "failed"
	}
}
