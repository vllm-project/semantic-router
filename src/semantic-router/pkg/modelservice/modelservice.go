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
// Truncate asks a model with a scan budget to read a long part's first tokens
// only (one forward on a CPU) instead of whole.
// RequireFullInput rejects a judgment unless the runtime proves every supplied
// input part was read completely; it is independent of scan-budget support.
// DecisionAdapter asks a vela3_d3 package with an opt-in decision adapter to
// answer with it; only that family takes the field.
type Question struct {
	// TaskID and Stage describe the internal consumer, and Reads the
	// options its askers read (nil: every option), which a family's trained
	// form must cover to answer it; they are not wire fields.
	TaskID           string
	Stage            string
	Reads            []string
	ID               string
	Type             string
	Instructions     string
	Choices          []Choice
	Levels           []string
	Labels           []Choice
	Threshold        *float64
	Head             string
	Preset           string
	Truncate         bool
	RequireFullInput bool
	DecisionAdapter  bool
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
// Images and Videos are base64 data URLs every question of the state sees,
// for a model whose card lists the image or video modality.
// MaxTokens, when positive, is the scan budget of a model that reads a long
// part in windows (Vela 2.0), zero keeping the model's own: a part with more
// tokens fails the questions that read it whole with scan_budget_exceeded. A
// model without a scan budget ignores it and every question's Truncate.
// Window, when positive, makes the state a window of a guard's scan and is
// its rank in reading order. In a stage's bundle a window travels in a call
// of its own, with the other calls about the same window, so its answers
// come back as soon as it is read; at most Parallel of the stage's windows
// to one deployment are in flight (DefaultWindowParallel when 0), the lowest
// rank first. Neither is a wire field, and the answers stay as they are.
type Request struct {
	Model     string
	State     string
	Parts     map[string]string
	Images    []string
	Videos    []string
	Questions []Question
	MaxTokens int
	Window    int
	Parallel  int
}

// DefaultWindowParallel is how many windows of a stage's guard scans one
// deployment reads at a time unless the request says otherwise.
const DefaultWindowParallel = 4

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
	InputCoverage string
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
	// ErrSignalsAbandoned is the cause a caller cancels a request's signals
	// with once it has answered the request without them, as when the
	// request is refused for its length. The calls it cuts short are not
	// failures of their deployments, and are neither counted nor logged.
	ErrSignalsAbandoned = errors.New("the request no longer needs its signals")
)

// SignalsAbandoned reports a context its caller cancelled with
// ErrSignalsAbandoned.
func SignalsAbandoned(ctx context.Context) bool {
	return ctx != nil && errors.Is(context.Cause(ctx), ErrSignalsAbandoned)
}

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
