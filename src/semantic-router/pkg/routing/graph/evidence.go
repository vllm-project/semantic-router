package graph

import (
	"context"
	"errors"
	"time"
)

// Bounds on a run's attempt evidence, the Looper attempt traces' bounds.
const (
	maxAttempts      = 100
	maxEvidenceBytes = 256
)

// AttemptStatus is the bounded terminal state of one hop.
type AttemptStatus string

const (
	AttemptSucceeded AttemptStatus = "succeeded"
	AttemptFailed    AttemptStatus = "failed"
	AttemptCancelled AttemptStatus = "cancelled"
	AttemptTimedOut  AttemptStatus = "timed_out"
)

// Attempt is bounded, content-free evidence of one hop, shaped like a Looper
// attempt trace: no prompt, response or provider error text.
type Attempt struct {
	Ordinal        int           `json:"ordinal"`
	Step           string        `json:"step,omitempty"`
	Model          string        `json:"model,omitempty"`
	Status         AttemptStatus `json:"status"`
	Reason         string        `json:"reason,omitempty"`
	HTTPStatus     int           `json:"http_status,omitempty"`
	Usage          Usage         `json:"usage,omitempty"`
	TotalLatencyMs int64         `json:"total_latency_ms"`
}

type evidence struct {
	attempts []Attempt
	dropped  int
}

func (e *evidence) add(attempt Attempt) {
	if len(e.attempts) == maxAttempts {
		e.dropped++
		return
	}
	e.attempts = append(e.attempts, attempt)
}

func (e *evidence) snapshot() ([]Attempt, int) {
	return append([]Attempt(nil), e.attempts...), e.dropped
}

func (x *Exec) record(ctx context.Context, ordinal int, model string, status int, usage Usage, latency time.Duration, err error) {
	attempt := Attempt{
		Ordinal:        ordinal,
		Step:           bounded(stepOf(ctx)),
		Model:          bounded(model),
		HTTPStatus:     status,
		Usage:          usage,
		TotalLatencyMs: latency.Milliseconds(),
	}
	switch {
	case err == nil && status >= 200 && status < 300:
		attempt.Status = AttemptSucceeded
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, ErrDeadline):
		attempt.Status, attempt.Reason = AttemptTimedOut, "deadline_exceeded"
	case errors.Is(err, context.Canceled):
		attempt.Status, attempt.Reason = AttemptCancelled, "request_cancelled"
	case err != nil:
		attempt.Status, attempt.Reason = AttemptFailed, errorReason(err)
	default:
		attempt.Status, attempt.Reason = AttemptFailed, "upstream_error"
	}
	x.mu.Lock()
	x.evidence.add(attempt)
	x.mu.Unlock()
}

// bounded cuts a label to the evidence bound on a UTF-8 boundary.
func bounded(value string) string {
	if len(value) <= maxEvidenceBytes {
		return value
	}
	cut := maxEvidenceBytes
	for cut > 0 && value[cut]&0xC0 == 0x80 {
		cut--
	}
	return value[:cut]
}
