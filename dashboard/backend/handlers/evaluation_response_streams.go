package handlers

import (
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

const (
	maxEvaluationResponseStreams             = 64
	maxEvaluationResponseStreamsPerPrincipal = 8
	evaluationResponseWriteTimeout           = 30 * time.Second
)

// evaluationResponseStreamLimiter is a transport boundary: verified evidence
// descriptors and SSE subscriptions never consume unbounded HTTP connections,
// and one principal cannot starve unrelated users. Store locks and Service
// operation leases are deliberately outside this network-facing lifetime.
type evaluationResponseStreamLimiter struct {
	mu          sync.Mutex
	total       int
	byPrincipal map[string]int
}

func newEvaluationResponseStreamLimiter() *evaluationResponseStreamLimiter {
	return &evaluationResponseStreamLimiter{byPrincipal: make(map[string]int)}
}

func (limiter *evaluationResponseStreamLimiter) acquire(actor evaluationplane.Actor) (func(), error) {
	principal := actor.PrincipalDigest()
	limiter.mu.Lock()
	defer limiter.mu.Unlock()
	if limiter.total >= maxEvaluationResponseStreams ||
		limiter.byPrincipal[principal] >= maxEvaluationResponseStreamsPerPrincipal {
		return nil, fmt.Errorf("%w: evaluation response stream capacity is exhausted", evaluationplane.ErrConflict)
	}
	limiter.total++
	limiter.byPrincipal[principal]++
	var once sync.Once
	return func() {
		once.Do(func() {
			limiter.mu.Lock()
			defer limiter.mu.Unlock()
			limiter.total--
			limiter.byPrincipal[principal]--
			if limiter.byPrincipal[principal] == 0 {
				delete(limiter.byPrincipal, principal)
			}
		})
	}, nil
}

// evaluationDeadlineWriter refreshes an idle write deadline for every network
// write or flush. It preserves long active downloads while evicting a client
// that stops reading; httptest and non-network writers may report that write
// deadlines are unsupported.
type evaluationDeadlineWriter struct {
	http.ResponseWriter
	controller *http.ResponseController
	timeout    time.Duration
}

func newEvaluationDeadlineWriter(writer http.ResponseWriter, timeout time.Duration) *evaluationDeadlineWriter {
	return &evaluationDeadlineWriter{
		ResponseWriter: writer,
		controller:     http.NewResponseController(writer),
		timeout:        timeout,
	}
}

func (writer *evaluationDeadlineWriter) arm() error {
	err := writer.controller.SetWriteDeadline(time.Now().Add(writer.timeout))
	if errors.Is(err, http.ErrNotSupported) {
		return nil
	}
	return err
}

func (writer *evaluationDeadlineWriter) Write(data []byte) (int, error) {
	if err := writer.arm(); err != nil {
		return 0, err
	}
	return writer.ResponseWriter.Write(data)
}

func (writer *evaluationDeadlineWriter) flush() error {
	if err := writer.arm(); err != nil {
		return err
	}
	return writer.controller.Flush()
}

func (writer *evaluationDeadlineWriter) clearDeadline() {
	_ = writer.controller.SetWriteDeadline(time.Time{})
}
