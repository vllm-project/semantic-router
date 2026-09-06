//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

const (
	learningOutcomeIdempotencyHeader = "Idempotency-Key"
	learningOutcomeRateLimit         = 60
	learningOutcomeRateWindow        = time.Minute
)

type learningOutcomePolicyError struct {
	Status  int
	Code    string
	Message string
}

// learningOutcomeIngestPolicy enforces request-level learning ingest rules:
// required Idempotency-Key, per-principal rate limit, and principal-derived source.
type learningOutcomeIngestPolicy struct {
	mu     sync.Mutex
	limit  int
	window time.Duration
	hits   map[string][]time.Time
}

func newLearningOutcomeIngestPolicy() *learningOutcomeIngestPolicy {
	return &learningOutcomeIngestPolicy{
		limit:  learningOutcomeRateLimit,
		window: learningOutcomeRateWindow,
		hits:   map[string][]time.Time{},
	}
}

func (s *ClassificationAPIServer) learningOutcomeIngestPolicy() *learningOutcomeIngestPolicy {
	if s == nil {
		return newLearningOutcomeIngestPolicy()
	}
	s.learningOutcomePolicyOnce.Do(func() {
		s.learningOutcomePolicy = newLearningOutcomeIngestPolicy()
	})
	return s.learningOutcomePolicy
}

func (p *learningOutcomeIngestPolicy) enforce(
	r *http.Request,
	principal managementPrincipal,
) (idempotencyKey string, source routerruntime.RouterOutcomeSource, policyErr *learningOutcomePolicyError) {
	if p == nil {
		p = newLearningOutcomeIngestPolicy()
	}
	key := strings.TrimSpace(r.Header.Get(learningOutcomeIdempotencyHeader))
	if key == "" {
		return "", "", &learningOutcomePolicyError{
			Status:  http.StatusBadRequest,
			Code:    "MISSING_IDEMPOTENCY_KEY",
			Message: "Idempotency-Key header is required for learning outcome ingestion",
		}
	}
	if len(key) > 256 {
		return "", "", &learningOutcomePolicyError{
			Status:  http.StatusBadRequest,
			Code:    "INVALID_IDEMPOTENCY_KEY",
			Message: "Idempotency-Key must be at most 256 characters",
		}
	}
	if !p.allow(principalKey(principal)) {
		return "", "", &learningOutcomePolicyError{
			Status:  http.StatusTooManyRequests,
			Code:    "RATE_LIMITED",
			Message: "learning outcome ingestion rate limit exceeded",
		}
	}
	return key, sourceFromManagementPrincipal(principal), nil
}

func (p *learningOutcomeIngestPolicy) allow(bucket string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	now := time.Now()
	cutoff := now.Add(-p.window)
	hits := p.hits[bucket]
	kept := hits[:0]
	for _, ts := range hits {
		if ts.After(cutoff) {
			kept = append(kept, ts)
		}
	}
	if len(kept) >= p.limit {
		p.hits[bucket] = kept
		return false
	}
	p.hits[bucket] = append(kept, now)
	return true
}

func principalKey(principal managementPrincipal) string {
	role := strings.TrimSpace(principal.Role)
	if role == "" {
		role = "anonymous"
	}
	if principal.AuthEnabled {
		return "auth:" + role
	}
	return "local:" + role
}

// sourceFromManagementPrincipal derives learning provenance from auth credentials.
// Management-API ingest is always attributed as operator (never caller-supplied).
func sourceFromManagementPrincipal(_ managementPrincipal) routerruntime.RouterOutcomeSource {
	return routerruntime.RouterOutcomeSourceOperator
}
