package extproc

import (
	"strings"
	"sync"
)

// routerLearningContextualState owns the per-arm matrix state that LinUCB and
// Linear Thompson Sampling need. It lives on the routerLearningRuntime so
// state is shared across requests but isolated by strategy name + arm key.
//
// Key shape: "<strategy>|<decision>|<tier>|<model>". Same fallback ladder as
// modelExperienceKey: per-decision -> per-tier -> per-model.
type routerLearningContextualState struct {
	mu     sync.Mutex
	dim    int
	lambda float64
	arms   map[string]*learningMatrix
}

func newRouterLearningContextualState(dim int, lambda float64) *routerLearningContextualState {
	return &routerLearningContextualState{
		dim:    dim,
		lambda: lambda,
		arms:   map[string]*learningMatrix{},
	}
}

// arm returns the per-arm matrix, lazily initialising it on first use.
// The caller must not retain references after the function returns; the
// state is mutex-protected and we explicitly do not return a snapshot.
func (s *routerLearningContextualState) arm(key string) *learningMatrix {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if existing, ok := s.arms[key]; ok {
		return existing
	}
	m := newLearningMatrix(s.dim, s.lambda)
	s.arms[key] = m
	return m
}

// dimension returns the configured feature dimension.
func (s *routerLearningContextualState) dimension() int {
	if s == nil {
		return 0
	}
	return s.dim
}

func contextualBanditKey(strategy, decisionName string, decisionTier int, model string) string {
	if strings.TrimSpace(decisionName) == "" {
		decisionName = "_global"
	}
	return strategy + "|" + decisionName + "|" + intToString(decisionTier) + "|" + strings.TrimSpace(model)
}

func intToString(i int) string {
	// Avoid pulling strconv into a tight loop; tier values are small integers.
	if i == 0 {
		return "0"
	}
	negative := i < 0
	if negative {
		i = -i
	}
	digits := [20]byte{}
	pos := len(digits)
	for i > 0 {
		pos--
		digits[pos] = byte('0' + i%10)
		i /= 10
	}
	if negative {
		pos--
		digits[pos] = '-'
	}
	return string(digits[pos:])
}
