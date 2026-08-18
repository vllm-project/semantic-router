package handlers

import (
	"strings"
	"sync"
	"time"
)

const (
	modelVerificationRateLimitWindow = time.Minute
	modelVerificationRateLimitCount  = 10
	modelVerificationMaxRateUsers    = 4096
)

type modelVerificationRateWindow struct {
	startedAt time.Time
	count     int
}

type modelVerificationRateLimiter struct {
	mu      sync.Mutex
	windows map[string]modelVerificationRateWindow
	now     func() time.Time
}

func newModelVerificationRateLimiter(now func() time.Time) *modelVerificationRateLimiter {
	if now == nil {
		now = time.Now
	}
	return &modelVerificationRateLimiter{
		windows: make(map[string]modelVerificationRateWindow),
		now:     now,
	}
}

func (limiter *modelVerificationRateLimiter) Allow(identity string) (bool, time.Duration) {
	identity = strings.TrimSpace(identity)
	if identity == "" {
		identity = "unauthenticated"
	}
	now := limiter.now()
	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	for key, window := range limiter.windows {
		if now.Sub(window.startedAt) >= modelVerificationRateLimitWindow {
			delete(limiter.windows, key)
		}
	}
	window, found := limiter.windows[identity]
	if !found {
		if len(limiter.windows) >= modelVerificationMaxRateUsers {
			return false, modelVerificationRateLimitWindow
		}
		limiter.windows[identity] = modelVerificationRateWindow{startedAt: now, count: 1}
		return true, 0
	}
	if window.count >= modelVerificationRateLimitCount {
		retryAfter := modelVerificationRateLimitWindow - now.Sub(window.startedAt)
		if retryAfter < time.Second {
			retryAfter = time.Second
		}
		return false, retryAfter
	}
	window.count++
	limiter.windows[identity] = window
	return true, 0
}
