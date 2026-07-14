package auth

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

const (
	defaultAccountFailureThreshold = 5
	defaultSourceFailureThreshold  = 20
	defaultLoginLimiterCapacity    = 4096
	defaultLoginLimiterTTL         = 15 * time.Minute
	defaultLoginLimiterBaseDelay   = time.Second
	defaultLoginLimiterMaxDelay    = time.Minute
	defaultGlobalAttemptBurst      = 3
	defaultGlobalRefillInterval    = time.Second
)

// LoginLimiterConfig controls the bounded, process-local login limiter. The
// dashboard is deliberately single-replica while its auth store is SQLite.
type LoginLimiterConfig struct {
	AccountFailureThreshold int
	SourceFailureThreshold  int
	MaxAccounts             int
	MaxSources              int
	EntryTTL                time.Duration
	BaseDelay               time.Duration
	MaxDelay                time.Duration
	GlobalAttemptBurst      int
	GlobalRefillInterval    time.Duration
	Now                     func() time.Time
}

type loginAttemptBucket struct {
	failures     int
	inFlight     int
	blockedUntil time.Time
	lastSeen     time.Time
}

type loginAttemptTargetKind uint8

const (
	loginAttemptTargetNone loginAttemptTargetKind = iota
	loginAttemptTargetIndividual
	loginAttemptTargetOverflow
)

type loginAttemptTarget struct {
	kind loginAttemptTargetKind
	key  string
}

// LoginLimiter independently bounds attempts by normalized account and direct
// network peer. A global token budget still applies when the peer is a shared
// private ingress and source bucketing is deliberately disabled.
type LoginLimiter struct {
	mu                 sync.Mutex
	config             LoginLimiterConfig
	accounts           map[string]loginAttemptBucket
	sources            map[string]loginAttemptBucket
	accountOverflow    loginAttemptBucket
	sourceOverflow     loginAttemptBucket
	globalTokens       int
	globalLastRefillAt time.Time
}

func NewLoginLimiter(config LoginLimiterConfig) *LoginLimiter {
	config = withLoginLimiterDefaults(config)
	return &LoginLimiter{
		config:             config,
		accounts:           make(map[string]loginAttemptBucket),
		sources:            make(map[string]loginAttemptBucket),
		globalTokens:       config.GlobalAttemptBurst,
		globalLastRefillAt: config.Now(),
	}
}

func withLoginLimiterDefaults(config LoginLimiterConfig) LoginLimiterConfig {
	if config.AccountFailureThreshold <= 0 {
		config.AccountFailureThreshold = defaultAccountFailureThreshold
	}
	if config.SourceFailureThreshold <= 0 {
		config.SourceFailureThreshold = defaultSourceFailureThreshold
	}
	if config.MaxAccounts <= 0 {
		config.MaxAccounts = defaultLoginLimiterCapacity
	}
	if config.MaxSources <= 0 {
		config.MaxSources = defaultLoginLimiterCapacity
	}
	if config.EntryTTL <= 0 {
		config.EntryTTL = defaultLoginLimiterTTL
	}
	if config.BaseDelay <= 0 {
		config.BaseDelay = defaultLoginLimiterBaseDelay
	}
	if config.MaxDelay <= 0 {
		config.MaxDelay = defaultLoginLimiterMaxDelay
	}
	if config.GlobalAttemptBurst <= 0 {
		config.GlobalAttemptBurst = defaultGlobalAttemptBurst
	}
	if config.GlobalRefillInterval <= 0 {
		config.GlobalRefillInterval = defaultGlobalRefillInterval
	}
	if config.Now == nil {
		config.Now = time.Now
	}
	return config
}

// RetryAfter reports whether a reservation would currently be denied. Service
// code uses Reserve so admission and in-flight accounting remain atomic.
func (l *LoginLimiter) RetryAfter(account, source string) time.Duration {
	if l == nil {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.retryAfterLocked(account, source, l.config.Now())
}

// RecordFailure is retained for sequential callers and tests. Authentication
// paths use Reserve followed by Fail/Succeed/Finish instead.
func (l *LoginLimiter) RecordFailure(account, source string) time.Duration {
	if l == nil {
		return 0
	}
	attempt, retryAfter := l.Reserve(account, source)
	if attempt == nil {
		return retryAfter
	}
	attempt.Fail()
	return l.RetryAfter(account, source)
}

// RecordSuccess clears completed account evidence without erasing source
// spraying evidence or another attempt's in-flight reservation.
func (l *LoginLimiter) RecordSuccess(account string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	key := limiterKey(account)
	bucket, exists := l.accounts[key]
	if !exists {
		return
	}
	bucket.failures = 0
	bucket.blockedUntil = time.Time{}
	if bucket.inFlight == 0 {
		delete(l.accounts, key)
		return
	}
	l.accounts[key] = bucket
}

func (l *LoginLimiter) incrementFailure(
	bucket loginAttemptBucket,
	threshold int,
	now time.Time,
) loginAttemptBucket {
	bucket.failures++
	bucket.lastSeen = now
	if bucket.failures >= threshold {
		delay := exponentialLoginDelay(
			bucket.failures-threshold,
			l.config.BaseDelay,
			l.config.MaxDelay,
		)
		bucket.blockedUntil = now.Add(delay)
	}
	return bucket
}

func (l *LoginLimiter) pruneExpired(now time.Time) {
	for _, buckets := range []map[string]loginAttemptBucket{l.accounts, l.sources} {
		for key, bucket := range buckets {
			if bucket.inFlight == 0 && loginBucketExpired(bucket, now, l.config.EntryTTL) {
				delete(buckets, key)
			}
		}
	}
	if l.accountOverflow.inFlight == 0 && loginBucketExpired(l.accountOverflow, now, l.config.EntryTTL) {
		l.accountOverflow = loginAttemptBucket{}
	}
	if l.sourceOverflow.inFlight == 0 && loginBucketExpired(l.sourceOverflow, now, l.config.EntryTTL) {
		l.sourceOverflow = loginAttemptBucket{}
	}
}

func exponentialLoginDelay(exponent int, base, maximum time.Duration) time.Duration {
	delay := base
	for range exponent {
		if delay >= maximum/2 {
			return maximum
		}
		delay *= 2
	}
	if delay > maximum {
		return maximum
	}
	return delay
}

func bucketRetryAfter(bucket loginAttemptBucket, now time.Time) time.Duration {
	if !bucket.blockedUntil.After(now) {
		return 0
	}
	return bucket.blockedUntil.Sub(now)
}

func loginBucketExpired(bucket loginAttemptBucket, now time.Time, ttl time.Duration) bool {
	return !bucket.lastSeen.IsZero() &&
		now.Sub(bucket.lastSeen) >= ttl &&
		!bucket.blockedUntil.After(now)
}

func limiterKey(raw string) string {
	if raw == "" {
		return ""
	}
	digest := sha256.Sum256([]byte(raw))
	encoded := make([]byte, hex.EncodedLen(len(digest)))
	hex.Encode(encoded, digest[:])
	return string(encoded)
}
