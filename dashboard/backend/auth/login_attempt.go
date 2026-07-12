package auth

import (
	"sync"
	"time"
)

type loginAttemptOutcome uint8

const (
	loginAttemptCanceled loginAttemptOutcome = iota
	loginAttemptFinished
	loginAttemptFailed
	loginAttemptSucceeded
)

type loginAttemptReservation struct {
	limiter     *LoginLimiter
	account     loginAttemptTarget
	source      loginAttemptTarget
	globalToken bool
	once        sync.Once
}

// Reserve atomically checks all backoff and global admission state, then marks
// this attempt in flight before any bcrypt work can begin.
func (l *LoginLimiter) Reserve(
	account string,
	source string,
) (*loginAttemptReservation, time.Duration) {
	return l.reserve(account, source, true)
}

// ReservePasswordManagement keeps account/source verification atomic but does
// not consume the anonymous login CPU budget. Authenticated users therefore
// retain the dedicated management bcrypt slot needed to rotate a leaked key.
func (l *LoginLimiter) ReservePasswordManagement(
	account string,
	source string,
) (*loginAttemptReservation, time.Duration) {
	return l.reserve(account, source, false)
}

func (l *LoginLimiter) reserve(
	account string,
	source string,
	useGlobalBudget bool,
) (*loginAttemptReservation, time.Duration) {
	if l == nil {
		return &loginAttemptReservation{}, 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	now := l.config.Now()
	l.pruneExpired(now)
	if useGlobalBudget {
		l.refillGlobalTokens(now)
	}
	accountTarget, accountDelay := l.planTarget(
		l.accounts,
		limiterKey(account),
		l.config.MaxAccounts,
		l.accountOverflow,
		l.config.AccountFailureThreshold,
		now,
	)
	sourceTarget, sourceDelay := l.planTarget(
		l.sources,
		limiterKey(source),
		l.config.MaxSources,
		l.sourceOverflow,
		l.config.SourceFailureThreshold,
		now,
	)
	globalDelay := time.Duration(0)
	if useGlobalBudget {
		globalDelay = l.globalRetryAfter(now)
	}
	retryAfter := maxDuration(accountDelay, sourceDelay, globalDelay)
	if retryAfter > 0 {
		return nil, retryAfter
	}

	l.reserveTarget(l.accounts, &l.accountOverflow, accountTarget, now)
	l.reserveTarget(l.sources, &l.sourceOverflow, sourceTarget, now)
	if useGlobalBudget {
		l.globalTokens--
	}
	return &loginAttemptReservation{
		limiter:     l,
		account:     accountTarget,
		source:      sourceTarget,
		globalToken: useGlobalBudget,
	}, 0
}

func (a *loginAttemptReservation) Cancel()  { a.complete(loginAttemptCanceled) }
func (a *loginAttemptReservation) Finish()  { a.complete(loginAttemptFinished) }
func (a *loginAttemptReservation) Fail()    { a.complete(loginAttemptFailed) }
func (a *loginAttemptReservation) Succeed() { a.complete(loginAttemptSucceeded) }

func (a *loginAttemptReservation) complete(outcome loginAttemptOutcome) {
	if a == nil || a.limiter == nil {
		return
	}
	a.once.Do(func() {
		l := a.limiter
		l.mu.Lock()
		defer l.mu.Unlock()
		now := l.config.Now()
		l.finishTarget(
			l.accounts,
			&l.accountOverflow,
			a.account,
			l.config.AccountFailureThreshold,
			outcome,
			true,
			now,
		)
		l.finishTarget(
			l.sources,
			&l.sourceOverflow,
			a.source,
			l.config.SourceFailureThreshold,
			outcome,
			false,
			now,
		)
		if outcome == loginAttemptCanceled && a.globalToken {
			l.refillGlobalTokens(now)
			if l.globalTokens < l.config.GlobalAttemptBurst {
				l.globalTokens++
			}
		}
	})
}

func (l *LoginLimiter) retryAfterLocked(account, source string, now time.Time) time.Duration {
	l.pruneExpired(now)
	l.refillGlobalTokens(now)
	_, accountDelay := l.planTarget(
		l.accounts,
		limiterKey(account),
		l.config.MaxAccounts,
		l.accountOverflow,
		l.config.AccountFailureThreshold,
		now,
	)
	_, sourceDelay := l.planTarget(
		l.sources,
		limiterKey(source),
		l.config.MaxSources,
		l.sourceOverflow,
		l.config.SourceFailureThreshold,
		now,
	)
	return maxDuration(accountDelay, sourceDelay, l.globalRetryAfter(now))
}

func (l *LoginLimiter) planTarget(
	buckets map[string]loginAttemptBucket,
	key string,
	capacity int,
	overflow loginAttemptBucket,
	threshold int,
	now time.Time,
) (loginAttemptTarget, time.Duration) {
	if key == "" {
		return loginAttemptTarget{}, 0
	}
	target := loginAttemptTarget{kind: loginAttemptTargetIndividual, key: key}
	bucket, exists := buckets[key]
	if !exists && len(buckets) >= capacity {
		target.kind = loginAttemptTargetOverflow
		bucket = overflow
		threshold = 1
	}
	if delay := bucketRetryAfter(bucket, now); delay > 0 {
		return target, delay
	}
	if bucket.failures < threshold {
		if bucket.failures+bucket.inFlight >= threshold {
			return target, l.config.BaseDelay
		}
	} else if bucket.inFlight > 0 {
		return target, l.config.BaseDelay
	}
	return target, 0
}

func (l *LoginLimiter) reserveTarget(
	buckets map[string]loginAttemptBucket,
	overflow *loginAttemptBucket,
	target loginAttemptTarget,
	now time.Time,
) {
	switch target.kind {
	case loginAttemptTargetIndividual:
		bucket := buckets[target.key]
		bucket.inFlight++
		bucket.lastSeen = now
		buckets[target.key] = bucket
	case loginAttemptTargetOverflow:
		overflow.inFlight++
		overflow.lastSeen = now
	}
}

func (l *LoginLimiter) finishTarget(
	buckets map[string]loginAttemptBucket,
	overflow *loginAttemptBucket,
	target loginAttemptTarget,
	threshold int,
	outcome loginAttemptOutcome,
	account bool,
	now time.Time,
) {
	if target.kind == loginAttemptTargetNone {
		return
	}
	bucket := *overflow
	if target.kind == loginAttemptTargetIndividual {
		bucket = buckets[target.key]
	} else {
		threshold = 1
	}
	if bucket.inFlight > 0 {
		bucket.inFlight--
	}
	if outcome == loginAttemptFailed {
		bucket = l.incrementFailure(bucket, threshold, now)
	} else {
		bucket.lastSeen = now
	}
	if outcome == loginAttemptSucceeded && account && target.kind == loginAttemptTargetIndividual {
		bucket.failures = 0
		bucket.blockedUntil = time.Time{}
	}
	if target.kind == loginAttemptTargetIndividual {
		if bucket.failures == 0 && bucket.inFlight == 0 && !bucket.blockedUntil.After(now) {
			delete(buckets, target.key)
			return
		}
		buckets[target.key] = bucket
		return
	}
	if bucket.failures == 0 && bucket.inFlight == 0 && !bucket.blockedUntil.After(now) {
		*overflow = loginAttemptBucket{}
		return
	}
	*overflow = bucket
}

func (l *LoginLimiter) refillGlobalTokens(now time.Time) {
	if l.globalLastRefillAt.IsZero() {
		l.globalLastRefillAt = now
		l.globalTokens = l.config.GlobalAttemptBurst
		return
	}
	if !now.After(l.globalLastRefillAt) {
		return
	}
	refills := int(now.Sub(l.globalLastRefillAt) / l.config.GlobalRefillInterval)
	if refills <= 0 {
		return
	}
	l.globalTokens = min(l.config.GlobalAttemptBurst, l.globalTokens+refills)
	l.globalLastRefillAt = l.globalLastRefillAt.Add(
		time.Duration(refills) * l.config.GlobalRefillInterval,
	)
}

func (l *LoginLimiter) globalRetryAfter(now time.Time) time.Duration {
	if l.globalTokens > 0 {
		return 0
	}
	next := l.globalLastRefillAt.Add(l.config.GlobalRefillInterval)
	if !next.After(now) {
		return l.config.GlobalRefillInterval
	}
	return next.Sub(now)
}

func maxDuration(values ...time.Duration) time.Duration {
	var maximum time.Duration
	for _, value := range values {
		if value > maximum {
			maximum = value
		}
	}
	return maximum
}
