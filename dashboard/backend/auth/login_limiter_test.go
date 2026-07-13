package auth

import (
	"testing"
	"time"
)

type loginLimiterTestClock struct {
	now time.Time
}

func (c *loginLimiterTestClock) Now() time.Time { return c.now }
func (c *loginLimiterTestClock) Advance(delta time.Duration) {
	c.now = c.now.Add(delta)
}

func TestLoginLimiterAccountAndSourceBuckets(t *testing.T) {
	t.Parallel()

	clock := &loginLimiterTestClock{now: time.Unix(1000, 0)}
	limiter := NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 2,
		SourceFailureThreshold:  2,
		EntryTTL:                time.Hour,
		BaseDelay:               10 * time.Second,
		MaxDelay:                time.Minute,
		Now:                     clock.Now,
	})

	if retry := limiter.RecordFailure("first@example.com", "source-a"); retry != 0 {
		t.Fatalf("first failure retry = %v, want 0", retry)
	}
	if retry := limiter.RecordFailure("first@example.com", "source-a"); retry != 10*time.Second {
		t.Fatalf("threshold retry = %v, want 10s", retry)
	}
	clock.Advance(4 * time.Second)
	if retry := limiter.RetryAfter("first@example.com", "source-a"); retry != 6*time.Second {
		t.Fatalf("remaining retry = %v, want 6s", retry)
	}

	// A source bucket spans account names and independently catches spraying.
	clock.Advance(7 * time.Second)
	limiter.RecordFailure("second@example.com", "source-b")
	if retry := limiter.RecordFailure("third@example.com", "source-b"); retry != 10*time.Second {
		t.Fatalf("source retry = %v, want 10s", retry)
	}

	limiter.RecordSuccess("first@example.com")
	if retry := limiter.RetryAfter("first@example.com", "source-c"); retry != 0 {
		t.Fatalf("successful account retry = %v, want 0", retry)
	}
	if retry := limiter.RetryAfter("new@example.com", "source-b"); retry == 0 {
		t.Fatal("account success must not erase source spraying evidence")
	}
}

func TestLoginLimiterBoundsAndExpiresState(t *testing.T) {
	t.Parallel()

	clock := &loginLimiterTestClock{now: time.Unix(2000, 0)}
	limiter := NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 100,
		SourceFailureThreshold:  100,
		MaxAccounts:             2,
		MaxSources:              2,
		EntryTTL:                time.Minute,
		BaseDelay:               10 * time.Second,
		MaxDelay:                time.Minute,
		Now:                     clock.Now,
	})

	limiter.RecordFailure("a@example.com", "source-a")
	clock.Advance(time.Second)
	limiter.RecordFailure("b@example.com", "source-b")
	clock.Advance(time.Second)
	if retry := limiter.RecordFailure("c@example.com", "source-c"); retry != 10*time.Second {
		t.Fatalf("overflow retry = %v, want 10s", retry)
	}
	if len(limiter.accounts) != 2 || len(limiter.sources) != 2 {
		t.Fatalf("bounded sizes = accounts %d sources %d, want 2/2", len(limiter.accounts), len(limiter.sources))
	}
	if _, found := limiter.accounts[limiterKey("a@example.com")]; !found {
		t.Fatal("live account evidence was evicted at capacity")
	}
	if _, found := limiter.sources[limiterKey("source-a")]; !found {
		t.Fatal("live source evidence was evicted at capacity")
	}
	if _, found := limiter.accounts[limiterKey("c@example.com")]; found {
		t.Fatal("overflow account unexpectedly consumed an unbounded map entry")
	}
	if _, found := limiter.sources[limiterKey("source-c")]; found {
		t.Fatal("overflow source unexpectedly consumed an unbounded map entry")
	}
	if retry := limiter.RetryAfter("c@example.com", "source-c"); retry != 10*time.Second {
		t.Fatalf("overflow RetryAfter() = %v, want 10s", retry)
	}

	clock.Advance(2 * time.Minute)
	_ = limiter.RetryAfter("unused@example.com", "unused-source")
	if len(limiter.accounts) != 0 || len(limiter.sources) != 0 {
		t.Fatalf("expired sizes = accounts %d sources %d, want 0/0", len(limiter.accounts), len(limiter.sources))
	}
	if !limiter.accountOverflow.lastSeen.IsZero() || !limiter.sourceOverflow.lastSeen.IsZero() {
		t.Fatal("expired overflow evidence was not pruned")
	}
}

func TestLoginLimiterCapacityFloodCannotEraseTargetAccountEvidence(t *testing.T) {
	t.Parallel()

	clock := &loginLimiterTestClock{now: time.Unix(3000, 0)}
	limiter := NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 2,
		SourceFailureThreshold:  100,
		MaxAccounts:             2,
		MaxSources:              100,
		EntryTTL:                time.Hour,
		BaseDelay:               10 * time.Second,
		MaxDelay:                time.Minute,
		Now:                     clock.Now,
	})

	limiter.RecordFailure("target@example.com", "")
	clock.Advance(time.Second)
	limiter.RecordFailure("filler@example.com", "")
	clock.Advance(time.Second)
	if retry := limiter.RecordFailure("spoofed@example.com", ""); retry != 10*time.Second {
		t.Fatalf("overflow retry = %v, want 10s", retry)
	}
	if _, found := limiter.accounts[limiterKey("target@example.com")]; !found {
		t.Fatal("capacity flood evicted target account evidence")
	}

	if retry := limiter.RecordFailure("target@example.com", ""); retry != 10*time.Second {
		t.Fatalf("target second-failure retry = %v, want 10s", retry)
	}
	if retry := limiter.RetryAfter("target@example.com", ""); retry != 10*time.Second {
		t.Fatalf("target RetryAfter() = %v, want preserved threshold backoff", retry)
	}
}

func TestLoginLimiterGlobalAdmissionBudgetsRandomAccounts(t *testing.T) {
	t.Parallel()

	clock := &loginLimiterTestClock{now: time.Unix(4000, 0)}
	limiter := NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 100,
		SourceFailureThreshold:  100,
		GlobalAttemptBurst:      2,
		GlobalRefillInterval:    10 * time.Second,
		Now:                     clock.Now,
	})
	for _, account := range []string{"random-1@example.com", "random-2@example.com"} {
		attempt, retry := limiter.Reserve(account, "")
		if attempt == nil || retry != 0 {
			t.Fatalf("Reserve(%q) = (%#v, %v), want admission", account, attempt, retry)
		}
		attempt.Fail()
	}
	if attempt, retry := limiter.Reserve("random-3@example.com", ""); attempt != nil || retry != 10*time.Second {
		t.Fatalf("third Reserve() = (%#v, %v), want global 10s backoff", attempt, retry)
	}
	clock.Advance(10 * time.Second)
	attempt, retry := limiter.Reserve("random-4@example.com", "")
	if attempt == nil || retry != 0 {
		t.Fatalf("refilled Reserve() = (%#v, %v), want admission", attempt, retry)
	}
	attempt.Cancel()
}
