package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
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

func TestLoginUsesUniformFailureAndDummyHashWork(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	active := newTestUser(t, svc, "active@example.com", RoleRead, defaultUserStatusActive)
	inactive := newTestUser(t, svc, "inactive@example.com", RoleRead, "inactive")
	_ = active
	_ = inactive

	tests := []struct {
		name  string
		email string
	}{
		{name: "missing", email: "missing@example.com"},
		{name: "inactive", email: "inactive@example.com"},
		{name: "wrong password", email: "active@example.com"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			calls := 0
			seenHash := ""
			originalVerify := svc.verify
			svc.verify = func(hash, password string) bool {
				calls++
				seenHash = hash
				return originalVerify(hash, password)
			}
			t.Cleanup(func() { svc.verify = originalVerify })

			_, _, err := svc.LoginWithSource(
				context.Background(),
				test.email,
				"definitely-wrong-password",
				"source-"+test.name,
			)
			if !errors.Is(err, ErrInvalidCredentials) {
				t.Fatalf("LoginWithSource() error = %v, want generic invalid credentials", err)
			}
			if calls != 1 {
				t.Fatalf("password verifier calls = %d, want 1", calls)
			}
			if test.name == "missing" && seenHash != dummyPasswordHash {
				t.Fatalf("missing account hash = %q, want dummy hash", seenHash)
			}
		})
	}
}

func TestLoginHandlerReturns429WithRetryAfter(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.limiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 1,
		SourceFailureThreshold:  100,
		BaseDelay:               3 * time.Second,
		MaxDelay:                3 * time.Second,
	})

	request := func() *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(
			http.MethodPost,
			"/api/auth/login",
			strings.NewReader(`{"email":"missing@example.com","password":"wrong-password-value"}`),
		)
		req.Header.Set("Content-Type", "application/json")
		req.RemoteAddr = "192.0.2.10:4567"
		loginHandler(svc).ServeHTTP(recorder, req)
		return recorder
	}

	if first := request(); first.Code != http.StatusUnauthorized {
		t.Fatalf("first status = %d, want 401", first.Code)
	}
	second := request()
	if second.Code != http.StatusTooManyRequests {
		t.Fatalf("second status = %d, want 429", second.Code)
	}
	if second.Header().Get("Retry-After") != "3" {
		t.Fatalf("Retry-After = %q, want 3", second.Header().Get("Retry-After"))
	}
	if !strings.Contains(second.Body.String(), "too many authentication attempts") {
		t.Fatalf("body = %q, want generic rate-limit message", second.Body.String())
	}
}

func TestPasswordVerificationConcurrencyIsBoundedBeforeWorkStarts(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.loginPasswordWork = make(chan struct{}, 1)
	entered := make(chan struct{})
	release := make(chan struct{})
	released := false
	defer func() {
		if !released {
			close(release)
		}
	}()
	firstResult := make(chan error, 1)
	var verifyCalls atomic.Int32
	svc.verify = func(_, _ string) bool {
		if verifyCalls.Add(1) == 1 {
			close(entered)
		}
		<-release
		return false
	}

	go func() {
		_, _, err := svc.LoginWithSource(
			context.Background(),
			"first-missing@example.com",
			"wrong-password-value",
			"source-a",
		)
		firstResult <- err
	}()

	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("first password verification did not start")
	}

	_, _, secondErr := svc.LoginWithSource(
		context.Background(),
		"second-missing@example.com",
		"wrong-password-value",
		"source-b",
	)
	var rateErr *LoginRateLimitError
	if !errors.As(secondErr, &rateErr) {
		t.Fatalf("second error = %v, want LoginRateLimitError", secondErr)
	}
	if got := verifyCalls.Load(); got != 1 {
		t.Fatalf("password verifier calls before release = %d, want 1", got)
	}

	close(release)
	released = true
	select {
	case firstErr := <-firstResult:
		if !errors.Is(firstErr, ErrInvalidCredentials) {
			t.Fatalf("first error = %v, want ErrInvalidCredentials", firstErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("first password verification did not finish")
	}
}

func TestLoginReservationPreventsConcurrentThresholdOvershoot(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.limiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 1,
		SourceFailureThreshold:  100,
		GlobalAttemptBurst:      10,
		BaseDelay:               10 * time.Second,
		MaxDelay:                time.Minute,
	})
	svc.loginPasswordWork = make(chan struct{}, 2)
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseWork := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseWork()
	firstResult := make(chan error, 1)
	var verifyCalls atomic.Int32
	svc.verify = func(_, _ string) bool {
		if verifyCalls.Add(1) == 1 {
			close(entered)
		}
		<-release
		return false
	}
	go func() {
		_, _, err := svc.Login(context.Background(), "same@example.com", "wrong-password")
		firstResult <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("first verification did not start")
	}

	secondResult := make(chan error, 1)
	go func() {
		_, _, err := svc.Login(context.Background(), "same@example.com", "wrong-password")
		secondResult <- err
	}()
	var secondErr error
	select {
	case secondErr = <-secondResult:
	case <-time.After(5 * time.Second):
		releaseWork()
		t.Fatal("second concurrent attempt was not rejected before verification")
	}
	var rateErr *LoginRateLimitError
	if !errors.As(secondErr, &rateErr) {
		t.Fatalf("second error = %v, want atomic rate-limit rejection", secondErr)
	}
	if got := verifyCalls.Load(); got != 1 {
		t.Fatalf("verifier calls = %d, want only the reserved attempt", got)
	}
	releaseWork()
	if firstErr := <-firstResult; !errors.Is(firstErr, ErrInvalidCredentials) {
		t.Fatalf("first error = %v, want ErrInvalidCredentials", firstErr)
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

func TestPasswordWorkPoolsReserveManagementCapacity(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	releases := make([]func(), 0, defaultLoginPasswordWorkConcurrency)
	for range defaultLoginPasswordWorkConcurrency {
		release, ok := acquirePasswordWork(svc.loginPasswordWork)
		if !ok {
			t.Fatal("login pool saturated before its configured capacity")
		}
		releases = append(releases, release)
	}
	defer func() {
		for _, release := range releases {
			release()
		}
	}()
	if _, ok := acquirePasswordWork(svc.loginPasswordWork); ok {
		t.Fatal("login pool exceeded its configured capacity")
	}
	managementRelease, ok := acquirePasswordWork(svc.managementPasswordWork)
	if !ok {
		t.Fatal("login saturation consumed reserved password-management capacity")
	}
	managementRelease()
}
