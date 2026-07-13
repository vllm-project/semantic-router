package auth

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

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
