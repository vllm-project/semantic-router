package managementauth

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

const (
	testSessionID   = "11111111-1111-4111-8111-111111111111"
	testPrincipalID = "22222222-2222-4222-8222-222222222222"
	testSourceID    = "33333333-3333-4333-8333-333333333333"
	testNamespaceID = "44444444-4444-4444-8444-444444444444"
)

type memorySessionRepository struct {
	mu         sync.Mutex
	session    LiveSession
	getErr     error
	refreshErr error
	revokeErr  error
}

func (r *memorySessionRepository) Create(context.Context, SessionDraft) (LiveSession, error) {
	return LiveSession{}, errors.New("not implemented in runtime fake")
}

func (r *memorySessionRepository) Get(context.Context, string) (LiveSession, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.session, r.getErr
}

func (r *memorySessionRepository) RotateTokenID(_ context.Context, id, expected, replacement string) (LiveSession, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.refreshErr != nil {
		return LiveSession{}, r.refreshErr
	}
	if r.session.ID != id || r.session.TokenID != expected || r.session.Status != SessionActive {
		return LiveSession{}, ErrSessionConflict
	}
	r.session.TokenID = replacement
	return r.session, nil
}

func (r *memorySessionRepository) Revoke(_ context.Context, id, expected string) (SessionMutation, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.revokeErr != nil {
		return SessionMutation{}, r.revokeErr
	}
	if r.session.ID != id || r.session.TokenID != expected {
		return SessionMutation{}, ErrSessionConflict
	}
	changed := r.session.Status == SessionActive
	when := time.Date(2026, 8, 22, 2, 0, 0, 0, time.UTC)
	if changed {
		r.session.Status = SessionRevoked
		r.session.RevokedAt = &when
	}
	return SessionMutation{SessionID: id, TokenID: expected, Changed: changed, ChangedAt: when}, nil
}

type memoryBarriers struct {
	mu         sync.Mutex
	state      BarrierState
	checkErr   error
	installErr error
	checks     []BarrierCheck
	installed  []string
}

type memorySessionPolicyLoader struct {
	mu     sync.Mutex
	policy SessionPolicy
	err    error
}

func (loader *memorySessionPolicyLoader) LoadSessionPolicy(context.Context) (SessionPolicy, error) {
	loader.mu.Lock()
	defer loader.mu.Unlock()
	return loader.policy, loader.err
}

func testSessionPolicyLoader(accessTTL time.Duration) *memorySessionPolicyLoader {
	return &memorySessionPolicyLoader{policy: SessionPolicy{
		AccessTokenTTL: accessTTL, SessionTTL: 8 * time.Hour, MaxActiveSessions: 5,
		ActionRequirements: map[string]ActionRequirement{
			"cluster_sensitive": {AnyOf: []AuthenticationRequirement{{
				Kind:  RequirementHuman,
				Human: &HumanRequirement{MinimumAAL: "aal2", MaxAuthenticationAgeSeconds: 900},
			}}},
		},
		SeedVersion: SupportedSessionPolicySeedVersion, Revision: 1,
		UpdatedAt: time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC),
	}}
}

func (b *memoryBarriers) Check(_ context.Context, check BarrierCheck) (BarrierState, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.checks = append(b.checks, check)
	return b.state, b.checkErr
}

func (b *memoryBarriers) InstallDeny(_ context.Context, kind BarrierKind, id string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.installed = append(b.installed, string(kind)+":"+id)
	return b.installErr
}

func validLiveSession(now time.Time) LiveSession {
	authenticatedAt := now.Add(-5 * time.Minute).Truncate(time.Second)
	return LiveSession{
		Session: Session{
			ID: testSessionID, PrincipalID: testPrincipalID, TokenID: "token-current",
			Audience: "vllm-sr-management", AuthSourceKind: AuthSourceIssuer,
			AuthSourceID: testSourceID, EvidenceKind: EvidenceHuman,
			Human:           &HumanEvidence{AuthenticationTime: authenticatedAt.Unix(), AAL: "aal2", AMR: []string{"pwd", "otp"}},
			AuthenticatedAt: authenticatedAt, ExpiresAt: now.Add(8 * time.Hour),
			Status: SessionActive, CreatedAt: now.Add(-time.Minute),
		},
		PrincipalStatus: ResourceActive, AuthSourceStatus: ResourceActive,
	}
}

func TestSessionRuntimeIssueAndAuthenticate(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	repository := &memorySessionRepository{session: validLiveSession(now)}
	barriers := &memoryBarriers{state: BarrierState{Ready: true}}
	runtime := SessionRuntime{
		Codec: testTokenCodec(t), Sessions: repository, Barriers: barriers,
		PolicyLoader: testSessionPolicyLoader(15 * time.Minute),
	}
	issued, err := runtime.Issue(context.Background(), testSessionID, now)
	if err != nil {
		t.Fatalf("Issue() error = %v", err)
	}
	if issued.TokenType != "Bearer" || issued.ExpiresIn != 15*time.Minute || issued.ManagementSessionID != testSessionID {
		t.Fatalf("Issue() = %+v", issued)
	}
	authenticated, err := runtime.Authenticate(context.Background(), issued.AccessToken, testNamespaceID, now.Add(time.Minute))
	if err != nil {
		t.Fatalf("Authenticate() error = %v", err)
	}
	if authenticated.Session.ID != testSessionID || authenticated.NamespaceID != testNamespaceID {
		t.Fatalf("Authenticate() = %+v", authenticated)
	}
	barriers.mu.Lock()
	last := barriers.checks[len(barriers.checks)-1]
	barriers.mu.Unlock()
	if last.NamespaceID != testNamespaceID || last.PrincipalID != testPrincipalID {
		t.Fatalf("barrier check = %+v", last)
	}
}

func TestSessionRuntimeAppliesCurrentPolicyToExistingAndNewTokens(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	repository := &memorySessionRepository{session: validLiveSession(now)}
	policy := testSessionPolicyLoader(15 * time.Minute)
	runtime := SessionRuntime{
		Codec: testTokenCodec(t), Sessions: repository,
		Barriers: &memoryBarriers{state: BarrierState{Ready: true}}, PolicyLoader: policy,
	}
	oldToken, issueErr := runtime.Issue(context.Background(), testSessionID, now)
	if issueErr != nil {
		t.Fatal(issueErr)
	}
	policy.mu.Lock()
	policy.policy.AccessTokenTTL = 5 * time.Minute
	policy.policy.Revision++
	policy.policy.UpdatedAt = now.Add(time.Minute)
	policy.mu.Unlock()
	if _, err := runtime.Authenticate(context.Background(), oldToken.AccessToken, "", now.Add(time.Minute)); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("old token under tightened policy error = %v, want denied", err)
	}
	newToken, issueErr := runtime.Issue(context.Background(), testSessionID, now.Add(time.Minute))
	if issueErr != nil {
		t.Fatal(issueErr)
	}
	if newToken.ExpiresIn != 5*time.Minute {
		t.Fatalf("new token TTL = %s, want 5m", newToken.ExpiresIn)
	}
}

func TestSessionRuntimeCapsTokenAtSessionExpiry(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	session := validLiveSession(now)
	session.ExpiresAt = now.Add(4 * time.Minute)
	runtime := SessionRuntime{
		Codec: testTokenCodec(t), Sessions: &memorySessionRepository{session: session},
		Barriers: &memoryBarriers{state: BarrierState{Ready: true}}, PolicyLoader: testSessionPolicyLoader(15 * time.Minute),
	}
	issued, err := runtime.Issue(context.Background(), testSessionID, now)
	if err != nil {
		t.Fatal(err)
	}
	if issued.ExpiresIn != 4*time.Minute {
		t.Fatalf("ExpiresIn = %s", issued.ExpiresIn)
	}
}

func TestSessionRuntimeFailsClosed(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	repository := &memorySessionRepository{session: validLiveSession(now)}
	barriers := &memoryBarriers{state: BarrierState{Ready: true}}
	runtime := SessionRuntime{
		Codec: testTokenCodec(t), Sessions: repository, Barriers: barriers,
		PolicyLoader: testSessionPolicyLoader(15 * time.Minute),
	}
	issued, err := runtime.Issue(context.Background(), testSessionID, now)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name     string
		mutate   func()
		expected error
	}{
		{"missing session", func() { repository.getErr = ErrSessionNotFound }, ErrAuthenticationDenied},
		{"database unavailable", func() { repository.getErr = errors.New("database unavailable") }, ErrAuthenticationUnavailable},
		{"barrier unavailable", func() { barriers.checkErr = errors.New("valkey unavailable") }, ErrAuthenticationUnavailable},
		{"barrier not ready", func() { barriers.state.Ready = false }, ErrAuthenticationUnavailable},
		{"session denied", func() { barriers.state = BarrierState{Ready: true, SessionDenied: true} }, ErrAuthenticationDenied},
		{"namespace denied", func() { barriers.state = BarrierState{Ready: true, NamespaceDenied: true} }, ErrAuthenticationDenied},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			repository.getErr = nil
			barriers.checkErr = nil
			barriers.state = BarrierState{Ready: true}
			test.mutate()
			_, err := runtime.Authenticate(context.Background(), issued.AccessToken, testNamespaceID, now.Add(time.Minute))
			if !errors.Is(err, test.expected) {
				t.Fatalf("Authenticate() error = %v, want %v", err, test.expected)
			}
		})
	}
}

func TestSessionRuntimeRejectsCurrentStateMismatch(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	repository := &memorySessionRepository{session: validLiveSession(now)}
	runtime := SessionRuntime{
		Codec: testTokenCodec(t), Sessions: repository,
		Barriers: &memoryBarriers{state: BarrierState{Ready: true}}, PolicyLoader: testSessionPolicyLoader(15 * time.Minute),
	}
	issued, err := runtime.Issue(context.Background(), testSessionID, now)
	if err != nil {
		t.Fatal(err)
	}
	repository.session.TokenID = "different-current-jti"
	if _, err := runtime.Authenticate(context.Background(), issued.AccessToken, "", now.Add(time.Minute)); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("Authenticate() error = %v", err)
	}
}

func TestSessionRuntimeReissueRequiresFreshExactSourceAndInvalidatesOldToken(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	repository := &memorySessionRepository{session: validLiveSession(now)}
	barriers := &memoryBarriers{state: BarrierState{Ready: true}}
	runtime := SessionRuntime{
		Codec: testTokenCodec(t), Sessions: repository, Barriers: barriers,
		PolicyLoader: testSessionPolicyLoader(15 * time.Minute),
		NewTokenID:   func() (string, error) { return "55555555-5555-4555-8555-555555555555", nil },
	}
	first, err := runtime.Issue(context.Background(), testSessionID, now)
	if err != nil {
		t.Fatal(err)
	}
	verified := VerifiedSessionSource{
		PrincipalID: testPrincipalID, AuthSourceKind: AuthSourceIssuer, AuthSourceID: testSourceID,
		EvidenceKind: EvidenceHuman,
		Human: &HumanEvidence{
			AuthenticationTime: repository.session.Human.AuthenticationTime,
			AAL:                repository.session.Human.AAL, AMR: append([]string(nil), repository.session.Human.AMR...),
		},
		AuthenticatedAt:   repository.session.AuthenticatedAt,
		EvidenceExpiresAt: repository.session.ExpiresAt,
	}
	refreshed, err := runtime.ReissueVerified(context.Background(), testSessionID, verified, now.Add(time.Minute))
	if err != nil {
		t.Fatalf("ReissueVerified() error = %v", err)
	}
	if _, err := runtime.Authenticate(context.Background(), first.AccessToken, "", now.Add(2*time.Minute)); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("old Authenticate() error = %v", err)
	}
	if _, err := runtime.Authenticate(context.Background(), refreshed.AccessToken, "", now.Add(2*time.Minute)); err != nil {
		t.Fatalf("refreshed Authenticate() error = %v", err)
	}
	verified.Human.AAL = "aal3"
	if _, err := runtime.ReissueVerified(context.Background(), testSessionID, verified, now.Add(3*time.Minute)); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("changed evidence error = %v, want ErrAuthenticationDenied", err)
	}
}

func TestSessionRuntimeRevokeRequiresBarrier(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	repository := &memorySessionRepository{session: validLiveSession(now)}
	barriers := &memoryBarriers{state: BarrierState{Ready: true}}
	runtime := SessionRuntime{
		Codec: testTokenCodec(t), Sessions: repository, Barriers: barriers,
		PolicyLoader: testSessionPolicyLoader(15 * time.Minute),
	}
	mutation, err := runtime.Revoke(context.Background(), testSessionID, "token-current")
	if err != nil || !mutation.Changed {
		t.Fatalf("Revoke() = %+v, %v", mutation, err)
	}
	if len(barriers.installed) != 1 || barriers.installed[0] != "management_session:"+testSessionID {
		t.Fatalf("installed barriers = %v", barriers.installed)
	}

	repository.session.Status = SessionActive
	repository.session.RevokedAt = nil
	barriers.installErr = errors.New("quorum unavailable")
	if _, err := runtime.Revoke(context.Background(), testSessionID, "token-current"); !errors.Is(err, ErrAuthenticationUnavailable) {
		t.Fatalf("Revoke() barrier error = %v", err)
	}
}

func TestLiveSessionRejectsMixedEvidenceAndStaleWorkloadSource(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	mixed := validLiveSession(now)
	mixed.Workload = &WorkloadEvidence{Class: "workload_strong", SourceAssuredAt: now.Add(-time.Hour).Unix()}
	if err := mixed.ValidateAt(now); err == nil {
		t.Fatal("ValidateAt() accepted mixed evidence")
	}

	assured := now.Add(-time.Hour).Truncate(time.Second)
	other := assured.Add(-time.Second)
	workload := LiveSession{
		Session: Session{
			ID: testSessionID, PrincipalID: testPrincipalID, TokenID: "token-workload",
			Audience: "vllm-sr-management", AuthSourceKind: AuthSourceServiceCredential,
			AuthSourceID: testSourceID, EvidenceKind: EvidenceWorkload,
			Workload:        &WorkloadEvidence{Class: "workload_strong", SourceAssuredAt: assured.Unix()},
			AuthenticatedAt: now.Add(-time.Minute), ExpiresAt: now.Add(time.Hour),
			Status: SessionActive, CreatedAt: now.Add(-time.Minute),
		},
		PrincipalStatus: ResourceActive, AuthSourceStatus: ResourceActive,
		AuthSourceAssuredAt: &other,
	}
	if !errors.Is(workload.ValidateAt(now), ErrSessionInactive) {
		t.Fatalf("ValidateAt() error = %v", workload.ValidateAt(now))
	}
}
