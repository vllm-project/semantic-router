package auth

import (
	"context"
	"errors"
	"testing"
	"time"
)

type recordingFirstAdminProvisioner struct {
	identities []FirstAdminIdentity
	failures   int
}

func (provisioner *recordingFirstAdminProvisioner) ProvisionFirstAdmin(
	_ context.Context,
	identity FirstAdminIdentity,
) error {
	provisioner.identities = append(provisioner.identities, identity)
	if provisioner.failures > 0 {
		provisioner.failures--
		return errors.New("temporary Router failure")
	}
	return nil
}

func TestManagedBootstrapPersistsOneIdentityAndResumesBeforeLogin(t *testing.T) {
	svc := newTestAuthService(t)
	now := time.Date(2026, time.August, 24, 10, 0, 0, 0, time.UTC)
	svc.now = func() time.Time { return now }
	provisioner := &recordingFirstAdminProvisioner{failures: 1}
	svc.ConfigureFirstAdminProvisioner(provisioner)

	firstHash, err := svc.HashPassword("secret-password")
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = svc.BootstrapRegister(
		context.Background(), "Admin@Example.test", "Admin", "secret-password", firstHash,
	)
	if !errors.Is(err, ErrFirstAdminProvisioningUnavailable) {
		t.Fatalf("first BootstrapRegister() error = %v", err)
	}
	var sessionCount int
	if queryErr := svc.store.db.QueryRow(`SELECT COUNT(*) FROM auth_sessions`).Scan(&sessionCount); queryErr != nil {
		t.Fatal(queryErr)
	}
	if sessionCount != 0 {
		t.Fatalf("failed provisioning created %d active browser sessions", sessionCount)
	}
	if can, canErr := svc.CanBootstrap(context.Background()); canErr != nil || !can {
		t.Fatalf("CanBootstrap() = %v, %v; want resumable", can, canErr)
	}
	if _, _, loginErr := svc.Login(context.Background(), "admin@example.test", "secret-password"); loginErr == nil {
		t.Fatal("Login() succeeded before Router provisioning completed")
	}

	retryHash, err := svc.HashPassword("secret-password")
	if err != nil {
		t.Fatal(err)
	}
	user, token, err := svc.BootstrapRegister(
		context.Background(), "admin@example.test", "Admin", "secret-password", retryHash,
	)
	if err != nil {
		t.Fatalf("retry BootstrapRegister() error = %v", err)
	}
	if user.Status != defaultUserStatusActive {
		t.Fatalf("user status = %q, want active", user.Status)
	}
	claims, err := svc.ParseToken(token)
	if err != nil || claims.ID != provisioner.identities[1].SessionID || claims.ExpiresAt == nil ||
		claims.IssuedAt == nil ||
		!claims.IssuedAt.Time.UTC().Equal(provisioner.identities[1].AuthenticatedAt) ||
		!claims.ExpiresAt.Time.UTC().Equal(provisioner.identities[1].ExpiresAt) {
		t.Fatalf("bootstrap token claims=%#v identity=%#v error=%v", claims, provisioner.identities[1], err)
	}
	if len(provisioner.identities) != 2 ||
		provisioner.identities[0].UserID != provisioner.identities[1].UserID ||
		provisioner.identities[0].SessionID != provisioner.identities[1].SessionID ||
		!provisioner.identities[0].AuthenticatedAt.Equal(now) ||
		!provisioner.identities[1].AuthenticatedAt.Equal(now) ||
		!provisioner.identities[0].ExpiresAt.Equal(now.Add(time.Hour)) ||
		!provisioner.identities[1].ExpiresAt.Equal(now.Add(time.Hour)) {
		t.Fatalf("provisioned identities = %#v; want stable retry identity", provisioner.identities)
	}
	var storedSession localSessionDraft
	if err := svc.store.db.QueryRow(
		`SELECT id,issued_at,expires_at FROM auth_sessions WHERE user_id=?`, user.ID,
	).Scan(&storedSession.ID, &storedSession.IssuedAt, &storedSession.ExpiresAt); err != nil {
		t.Fatal(err)
	}
	if storedSession.ID != claims.ID || storedSession.IssuedAt != claims.IssuedAt.Time.Unix() ||
		storedSession.ExpiresAt != claims.ExpiresAt.Time.Unix() {
		t.Fatalf("stored session=%#v claims=%#v", storedSession, claims)
	}
	if can, canErr := svc.CanBootstrap(context.Background()); canErr != nil || can {
		t.Fatalf("CanBootstrap() = %v, %v after completion; want closed", can, canErr)
	}
	if _, loggedIn, err := svc.Login(context.Background(), "admin@example.test", "secret-password"); err != nil || loggedIn.ID != user.ID {
		t.Fatalf("Login() user=%#v error=%v", loggedIn, err)
	}
}

func TestManagedBootstrapRotatesOnlyExpiredDurableSourceSession(t *testing.T) {
	svc := newTestAuthService(t)
	now := time.Date(2026, time.August, 24, 10, 0, 0, 0, time.UTC)
	svc.now = func() time.Time { return now }
	provisioner := &recordingFirstAdminProvisioner{failures: 1}
	svc.ConfigureFirstAdminProvisioner(provisioner)
	hash, err := svc.HashPassword("secret-password")
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = svc.BootstrapRegister(
		context.Background(), "admin@example.test", "Admin", "secret-password", hash,
	)
	if !errors.Is(err, ErrFirstAdminProvisioningUnavailable) {
		t.Fatalf("first BootstrapRegister() error = %v", err)
	}
	first := provisioner.identities[0]
	var storedExpiry int64
	if queryErr := svc.store.db.QueryRow(
		`SELECT source_expires_at FROM dashboard_bootstrap_installation WHERE singleton=1`,
	).Scan(&storedExpiry); queryErr != nil {
		t.Fatal(queryErr)
	}
	if storedExpiry != first.ExpiresAt.Unix() {
		t.Fatalf("stored source expiry = %d, want %d", storedExpiry, first.ExpiresAt.Unix())
	}

	now = now.Add(2 * time.Hour)
	retryHash, err := svc.HashPassword("secret-password")
	if err != nil {
		t.Fatal(err)
	}
	if _, token, err := svc.BootstrapRegister(
		context.Background(), "admin@example.test", "Admin", "secret-password", retryHash,
	); err != nil {
		t.Fatalf("retry BootstrapRegister() error = %v", err)
	} else if claims, parseErr := svc.ParseToken(token); parseErr != nil || claims.ExpiresAt == nil ||
		!claims.ExpiresAt.Time.UTC().Equal(now.Add(time.Hour)) {
		t.Fatalf("rotated bootstrap token claims=%#v error=%v", claims, parseErr)
	}
	if len(provisioner.identities) != 2 {
		t.Fatalf("provisioner calls = %d, want 2", len(provisioner.identities))
	}
	second := provisioner.identities[1]
	if second.UserID != first.UserID || second.SessionID == first.SessionID ||
		!first.AuthenticatedAt.Equal(now.Add(-2*time.Hour)) || !second.AuthenticatedAt.Equal(now) ||
		!second.ExpiresAt.Equal(now.Add(time.Hour)) {
		t.Fatalf("rotated bootstrap identities = %#v", provisioner.identities)
	}
}

func TestCompleteBootstrapAdminRejectsAnyReservedSessionTupleMismatch(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*localSessionDraft)
	}{
		{name: "issued at", mutate: func(session *localSessionDraft) { session.IssuedAt++ }},
		{name: "expires at", mutate: func(session *localSessionDraft) { session.ExpiresAt++ }},
	} {
		t.Run(test.name, func(t *testing.T) {
			svc := newTestAuthService(t)
			now := time.Date(2026, time.August, 24, 10, 0, 0, 0, time.UTC)
			hash, err := svc.HashPassword("secret-password")
			if err != nil {
				t.Fatal(err)
			}
			pending, _, _, err := svc.store.PrepareBootstrapAdmin(
				t.Context(), "admin@example.test", "Admin", hash, now, now.Add(time.Hour),
			)
			if err != nil {
				t.Fatal(err)
			}
			session := localSessionDraft{
				ID: pending.SessionID, IssuedAt: pending.SessionIssuedAt.Unix(),
				ExpiresAt: pending.SessionExpiresAt.Unix(),
			}
			test.mutate(&session)
			if _, err := svc.store.CompleteBootstrapAdmin(
				t.Context(), pending.User.ID, session, now,
			); !errors.Is(err, ErrBootstrapClosed) {
				t.Fatalf("CompleteBootstrapAdmin() error = %v, want ErrBootstrapClosed", err)
			}
			var status string
			if err := svc.store.db.QueryRow(`SELECT status FROM users WHERE id=?`, pending.User.ID).Scan(&status); err != nil {
				t.Fatal(err)
			}
			var activeSessions int
			if err := svc.store.db.QueryRow(`SELECT COUNT(*) FROM auth_sessions WHERE user_id=?`, pending.User.ID).Scan(&activeSessions); err != nil {
				t.Fatal(err)
			}
			if status != bootstrapUserStatusProvisioning || activeSessions != 0 {
				t.Fatalf("status=%q active sessions=%d", status, activeSessions)
			}
		})
	}
}

func TestManagedBootstrapRejectsASecondIdentityDuringRetry(t *testing.T) {
	svc := newTestAuthService(t)
	provisioner := &recordingFirstAdminProvisioner{failures: 1}
	svc.ConfigureFirstAdminProvisioner(provisioner)
	hash, err := svc.HashPassword("secret-password")
	if err != nil {
		t.Fatal(err)
	}
	_, _, _ = svc.BootstrapRegister(
		context.Background(), "admin@example.test", "Admin", "secret-password", hash,
	)

	secondHash, err := svc.HashPassword("secret-password")
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = svc.BootstrapRegister(
		context.Background(), "other@example.test", "Other", "secret-password", secondHash,
	)
	if !errors.Is(err, ErrBootstrapClosed) {
		t.Fatalf("second identity error = %v, want ErrBootstrapClosed", err)
	}
	if len(provisioner.identities) != 1 {
		t.Fatalf("provisioner calls = %d, want 1", len(provisioner.identities))
	}
}
