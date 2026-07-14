package auth

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestSessionStoreLifecycle(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "session-store@example.com", RoleRead, "active")
	now := time.Now().Unix()
	sessionID := "session-1"

	if err := svc.store.CreateSession(context.Background(), sessionID, user.ID, now, now+60); err != nil {
		t.Fatalf("CreateSession() error = %v", err)
	}

	active, err := svc.store.SessionActive(context.Background(), sessionID, user.ID, now)
	if err != nil {
		t.Fatalf("SessionActive() error = %v", err)
	}
	if !active {
		t.Fatalf("session should be active before expiry")
	}

	active, err = svc.store.SessionActive(context.Background(), sessionID, user.ID, now+61)
	if err != nil {
		t.Fatalf("SessionActive() after expiry error = %v", err)
	}
	if active {
		t.Fatalf("session should be inactive after expiry")
	}

	if revokeErr := svc.store.RevokeSession(context.Background(), sessionID); revokeErr != nil {
		t.Fatalf("RevokeSession() error = %v", revokeErr)
	}
	active, err = svc.store.SessionActive(context.Background(), sessionID, user.ID, now)
	if err != nil {
		t.Fatalf("SessionActive() after revoke error = %v", err)
	}
	if active {
		t.Fatalf("session should be inactive after revoke")
	}
}

func TestSessionStoreMissingSessionIsInactive(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "missing-session@example.com", RoleRead, "active")

	active, err := svc.store.SessionActive(context.Background(), "missing-session", user.ID, time.Now().Unix())
	if err != nil {
		t.Fatalf("SessionActive() error = %v", err)
	}
	if active {
		t.Fatalf("missing session should be inactive")
	}
}

func TestCreateSessionRejectsMissingIdentifiers(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	now := time.Now().Unix()
	for _, test := range []struct {
		name      string
		sessionID string
		userID    string
	}{
		{name: "empty session", sessionID: "", userID: "user-id"},
		{name: "blank session", sessionID: "   ", userID: "user-id"},
		{name: "empty user", sessionID: "session-id", userID: ""},
		{name: "blank user", sessionID: "session-id", userID: "\t"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := svc.store.CreateSession(
				context.Background(),
				test.sessionID,
				test.userID,
				now,
				now+60,
			)
			if err == nil || !strings.Contains(err.Error(), "required") {
				t.Fatalf("CreateSession() error = %v, want safe required-field error", err)
			}
		})
	}
}

func TestSessionIssuanceBoundsActiveAndRetainedRowsPerUser(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "bounded-sessions@example.com", RoleRead, "active")
	issuedTokens := make([]string, 0, maxRetainedAuthSessionsPerUser+20)
	for range maxRetainedAuthSessionsPerUser + 20 {
		token, err := svc.issueToken(user)
		if err != nil {
			t.Fatalf("issueToken() error = %v", err)
		}
		issuedTokens = append(issuedTokens, token)
	}

	now := time.Now().Unix()
	var activeCount, totalCount int
	if err := svc.store.db.QueryRowContext(
		context.Background(),
		`SELECT COUNT(*) FROM auth_sessions
		 WHERE user_id = ? AND revoked_at IS NULL AND expires_at > ?`,
		user.ID,
		now,
	).Scan(&activeCount); err != nil {
		t.Fatalf("count active sessions: %v", err)
	}
	if err := svc.store.db.QueryRowContext(
		context.Background(),
		`SELECT COUNT(*) FROM auth_sessions WHERE user_id = ?`,
		user.ID,
	).Scan(&totalCount); err != nil {
		t.Fatalf("count retained sessions: %v", err)
	}
	if activeCount != maxActiveAuthSessionsPerUser {
		t.Fatalf("active session count = %d, want %d", activeCount, maxActiveAuthSessionsPerUser)
	}
	if totalCount != maxRetainedAuthSessionsPerUser {
		t.Fatalf("retained session count = %d, want %d", totalCount, maxRetainedAuthSessionsPerUser)
	}

	// The newest token must survive eviction even when many tokens share the
	// same second-level issued_at timestamp.
	newestClaims, err := svc.ParseToken(issuedTokens[len(issuedTokens)-1])
	if err != nil {
		t.Fatalf("parse newest token: %v", err)
	}
	if _, _, resolveErr := svc.ResolveSessionUser(context.Background(), newestClaims); resolveErr != nil {
		t.Fatalf("newest token is not active: %v", resolveErr)
	}
	oldestClaims, err := svc.ParseToken(issuedTokens[0])
	if err != nil {
		t.Fatalf("parse oldest token: %v", err)
	}
	if _, _, err := svc.ResolveSessionUser(context.Background(), oldestClaims); err == nil {
		t.Fatal("oldest excess token remained active")
	}
}

func TestPruneInactiveSessionsKeepsRecentInactiveRecords(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "prune-session@example.com", RoleRead, "active")
	now := time.Now()
	oldCutoff := now.Add(-inactiveAuthSessionRetention - time.Hour).Unix()
	recentCutoff := now.Add(-inactiveAuthSessionRetention + time.Hour).Unix()
	activeExpiry := now.Add(time.Hour).Unix()

	sessions := []struct {
		id         string
		expiresAt  int64
		revokedAt  *int64
		wantActive bool
	}{
		{id: "old-expired", expiresAt: oldCutoff, wantActive: false},
		{id: "recent-expired", expiresAt: recentCutoff, wantActive: false},
		{id: "active", expiresAt: activeExpiry, wantActive: true},
		{id: "old-revoked", expiresAt: activeExpiry, revokedAt: int64Ptr(oldCutoff), wantActive: false},
		{id: "recent-revoked", expiresAt: activeExpiry, revokedAt: int64Ptr(recentCutoff), wantActive: false},
	}

	for _, session := range sessions {
		insertSessionWithoutEnforcement(t, svc.store, session.id, user.ID, now.Unix(), session.expiresAt)
		if session.revokedAt != nil {
			if _, err := svc.store.db.ExecContext(context.Background(), `UPDATE auth_sessions SET revoked_at = ? WHERE id = ?`, *session.revokedAt, session.id); err != nil {
				t.Fatalf("mark revoked %q: %v", session.id, err)
			}
		}
	}

	if err := svc.store.PruneInactiveSessions(context.Background(), now); err != nil {
		t.Fatalf("PruneInactiveSessions() error = %v", err)
	}

	for _, session := range sessions {
		active, err := svc.store.SessionActive(context.Background(), session.id, user.ID, now.Unix())
		if err != nil {
			t.Fatalf("SessionActive(%q) error = %v", session.id, err)
		}
		if active != session.wantActive {
			t.Fatalf("SessionActive(%q) = %v, want %v", session.id, active, session.wantActive)
		}
	}

	for _, sessionID := range []string{"old-expired", "old-revoked"} {
		if sessionExists(t, svc.store, sessionID) {
			t.Fatalf("session %q should be pruned", sessionID)
		}
	}
	for _, sessionID := range []string{"recent-expired", "recent-revoked", "active"} {
		if !sessionExists(t, svc.store, sessionID) {
			t.Fatalf("session %q should be retained", sessionID)
		}
	}
}

func TestNewStorePrunesInactiveSessionsOnOpen(t *testing.T) {
	t.Parallel()

	dbPath := t.TempDir() + "/auth.db"
	initialStore, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	svc, err := NewService(initialStore, testJWTSecret, 1)
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	user := newTestUser(t, svc, "startup-prune@example.com", RoleRead, "active")
	oldCutoff := time.Now().Add(-inactiveAuthSessionRetention - time.Hour)
	insertSessionWithoutEnforcement(
		t,
		svc.store,
		"old-expired",
		user.ID,
		oldCutoff.Add(-time.Hour).Unix(),
		oldCutoff.Unix(),
	)
	if err := initialStore.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	store := mustNewStore(t, dbPath)
	t.Cleanup(func() {
		_ = store.Close()
	})
	if sessionExists(t, store, "old-expired") {
		t.Fatalf("old inactive session should be pruned when store opens")
	}
}

func mustNewStore(t *testing.T, dbPath string) *Store {
	t.Helper()
	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	return store
}

func sessionExists(t *testing.T, store *Store, sessionID string) bool {
	t.Helper()
	var count int
	if err := store.db.QueryRowContext(context.Background(), `SELECT COUNT(*) FROM auth_sessions WHERE id = ?`, sessionID).Scan(&count); err != nil {
		t.Fatalf("count session %q: %v", sessionID, err)
	}
	return count > 0
}

func insertSessionWithoutEnforcement(
	t *testing.T,
	store *Store,
	sessionID string,
	userID string,
	issuedAt int64,
	expiresAt int64,
) {
	t.Helper()
	if _, err := store.db.ExecContext(
		context.Background(),
		`INSERT INTO auth_sessions(id, user_id, issued_at, expires_at)
		 VALUES (?, ?, ?, ?)`,
		sessionID,
		userID,
		issuedAt,
		expiresAt,
	); err != nil {
		t.Fatalf("insert session fixture %q: %v", sessionID, err)
	}
}

func int64Ptr(value int64) *int64 {
	return &value
}
