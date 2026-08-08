package auth

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestResetUserPasswordWithAuditCommitsPasswordSessionsAuditAndIdempotency(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-reset@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-reset@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}

	const rotatedPassword = "rotated-password"
	rotatedHash, err := svc.HashPassword(rotatedPassword)
	if err != nil {
		t.Fatalf("HashPassword() error = %v", err)
	}

	result, err := svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		Operation:          CredentialLifecycleAdminPasswordReset,
		AuditAction:        legacyUserPasswordAuditAction,
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       rotatedHash,
		IdempotencyKey:     "reset-key-1",
		RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, rotatedPassword),
		Method:             http.MethodPost,
		Path:               "/api/admin/users/password",
		StatusCode:         http.StatusOK,
	})
	if err != nil {
		t.Fatalf("ResetUserPasswordWithAudit() error = %v", err)
	}
	if result.Replayed {
		t.Fatalf("first reset should not be replayed")
	}
	if result.RevokedSessions != 1 {
		t.Fatalf("revoked sessions = %d, want 1", result.RevokedSessions)
	}

	storedHash := userPasswordHash(t, svc.store, target.ID)
	if !svc.VerifyPassword(storedHash, rotatedPassword) {
		t.Fatalf("stored password hash does not verify rotated password")
	}

	assertTokenUnauthorized(t, svc, targetToken)

	logs, total, err := svc.store.QueryAuditLogs(context.Background(), AuditLogListOptions{
		Action: legacyUserPasswordAuditAction,
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("QueryAuditLogs() error = %v", err)
	}
	if total != 1 || len(logs) != 1 {
		t.Fatalf("audit rows = %d/%d, want exactly one", len(logs), total)
	}
	if logs[0].UserID != actor.ID {
		t.Fatalf("audit user id = %q, want actor %q", logs[0].UserID, actor.ID)
	}
	assertCredentialAuditExtra(t, logs[0].ExtraJSON, target.ID, true, 1)
	assertAuditExtraDoesNotContainSecrets(t, logs[0].ExtraJSON, rotatedPassword, rotatedHash, targetToken)

	if count := credentialLifecycleRequestCount(t, svc.store); count != 1 {
		t.Fatalf("credential lifecycle request count = %d, want 1", count)
	}
}

func TestResetUserPasswordWithAuditReplaysWithoutSecondMutationOrAudit(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-replay@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-replay@example.com", RoleRead, defaultUserStatusActive)
	const rotatedPassword = "rotated-password"
	fingerprint := svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, rotatedPassword)

	firstHash, err := svc.HashPassword(rotatedPassword)
	if err != nil {
		t.Fatalf("HashPassword(first) error = %v", err)
	}
	_, resetErr := svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       firstHash,
		IdempotencyKey:     "reset-key-replay",
		RequestFingerprint: fingerprint,
	})
	if resetErr != nil {
		t.Fatalf("first ResetUserPasswordWithAudit() error = %v", resetErr)
	}
	storedAfterFirst := userPasswordHash(t, svc.store, target.ID)

	secondHash, err := svc.HashPassword(rotatedPassword)
	if err != nil {
		t.Fatalf("HashPassword(second) error = %v", err)
	}
	result, err := svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       secondHash,
		IdempotencyKey:     "reset-key-replay",
		RequestFingerprint: fingerprint,
	})
	if err != nil {
		t.Fatalf("replay ResetUserPasswordWithAudit() error = %v", err)
	}
	if !result.Replayed {
		t.Fatalf("second reset should be reported as replayed")
	}
	if storedAfterSecond := userPasswordHash(t, svc.store, target.ID); storedAfterSecond != storedAfterFirst {
		t.Fatalf("password hash changed on replay")
	}
	if count := auditLogCount(t, svc.store, legacyUserPasswordAuditAction); count != 1 {
		t.Fatalf("audit count = %d, want 1 after replay", count)
	}
}

func TestResetUserPasswordWithAuditRejectsConflictingIdempotencyKey(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-conflict@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-conflict@example.com", RoleRead, defaultUserStatusActive)
	firstHash, err := svc.HashPassword("first-password")
	if err != nil {
		t.Fatalf("HashPassword(first) error = %v", err)
	}
	_, resetErr := svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       firstHash,
		IdempotencyKey:     "reset-key-conflict",
		RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, "first-password"),
	})
	if resetErr != nil {
		t.Fatalf("first ResetUserPasswordWithAudit() error = %v", resetErr)
	}
	storedAfterFirst := userPasswordHash(t, svc.store, target.ID)

	secondHash, err := svc.HashPassword("second-password")
	if err != nil {
		t.Fatalf("HashPassword(second) error = %v", err)
	}
	_, err = svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       secondHash,
		IdempotencyKey:     "reset-key-conflict",
		RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, "second-password"),
	})
	if !errors.Is(err, ErrCredentialLifecycleConflict) {
		t.Fatalf("error = %v, want ErrCredentialLifecycleConflict", err)
	}
	if storedAfterConflict := userPasswordHash(t, svc.store, target.ID); storedAfterConflict != storedAfterFirst {
		t.Fatalf("password hash changed after idempotency conflict")
	}
	if count := auditLogCount(t, svc.store, legacyUserPasswordAuditAction); count != 1 {
		t.Fatalf("audit count = %d, want 1 after conflict", count)
	}
}

func TestResetUserPasswordWithAuditRequiresFingerprintForIdempotencyKey(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-missing-fingerprint@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-missing-fingerprint@example.com", RoleRead, defaultUserStatusActive)
	originalHash := userPasswordHash(t, svc.store, target.ID)
	rotatedHash, err := svc.HashPassword("rotated-password")
	if err != nil {
		t.Fatalf("HashPassword() error = %v", err)
	}

	_, err = svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:    actor.ID,
		TargetUserID:   target.ID,
		PasswordHash:   rotatedHash,
		IdempotencyKey: "reset-key-missing-fingerprint",
	})
	if err == nil || !strings.Contains(err.Error(), "request fingerprint is required") {
		t.Fatalf("error = %v, want missing fingerprint validation", err)
	}
	if storedAfterFailure := userPasswordHash(t, svc.store, target.ID); storedAfterFailure != originalHash {
		t.Fatalf("password hash changed after missing fingerprint validation")
	}
	if count := credentialLifecycleRequestCount(t, svc.store); count != 0 {
		t.Fatalf("credential lifecycle request count = %d, want 0", count)
	}
	if count := auditLogCount(t, svc.store, legacyUserPasswordAuditAction); count != 0 {
		t.Fatalf("audit count = %d, want 0", count)
	}
}

func TestResetUserPasswordWithAuditReplaysAfterStoreReopen(t *testing.T) {
	t.Parallel()

	dbPath := filepath.Join(t.TempDir(), "auth.db")
	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	svc := NewService(store, "test-secret", 1)
	actor := newTestUser(t, svc, "admin-restart@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-restart@example.com", RoleRead, defaultUserStatusActive)

	const rotatedPassword = "rotated-password"
	firstHash, err := svc.HashPassword(rotatedPassword)
	if err != nil {
		t.Fatalf("HashPassword(first) error = %v", err)
	}
	fingerprint := svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, rotatedPassword)
	_, resetErr := store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       firstHash,
		IdempotencyKey:     "reset-key-restart",
		RequestFingerprint: fingerprint,
	})
	if resetErr != nil {
		t.Fatalf("first ResetUserPasswordWithAudit() error = %v", resetErr)
	}
	storedAfterFirst := userPasswordHash(t, store, target.ID)
	if closeErr := store.Close(); closeErr != nil {
		t.Fatalf("Close() error = %v", closeErr)
	}

	reopened, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("reopen NewStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = reopened.Close()
	})
	reopenedSvc := NewService(reopened, "test-secret", 1)
	secondHash, err := reopenedSvc.HashPassword(rotatedPassword)
	if err != nil {
		t.Fatalf("HashPassword(second) error = %v", err)
	}
	result, err := reopened.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       secondHash,
		IdempotencyKey:     "reset-key-restart",
		RequestFingerprint: reopenedSvc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, rotatedPassword),
	})
	if err != nil {
		t.Fatalf("reopened ResetUserPasswordWithAudit() error = %v", err)
	}
	if !result.Replayed {
		t.Fatalf("reopened reset should replay")
	}
	if storedAfterReplay := userPasswordHash(t, reopened, target.ID); storedAfterReplay != storedAfterFirst {
		t.Fatalf("password hash changed after replay across reopen")
	}
	if count := auditLogCount(t, reopened, legacyUserPasswordAuditAction); count != 1 {
		t.Fatalf("audit count = %d, want 1 after reopen replay", count)
	}
}

func TestResetUserPasswordWithAuditCanceledContextDoesNotMutate(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-canceled@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-canceled@example.com", RoleRead, defaultUserStatusActive)
	originalHash := userPasswordHash(t, svc.store, target.ID)
	rotatedHash, err := svc.HashPassword("rotated-password")
	if err != nil {
		t.Fatalf("HashPassword() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = svc.store.ResetUserPasswordWithAudit(ctx, CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       rotatedHash,
		IdempotencyKey:     "reset-key-canceled",
		RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, "rotated-password"),
	})
	if err == nil {
		t.Fatalf("ResetUserPasswordWithAudit() error = nil, want canceled context failure")
	}
	if storedHash := userPasswordHash(t, svc.store, target.ID); storedHash != originalHash {
		t.Fatalf("password hash changed after canceled context")
	}
	if count := auditLogCount(t, svc.store, legacyUserPasswordAuditAction); count != 0 {
		t.Fatalf("audit count = %d, want 0 after canceled context", count)
	}
	if count := credentialLifecycleRequestCount(t, svc.store); count != 0 {
		t.Fatalf("credential lifecycle request count = %d, want 0 after canceled context", count)
	}
}

func TestResetUserPasswordWithAuditRollsBackWhenAuditInsertFails(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-rollback@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-rollback@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}
	originalHash := userPasswordHash(t, svc.store, target.ID)
	_, triggerErr := svc.store.db.ExecContext(context.Background(), `
		CREATE TRIGGER fail_password_reset_audit
		BEFORE INSERT ON user_audit_logs
		WHEN NEW.action = 'user.password'
		BEGIN
			SELECT RAISE(ABORT, 'audit insert failed');
		END;`)
	if triggerErr != nil {
		t.Fatalf("create audit failure trigger: %v", triggerErr)
	}

	rotatedHash, err := svc.HashPassword("rotated-password")
	if err != nil {
		t.Fatalf("HashPassword() error = %v", err)
	}
	_, err = svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       rotatedHash,
		IdempotencyKey:     "reset-key-rollback",
		RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, "rotated-password"),
	})
	if err == nil {
		t.Fatalf("ResetUserPasswordWithAudit() error = nil, want trigger failure")
	}
	if storedHash := userPasswordHash(t, svc.store, target.ID); storedHash != originalHash {
		t.Fatalf("password hash changed despite audit failure rollback")
	}
	assertTokenAuthorized(t, svc, targetToken)
	if count := credentialLifecycleRequestCount(t, svc.store); count != 0 {
		t.Fatalf("credential lifecycle request count = %d, want 0 after rollback", count)
	}
}

func TestResetUserPasswordWithAuditRollsBackWhenDatabaseIsFull(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-full@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-full@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}
	originalHash := userPasswordHash(t, svc.store, target.ID)
	if _, execErr := svc.store.db.ExecContext(context.Background(), `
		CREATE TABLE full_disk_fill(payload BLOB);
		CREATE TRIGGER fill_disk_before_password_reset_audit
		BEFORE INSERT ON user_audit_logs
		WHEN NEW.action = 'user.password'
		BEGIN
			INSERT INTO full_disk_fill(payload) VALUES (zeroblob(1048576));
		END;`); execErr != nil {
		t.Fatalf("install full disk trigger: %v", execErr)
	}
	pageCount := sqlitePragmaInt(t, svc.store, "page_count")
	if _, pragmaErr := svc.store.db.ExecContext(context.Background(), fmt.Sprintf("PRAGMA max_page_count = %d", pageCount)); pragmaErr != nil {
		t.Fatalf("set max_page_count: %v", pragmaErr)
	}

	rotatedHash, err := svc.HashPassword("rotated-password")
	if err != nil {
		t.Fatalf("HashPassword() error = %v", err)
	}
	_, err = svc.store.ResetUserPasswordWithAudit(context.Background(), CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       rotatedHash,
		IdempotencyKey:     "reset-key-full",
		RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, "rotated-password"),
	})
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "full") {
		t.Fatalf("ResetUserPasswordWithAudit() error = %v, want database/full error", err)
	}
	if storedHash := userPasswordHash(t, svc.store, target.ID); storedHash != originalHash {
		t.Fatalf("password hash changed despite full-disk rollback")
	}
	assertTokenAuthorized(t, svc, targetToken)
	if count := credentialLifecycleRequestCount(t, svc.store); count != 0 {
		t.Fatalf("credential lifecycle request count = %d, want 0 after full-disk rollback", count)
	}
	if count := auditLogCount(t, svc.store, legacyUserPasswordAuditAction); count != 0 {
		t.Fatalf("audit count = %d, want 0 after full-disk rollback", count)
	}
}

func TestResetUserPasswordWithAuditRollsBackWhenDatabaseIsBusy(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "auth.db")
	store, err := NewStore(dbPath + "?_busy_timeout=50")
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	svc := NewService(store, "test-secret", 1)
	actor := newTestUser(t, svc, "admin-busy@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-busy@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}
	originalHash := userPasswordHash(t, store, target.ID)

	lockDB, err := sql.Open("sqlite3", authSQLiteDSN(dbPath+"?_busy_timeout=50"))
	if err != nil {
		t.Fatalf("open lock db: %v", err)
	}
	t.Cleanup(func() {
		_ = lockDB.Close()
	})
	if _, beginErr := lockDB.ExecContext(context.Background(), "BEGIN IMMEDIATE"); beginErr != nil {
		t.Fatalf("begin immediate lock: %v", beginErr)
	}
	t.Cleanup(func() {
		_, _ = lockDB.ExecContext(context.Background(), "ROLLBACK")
	})

	rotatedHash, err := svc.HashPassword("rotated-password")
	if err != nil {
		t.Fatalf("HashPassword() error = %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	_, err = store.ResetUserPasswordWithAudit(ctx, CredentialLifecycleMutation{
		ActorUserID:        actor.ID,
		TargetUserID:       target.ID,
		PasswordHash:       rotatedHash,
		IdempotencyKey:     "reset-key-busy",
		RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, target.ID, "rotated-password"),
	})
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "locked") {
		t.Fatalf("ResetUserPasswordWithAudit() error = %v, want database locked error", err)
	}
	if storedHash := userPasswordHash(t, store, target.ID); storedHash != originalHash {
		t.Fatalf("password hash changed despite busy rollback")
	}
	assertTokenAuthorized(t, svc, targetToken)
	if count := credentialLifecycleRequestCount(t, store); count != 0 {
		t.Fatalf("credential lifecycle request count = %d, want 0 after busy rollback", count)
	}
	if count := auditLogCount(t, store, legacyUserPasswordAuditAction); count != 0 {
		t.Fatalf("audit count = %d, want 0 after busy rollback", count)
	}
}

func userPasswordHash(t *testing.T, store *Store, userID string) string {
	t.Helper()
	var hash string
	if err := store.db.QueryRowContext(context.Background(), `SELECT password_hash FROM users WHERE id = ?`, userID).Scan(&hash); err != nil {
		t.Fatalf("query password hash: %v", err)
	}
	return hash
}

func auditLogCount(t *testing.T, store *Store, action string) int {
	t.Helper()
	var count int
	if err := store.db.QueryRowContext(context.Background(), `SELECT COUNT(*) FROM user_audit_logs WHERE action = ?`, action).Scan(&count); err != nil {
		t.Fatalf("count audit logs: %v", err)
	}
	return count
}

func credentialLifecycleRequestCount(t *testing.T, store *Store) int {
	t.Helper()
	var count int
	if err := store.db.QueryRowContext(context.Background(), `SELECT COUNT(*) FROM credential_lifecycle_requests`).Scan(&count); err != nil {
		t.Fatalf("count credential lifecycle requests: %v", err)
	}
	return count
}

func sqlitePragmaInt(t *testing.T, store *Store, name string) int {
	t.Helper()
	var value int
	if err := store.db.QueryRowContext(context.Background(), "PRAGMA "+name).Scan(&value); err != nil {
		t.Fatalf("read PRAGMA %s: %v", name, err)
	}
	return value
}

func assertCredentialAuditExtra(
	t *testing.T,
	extraJSON string,
	targetUserID string,
	wantIdempotent bool,
	wantRevokedSessions int64,
) {
	t.Helper()
	var payload struct {
		EventType        string `json:"eventType"`
		TargetUserID     string `json:"targetUserId"`
		Outcome          string `json:"outcome"`
		RevokedSessions  int64  `json:"revokedSessions"`
		Idempotent       bool   `json:"idempotent"`
		IdempotencyKeyID string `json:"idempotencyKeyID"`
	}
	if err := json.Unmarshal([]byte(extraJSON), &payload); err != nil {
		t.Fatalf("decode audit extra: %v", err)
	}
	if payload.EventType != CredentialLifecycleAdminPasswordReset {
		t.Fatalf("event type = %q, want %q", payload.EventType, CredentialLifecycleAdminPasswordReset)
	}
	if payload.TargetUserID != targetUserID {
		t.Fatalf("target user id = %q, want %q", payload.TargetUserID, targetUserID)
	}
	if payload.Outcome != "success" {
		t.Fatalf("outcome = %q, want success", payload.Outcome)
	}
	if payload.RevokedSessions != wantRevokedSessions {
		t.Fatalf("revoked sessions = %d, want %d", payload.RevokedSessions, wantRevokedSessions)
	}
	if payload.Idempotent != wantIdempotent {
		t.Fatalf("idempotent = %v, want %v", payload.Idempotent, wantIdempotent)
	}
	if wantIdempotent && payload.IdempotencyKeyID == "" {
		t.Fatalf("expected idempotency key id")
	}
}

func assertAuditExtraDoesNotContainSecrets(t *testing.T, extraJSON string, secrets ...string) {
	t.Helper()
	for _, secret := range secrets {
		if secret != "" && strings.Contains(extraJSON, secret) {
			t.Fatalf("audit extra leaked secret material %q in %s", secret, extraJSON)
		}
	}
}

func assertTokenUnauthorized(t *testing.T, svc *Service, token string) {
	t.Helper()
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})).ServeHTTP(recorder, req)
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want unauthorized", recorder.Code)
	}
}

func assertTokenAuthorized(t *testing.T, svc *Service, token string) {
	t.Helper()
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})).ServeHTTP(recorder, req)
	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want authorized", recorder.Code)
	}
}
