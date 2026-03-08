package console

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"
)

func TestOpenStoreDefaultsToSQLite(t *testing.T) {
	store, err := OpenStore(StoreConfig{
		SQLitePath: filepath.Join(t.TempDir(), "console.db"),
	})
	if err != nil {
		t.Fatalf("OpenStore() error = %v", err)
	}
	defer func() {
		_ = store.Lifecycle.Close()
	}()
}

func TestOpenStoreRejectsUnimplementedBackend(t *testing.T) {
	_, err := OpenStore(StoreConfig{
		Backend: StoreBackendPostgres,
		DSN:     "postgres://localhost/console",
	})
	if err == nil {
		t.Fatal("expected postgres backend to be rejected for now")
	}
}

func TestSQLiteStoreRoundTripsDomainEntities(t *testing.T) {
	ctx := context.Background()
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "console.db"))
	if err != nil {
		t.Fatalf("NewSQLiteStore() error = %v", err)
	}
	defer func() {
		_ = store.Close()
	}()

	user := createTestUser()
	if err := store.SaveUser(ctx, user); err != nil {
		t.Fatalf("SaveUser() error = %v", err)
	}
	assertUserRoundTrip(t, ctx, store, user)
	assertRoleBindingRoundTrip(t, ctx, store, user)
	assertSessionRoundTrip(t, ctx, store, user)

	revision := assertConfigRevisionRoundTrip(t, ctx, store, user)
	assertDeployEventRoundTrip(t, ctx, store, revision)
	assertSecretRefRoundTrip(t, ctx, store)
	assertAuditEventRoundTrip(t, ctx, store, user, revision)
}

func createTestUser() *User {
	return &User{
		Email:           "alice@example.com",
		DisplayName:     "Alice",
		AuthProvider:    "oidc",
		ExternalSubject: "oidc:alice",
		Metadata: map[string]interface{}{
			"team": "platform",
		},
	}
}

func assertUserRoundTrip(t *testing.T, ctx context.Context, store *SQLiteStore, user *User) {
	t.Helper()
	gotUser, err := store.GetUser(ctx, user.ID)
	if err != nil {
		t.Fatalf("GetUser() error = %v", err)
	}
	if gotUser == nil || gotUser.Email != user.Email {
		t.Fatalf("GetUser() mismatch: %#v", gotUser)
	}
}

func assertRoleBindingRoundTrip(t *testing.T, ctx context.Context, store *SQLiteStore, user *User) {
	t.Helper()
	roleBinding := &RoleBinding{
		PrincipalID: user.ID,
		Role:        ConsoleRoleAdmin,
		ScopeType:   ScopeTypeGlobal,
		Metadata: map[string]interface{}{
			"reason": "bootstrap",
		},
	}
	if err := store.SaveRoleBinding(ctx, roleBinding); err != nil {
		t.Fatalf("SaveRoleBinding() error = %v", err)
	}
	roleBindings, err := store.ListRoleBindings(ctx, RoleBindingFilter{PrincipalID: user.ID})
	if err != nil {
		t.Fatalf("ListRoleBindings() error = %v", err)
	}
	if len(roleBindings) != 1 || roleBindings[0].Role != ConsoleRoleAdmin {
		t.Fatalf("ListRoleBindings() mismatch: %#v", roleBindings)
	}
}

func assertSessionRoundTrip(t *testing.T, ctx context.Context, store *SQLiteStore, user *User) {
	t.Helper()
	expiresAt := time.Now().UTC().Add(2 * time.Hour)
	session := &Session{
		UserID:          user.ID,
		AuthProvider:    "oidc",
		ExternalSubject: user.ExternalSubject,
		ExpiresAt:       &expiresAt,
	}
	if err := store.SaveSession(ctx, session); err != nil {
		t.Fatalf("SaveSession() error = %v", err)
	}
	gotSession, err := store.GetSession(ctx, session.ID)
	if err != nil {
		t.Fatalf("GetSession() error = %v", err)
	}
	if gotSession == nil || gotSession.UserID != user.ID {
		t.Fatalf("GetSession() mismatch: %#v", gotSession)
	}
	sessions, err := store.ListSessions(ctx, SessionFilter{UserID: user.ID})
	if err != nil {
		t.Fatalf("ListSessions() error = %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected one session, got %d", len(sessions))
	}
}

func assertConfigRevisionRoundTrip(t *testing.T, ctx context.Context, store *SQLiteStore, user *User) *ConfigRevision {
	t.Helper()
	documentJSON, err := json.Marshal(map[string]interface{}{
		"version": "v0.1",
		"mode":    "draft",
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	revision := &ConfigRevision{
		Status:       ConfigRevisionStatusDraft,
		Source:       "config_ui",
		Summary:      "Initial enterprise-console draft",
		DocumentJSON: documentJSON,
		CreatedBy:    user.ID,
		Metadata: map[string]interface{}{
			"surface": "dashboard",
		},
	}
	saveErr := store.SaveConfigRevision(ctx, revision)
	if saveErr != nil {
		t.Fatalf("SaveConfigRevision() error = %v", saveErr)
	}
	gotRevision, err := store.GetConfigRevision(ctx, revision.ID)
	if err != nil {
		t.Fatalf("GetConfigRevision() error = %v", err)
	}
	if gotRevision == nil || string(gotRevision.DocumentJSON) != string(documentJSON) {
		t.Fatalf("GetConfigRevision() mismatch: %#v", gotRevision)
	}
	revisions, err := store.ListConfigRevisions(ctx, ConfigRevisionFilter{Source: "config_ui"})
	if err != nil {
		t.Fatalf("ListConfigRevisions() error = %v", err)
	}
	if len(revisions) != 1 {
		t.Fatalf("expected one config revision, got %d", len(revisions))
	}
	return revision
}

func assertDeployEventRoundTrip(t *testing.T, ctx context.Context, store *SQLiteStore, revision *ConfigRevision) {
	t.Helper()
	now := time.Now().UTC()
	deployEvent := &DeployEvent{
		RevisionID:    revision.ID,
		Status:        DeployEventStatusRunning,
		TriggerSource: "manual",
		RuntimeTarget: "local-dev",
		StartedAt:     &now,
	}
	if err := store.SaveDeployEvent(ctx, deployEvent); err != nil {
		t.Fatalf("SaveDeployEvent() error = %v", err)
	}
	deployEvents, err := store.ListDeployEvents(ctx, DeployEventFilter{RevisionID: revision.ID})
	if err != nil {
		t.Fatalf("ListDeployEvents() error = %v", err)
	}
	if len(deployEvents) != 1 || deployEvents[0].RevisionID != revision.ID {
		t.Fatalf("ListDeployEvents() mismatch: %#v", deployEvents)
	}
}

func assertSecretRefRoundTrip(t *testing.T, ctx context.Context, store *SQLiteStore) {
	t.Helper()
	secretRef := &SecretRef{
		ScopeType:     ScopeTypeEnvironment,
		ScopeID:       "cpu-local",
		Provider:      "env",
		ExternalRef:   "OPENAI_API_KEY",
		RedactedLabel: "OPENAI_API_KEY",
	}
	if err := store.SaveSecretRef(ctx, secretRef); err != nil {
		t.Fatalf("SaveSecretRef() error = %v", err)
	}
	secretRefs, err := store.ListSecretRefs(ctx, SecretRefFilter{ScopeType: ScopeTypeEnvironment})
	if err != nil {
		t.Fatalf("ListSecretRefs() error = %v", err)
	}
	if len(secretRefs) != 1 || secretRefs[0].ExternalRef != "OPENAI_API_KEY" {
		t.Fatalf("ListSecretRefs() mismatch: %#v", secretRefs)
	}
}

func assertAuditEventRoundTrip(t *testing.T, ctx context.Context, store *SQLiteStore, user *User, revision *ConfigRevision) {
	t.Helper()
	auditEvent := &AuditEvent{
		ActorType:  PrincipalTypeUser,
		ActorID:    user.ID,
		Action:     "config.activate",
		TargetType: "config_revision",
		TargetID:   revision.ID,
		Outcome:    AuditOutcomeSuccess,
		Message:    "Activated console-managed config revision",
	}
	if err := store.AppendAuditEvent(ctx, auditEvent); err != nil {
		t.Fatalf("AppendAuditEvent() error = %v", err)
	}
	auditEvents, err := store.ListAuditEvents(ctx, AuditEventFilter{ActorID: user.ID})
	if err != nil {
		t.Fatalf("ListAuditEvents() error = %v", err)
	}
	if len(auditEvents) != 1 || auditEvents[0].Action != "config.activate" {
		t.Fatalf("ListAuditEvents() mismatch: %#v", auditEvents)
	}
}

func TestSQLiteStoreRequiresConfigRevisionDocument(t *testing.T) {
	ctx := context.Background()
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "console.db"))
	if err != nil {
		t.Fatalf("NewSQLiteStore() error = %v", err)
	}
	defer func() {
		_ = store.Close()
	}()

	if err := store.SaveConfigRevision(ctx, &ConfigRevision{}); err == nil {
		t.Fatal("expected SaveConfigRevision() to reject empty document_json")
	}
}
