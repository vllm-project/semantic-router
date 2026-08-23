package postgres

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestLoadSelfNamespaceIntegrationReadsLinkedUser(t *testing.T) {
	database := bootstrapTestDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	principalID := uuid.NewString()
	namespaceID := uuid.NewString()
	userID := uuid.NewString()
	sessionID := uuid.NewString()
	email := userID + "@example.test"
	if _, err := database.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,$2,$3,'USD','active')`, namespaceID, "Self view "+namespaceID, "quota:"+namespaceID); err != nil {
		t.Fatalf("seed namespace: %v", err)
	}
	if _, err := database.ExecContext(ctx,
		`INSERT INTO self_service_policies (namespace_id,seed_version) VALUES ($1,1)`, namespaceID); err != nil {
		t.Fatalf("seed self-service policy: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,$2,$3,'Self-view integration principal','active')`,
		principalID, "https://issuer.example/"+principalID, "subject-"+principalID); err != nil {
		t.Fatalf("seed principal: %v", err)
	}
	if _, err := database.ExecContext(ctx,
		`INSERT INTO access_subjects (namespace_id,id,kind) VALUES ($1,$2,'user')`, namespaceID, userID); err != nil {
		t.Fatalf("seed User subject: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO access_users
  (namespace_id,id,email,display_name,status)
VALUES ($1,$2,$3,'Linked self-view user','active')`, namespaceID, userID, email); err != nil {
		t.Fatalf("seed User: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principal_user_links
  (principal_id,namespace_id,user_id) VALUES ($1,$2,$3)`, principalID, namespaceID, userID); err != nil {
		t.Fatalf("seed principal User link: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,status)
VALUES ($1,$2,'10000000-0000-5000-8000-000000000001','cluster','active')`, uuid.NewString(), principalID); err != nil {
		t.Fatalf("seed cluster role binding: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_sessions
  (id,principal_id,token_id,audience,auth_source_kind,evidence_kind,assurance,
   authenticated_at,expires_at,status)
VALUES ($1,$2,$3,'vllm-sr-management','issuer','human','{}'::jsonb,
        clock_timestamp(),clock_timestamp()+interval '5 minutes','active')`,
		sessionID, principalID, "self-view-token-"+sessionID); err != nil {
		t.Fatalf("seed Management session: %v", err)
	}
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = commands.Close() })
	store, err := New(database, commands)
	if err != nil {
		t.Fatal(err)
	}
	view, err := store.LoadSelf(ctx, principalID, sessionID)
	if err != nil {
		t.Fatalf("LoadSelf() error = %v", err)
	}
	if len(view.Namespaces) != 1 {
		t.Fatalf("LoadSelf() namespaces = %#v", view.Namespaces)
	}
	user := view.Namespaces[0].User
	if user == nil || user.ID != userID || user.Email != email ||
		user.DisplayName != "Linked self-view user" || user.Status != "active" {
		t.Fatalf("LoadSelf() User = %#v", user)
	}
}
