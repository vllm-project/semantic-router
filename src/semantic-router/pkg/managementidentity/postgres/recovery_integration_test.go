package postgres

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const recoveryTestToken = "recovery-token-that-is-at-least-thirty-two-bytes"

func TestRecoveryRestoresExistingAdministratorAndReplaysExactly(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Now().UTC().Add(-time.Second)
	bootstrap := newBootstrapTestService(t, database, func() time.Time { return now })
	bootstrapResult, testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr := bootstrap.Bootstrap(
		context.Background(),
		bootstrapTestRequest("bootstrap-recovery-test-0001", "Recovery target"),
		bootstrapTestToken,
	)
	if testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr != nil {
		t.Fatal(testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr)
	}
	if _, err := database.Exec(`UPDATE management_principals SET status='disabled' WHERE id=$1`, bootstrapResult.PrincipalID); err != nil {
		t.Fatal(err)
	}
	if _, err := database.Exec(`UPDATE management_role_bindings SET status='disabled' WHERE id=$1`, bootstrapResult.RoleBindingID); err != nil {
		t.Fatal(err)
	}

	recovery, testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr := NewRecoveryService(RecoveryOptions{
		Database: database, RecoveryToken: []byte(recoveryTestToken),
		IdempotencyKeys: securitykeyring.Symmetric{
			ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{7}, 32)},
		},
		Now: func() time.Time { return now },
	})
	if testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr != nil {
		t.Fatal(testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr)
	}
	t.Cleanup(recovery.Close)
	request := managementidentity.RecoveryRequest{
		PrincipalID: bootstrapResult.PrincipalID, Reason: "Restore after identity-provider lockout",
		RequestID: "recovery-request-1", IdempotencyKey: "recovery-idempotency-0001",
		CanonicalRequest: []byte(`{"principalId":"` + bootstrapResult.PrincipalID + `","reason":"Restore after identity-provider lockout"}`),
	}
	result, testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr := recovery.Recover(context.Background(), request, recoveryTestToken)
	if testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr != nil {
		t.Fatal(testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr)
	}
	if result.PrincipalID != bootstrapResult.PrincipalID || result.RoleBindingID != bootstrapResult.RoleBindingID || result.Replayed {
		t.Fatalf("recovery result = %+v", result)
	}
	var principalStatus, bindingStatus string
	if err := database.QueryRow(`SELECT principal.status,binding.status
FROM management_principals principal
JOIN management_role_bindings binding ON binding.principal_id=principal.id
WHERE principal.id=$1 AND binding.id=$2`, result.PrincipalID, result.RoleBindingID).Scan(&principalStatus, &bindingStatus); err != nil {
		t.Fatal(err)
	}
	if principalStatus != "active" || bindingStatus != "active" {
		t.Fatalf("principal=%s binding=%s", principalStatus, bindingStatus)
	}

	replayed, testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr := recovery.Recover(context.Background(), request, recoveryTestToken)
	if testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr != nil || !replayed.Replayed || replayed.RoleBindingID != result.RoleBindingID {
		t.Fatalf("replay = %+v, %v", replayed, testRecoveryRestoresExistingAdministratorAndReplaysExactlyErr)
	}
	changed := request
	changed.CanonicalRequest = []byte(`{"principalId":"` + bootstrapResult.PrincipalID + `","reason":"Different request"}`)
	if _, err := recovery.Recover(context.Background(), changed, recoveryTestToken); !errors.Is(err, managementidentity.ErrRecoveryConflict) {
		t.Fatalf("changed recovery = %v", err)
	}
	otherKey := request
	otherKey.IdempotencyKey = "recovery-idempotency-0002"
	if _, err := recovery.Recover(context.Background(), otherKey, recoveryTestToken); !errors.Is(err, managementidentity.ErrRecoveryConsumed) {
		t.Fatalf("second recovery = %v", err)
	}
	if err := recovery.Ready(context.Background()); err == nil {
		t.Fatal("readiness accepted a consumed recovery credential")
	}
}
