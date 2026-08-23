package postgres

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestCreateToolCredentialCommitsDomainMutationAndCommandReceiptTogether(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })

	now := time.Now().UTC().Truncate(time.Microsecond)
	namespaceID := uuid.NewString()
	principalID := uuid.NewString()
	resourceID := uuid.NewString()
	command := testResourceCommand(t, namespaceID, principalID, now)
	active := command.ActiveDigest()

	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT clock_timestamp\(\)`).
		WithArgs(command.AdvisoryLockKey()).
		WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(now))
	mock.ExpectQuery(`SELECT request_digest, operation_id::text,`).
		WithArgs(
			string(managementcommand.ScopeNamespace), namespaceID, principalID,
			command.Endpoint, active.HMACVersion, active.KeyDigest[:],
		).
		WillReturnError(sql.ErrNoRows)
	mock.ExpectExec(`INSERT INTO agent_tool_credentials`).
		WithArgs(resourceID, namespaceID, "search-token", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(`INSERT INTO agent_tool_credential_versions`).
		WithArgs(
			sqlmock.AnyArg(), namespaceID, resourceID, []byte("ciphertext"),
			[]byte("nonce"), "kek-v1",
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(`INSERT INTO management_idempotency`).
		WithArgs(
			string(managementcommand.ScopeNamespace), namespaceID, principalID,
			command.Endpoint, active.HMACVersion, active.KeyDigest[:], active.RequestDigest[:],
			agentToolCredentialResourceType, resourceID, uint64(1), 201, command.ExpiresAt,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit()

	store := &Store{db: database}
	result, err := store.CreateToolCredential(
		context.Background(), namespaceID, resourceID, "search-token",
		agentmanagement.EncryptedSecret{
			Ciphertext: []byte("ciphertext"), Nonce: []byte("nonce"), KEKVersion: "kek-v1",
		},
		agentmanagement.ResourceCommand{
			Mutation: agentmanagement.MutationContext{PrincipalID: principalID},
			Command:  command,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.ResourceID != resourceID || result.ResourceRevision != 1 || result.Replayed {
		t.Fatalf("receipt = %+v", result)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestCreateToolCredentialReplaysReceiptWithoutDomainMutation(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })

	now := time.Now().UTC().Truncate(time.Microsecond)
	namespaceID := uuid.NewString()
	principalID := uuid.NewString()
	resourceID := uuid.NewString()
	command := testResourceCommand(t, namespaceID, principalID, now)
	active := command.ActiveDigest()

	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT clock_timestamp\(\)`).
		WithArgs(command.AdvisoryLockKey()).
		WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(now))
	mock.ExpectQuery(`SELECT request_digest, operation_id::text,`).
		WithArgs(
			string(managementcommand.ScopeNamespace), namespaceID, principalID,
			command.Endpoint, active.HMACVersion, active.KeyDigest[:],
		).
		WillReturnRows(resourceCommandRows().AddRow(
			active.RequestDigest[:], nil, agentToolCredentialResourceType, resourceID,
			int64(1), nil, 201, nil, nil, nil, nil, command.ExpiresAt,
		))
	mock.ExpectCommit()

	store := &Store{db: database}
	result, err := store.CreateToolCredential(
		context.Background(), namespaceID, uuid.NewString(), "ignored-on-replay",
		agentmanagement.EncryptedSecret{
			Ciphertext: []byte("ciphertext"), Nonce: []byte("nonce"), KEKVersion: "kek-v1",
		},
		agentmanagement.ResourceCommand{
			Mutation: agentmanagement.MutationContext{PrincipalID: principalID},
			Command:  command,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.ResourceID != resourceID || result.ResourceRevision != 1 || !result.Replayed {
		t.Fatalf("replayed receipt = %+v", result)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestResourceCommandRejectsScopeAndReceiptConfusion(t *testing.T) {
	now := time.Now().UTC()
	namespaceID := uuid.NewString()
	principalID := uuid.NewString()
	command := testResourceCommand(t, namespaceID, principalID, now)

	if _, _, err := lockResourceCommand(
		context.Background(), nil, namespaceID, agentToolCredentialResourceType,
		agentmanagement.ResourceCommand{
			Mutation: agentmanagement.MutationContext{PrincipalID: uuid.NewString()},
			Command:  command,
		},
	); !errors.Is(err, agentmanagement.ErrInvalid) {
		t.Fatalf("principal-confused command error = %v", err)
	}
	for _, stored := range []managementcommand.StoredResult{
		{Resource: &managementcommand.ResourceResult{
			ResourceType: "agent_skill", ResourceID: uuid.NewString(), ResourceRevision: 1,
		}},
		{Resource: &managementcommand.ResourceResult{
			ResourceType: agentToolCredentialResourceType, ResourceID: uuid.NewString(), ResourceRevision: 0,
		}},
	} {
		if _, err := resourceCommandResult(stored, agentToolCredentialResourceType, true); !errors.Is(err, agentmanagement.ErrConflict) {
			t.Fatalf("confused receipt error = %v", err)
		}
	}
}

func testResourceCommand(
	t *testing.T, namespaceID, principalID string, now time.Time,
) managementcommand.Command {
	t.Helper()
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1",
		Keys: map[string][]byte{
			"command-v1": []byte(strings.Repeat("k", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	command, err := codec.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), principalID,
		"/management/v1/agent-tool-credentials", "opaque-key-0123456789",
		[]byte(`{"name":"search-token"}`), now, now.Add(time.Hour),
	)
	if err != nil {
		t.Fatal(err)
	}
	return command
}

func resourceCommandRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"request_digest", "operation_id", "resource_type", "resource_id",
		"resource_revision", "desired_revision", "response_status",
		"secret_response_ciphertext", "secret_response_nonce", "response_kek_version",
		"secret_response_expires_at", "expires_at",
	})
}
