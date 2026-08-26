package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

func TestNewAgentSessionAuthorityOwnsDelegationPepperSnapshot(t *testing.T) {
	const version = "pepper-1"
	store, _ := newMockStore(t)
	borrowed := accesscredential.PepperKeyring{
		ActiveVersion: version,
		Keys:          map[string][]byte{version: bytes.Repeat([]byte{0x5a}, 32)},
	}
	authority, err := NewAgentSessionAuthority(AgentSessionAuthorityOptions{
		Store: store, Management: agentManagementAuthorityStub{}, Peppers: borrowed,
		Secrets: agentSecretCodecStub{}, Waiter: agentPublicationWaiterStub{}, Audience: "inference",
	})
	if err != nil {
		t.Fatalf("NewAgentSessionAuthority() error = %v", err)
	}

	clear(borrowed.Keys[version])
	issued, err := authority.peppers.Issue(accesscredential.KindDelegation, "delegation0001")
	if err != nil {
		t.Fatalf("Issue() error = %v", err)
	}
	independentVerifier := accesscredential.PepperKeyring{
		ActiveVersion: version,
		Keys:          map[string][]byte{version: bytes.Repeat([]byte{0x5a}, 32)},
	}
	if err := independentVerifier.Verify(issued.Plaintext, issued.Digest); err != nil {
		t.Fatalf("Agent authority retained cleared factory key bytes: %v", err)
	}
}

func TestAgentInferenceKeySelectionFiltersBeforeItsSingleRowLimit(t *testing.T) {
	if strings.Contains(eligibleAgentKeyForTargetSQL, "LIMIT $7") ||
		strings.Contains(eligibleAgentKeyForTargetSQL, "LIMIT 256") {
		t.Fatal("Agent inference key selection still truncates candidates before authorization")
	}
	for _, required := range []string{
		"effective_access.subject_id",
		"grant_record.permission=required_permission.value",
		"grant_record.effect='allow'",
		"grant_record.effect='deny'",
		"ORDER BY (k.owner_user_id IS NOT NULL) DESC,k.created_at,k.id LIMIT 1",
	} {
		if !strings.Contains(eligibleAgentKeyForTargetSQL, required) {
			t.Fatalf("Agent inference key SQL is missing %q", required)
		}
	}
}

func TestQueryAgentInferenceKeyUsesOneSQLSideAuthorizationQuery(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectBegin()
	tx, err := store.db.Begin()
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	permissions := []accesscontrol.GrantPermission{
		accesscontrol.GrantPermissionDiscover, accesscontrol.GrantPermissionInvoke,
	}
	mock.ExpectQuery(queryPattern(eligibleAgentKeyForTargetSQL)).
		WithArgs(
			string(testNamespaceID), string(testActorID), nil, true,
			agentmanagement.TargetModel, string(testResourceID),
			pq.Array([]string{"discover", "invoke"}), string(testAPIKeyID),
		).
		WillReturnRows(sqlmock.NewRows([]string{
			"id", "name", "owner_kind", "owner_id", "context_team_id", "expires_at",
			"delegation_epoch", "team_id", "created_at",
		}).AddRow(
			string(testAPIKeyID), "Agent key", "user", string(testUserID), "", nil,
			int64(4), "", testNow,
		))
	key, err := queryAgentInferenceKey(
		context.Background(), tx, string(testNamespaceID), string(testActorID), nil, true,
		string(testAPIKeyID),
		resolvedAgentTarget{Kind: agentmanagement.TargetModel, ResourceID: string(testResourceID)},
		permissions, false,
	)
	if err != nil {
		t.Fatalf("queryAgentInferenceKey() error = %v", err)
	}
	if key.KeyID != string(testAPIKeyID) || key.DelegationEpoch != 4 {
		t.Fatalf("queryAgentInferenceKey() = %#v", key)
	}
	mock.ExpectRollback()
	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("Agent inference key selection performed an unexpected N+1 query: %v", err)
	}
}

func TestAgentInferenceKeySelectionDerivesOmittedTeamFromPinnedKey(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectBegin()
	tx, err := store.db.Begin()
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	permissions := []string{"discover", "invoke"}
	mock.ExpectQuery(queryPattern(eligibleAgentKeyForTargetSQL)).
		WithArgs(
			string(testNamespaceID), string(testActorID), nil, false,
			agentmanagement.TargetEntrypoint, string(testResourceID),
			pq.Array(permissions), string(testAPIKeyID),
		).
		WillReturnRows(sqlmock.NewRows([]string{
			"id", "name", "owner_kind", "owner_id", "context_team_id", "expires_at",
			"delegation_epoch", "team_id", "created_at",
		}).AddRow(
			string(testAPIKeyID), "Team-context key", "user", string(testUserID),
			string(testTeamID), nil, int64(4), string(testTeamID), testNow,
		))
	key, err := selectAgentInferenceKeyRead(
		context.Background(), tx, string(testNamespaceID), string(testActorID),
		string(testAPIKeyID), "", string(testUserID),
		resolvedAgentTarget{Kind: agentmanagement.TargetEntrypoint, ResourceID: string(testResourceID)},
	)
	if err != nil {
		t.Fatalf("selectAgentInferenceKeyRead() error = %v", err)
	}
	if key.KeyID != string(testAPIKeyID) || key.TeamID != string(testTeamID) {
		t.Fatalf("selectAgentInferenceKeyRead() = %#v", key)
	}
	mock.ExpectRollback()
	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("omitted team was not derived from the pinned key: %v", err)
	}
}

func TestAgentInferenceKeySelectionKeepsExplicitTeamAsExactConstraint(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectBegin()
	tx, err := store.db.Begin()
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	mock.ExpectQuery(queryPattern(eligibleAgentKeyForTargetSQL)).
		WithArgs(
			string(testNamespaceID), string(testActorID), string(testTeamID), true,
			agentmanagement.TargetEntrypoint, string(testResourceID),
			pq.Array([]string{"discover", "invoke"}), string(testAPIKeyID),
		).
		WillReturnRows(sqlmock.NewRows([]string{
			"id", "name", "owner_kind", "owner_id", "context_team_id", "expires_at",
			"delegation_epoch", "team_id", "created_at",
		}))
	_, err = selectAgentInferenceKeyRead(
		context.Background(), tx, string(testNamespaceID), string(testActorID),
		string(testAPIKeyID), string(testTeamID), string(testUserID),
		resolvedAgentTarget{Kind: agentmanagement.TargetEntrypoint, ResourceID: string(testResourceID)},
	)
	if !errors.Is(err, agentmanagement.ErrNotFound) {
		t.Fatalf("selectAgentInferenceKeyRead(explicit mismatched team) error = %v, want ErrNotFound", err)
	}
	mock.ExpectRollback()
	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("explicit team did not remain an exact constraint: %v", err)
	}
}

func TestQueryAgentInferenceKeyRejectsEmptyPermissionSetWithoutSQL(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectBegin()
	tx, err := store.db.Begin()
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	_, err = queryAgentInferenceKey(
		context.Background(), tx, string(testNamespaceID), string(testActorID), nil, false,
		nil,
		resolvedAgentTarget{Kind: agentmanagement.TargetModel, ResourceID: string(testResourceID)},
		nil, false,
	)
	if !errors.Is(err, agentmanagement.ErrInvalid) {
		t.Fatalf("queryAgentInferenceKey(empty permissions) error = %v, want ErrInvalid", err)
	}
	mock.ExpectRollback()
	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("empty permission selection touched SQL: %v", err)
	}
}

type agentManagementAuthorityStub struct{}

func (agentManagementAuthorityStub) LoadInTransaction(
	context.Context,
	*sql.Tx,
	accesscontrol.ManagementPrincipalID,
	accesscontrol.NamespaceID,
) (managementauthorization.Snapshot, error) {
	return managementauthorization.Snapshot{}, nil
}

type agentSecretCodecStub struct{}

func (agentSecretCodecStub) Encrypt(
	context.Context, []byte,
) (agentmanagement.EncryptedSecret, error) {
	return agentmanagement.EncryptedSecret{}, nil
}

func (agentSecretCodecStub) Decrypt(
	context.Context, agentmanagement.EncryptedSecret,
) ([]byte, error) {
	return nil, nil
}

type agentPublicationWaiterStub struct{}

func (agentPublicationWaiterStub) WaitActive(
	context.Context, delegationmanagement.Session, uint64,
) error {
	return nil
}

func (agentPublicationWaiterStub) WaitApplied(context.Context, string, string, uint64) error {
	return nil
}
