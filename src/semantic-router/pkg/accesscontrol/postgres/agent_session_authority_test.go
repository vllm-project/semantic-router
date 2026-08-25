package postgres

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

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
