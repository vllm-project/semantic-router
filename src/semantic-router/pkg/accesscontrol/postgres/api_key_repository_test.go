package postgres

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestCreateAPIKeyStoresCredentialAndSafeOutboxAtomically(t *testing.T) {
	store, mock := newMockStore(t)
	key := testAPIKey(1)
	credential := testCredential(testCredentialID)

	mock.ExpectBegin()
	mock.ExpectExec(queryPattern(insertSubjectQuery)).
		WithArgs(testNamespaceID, testAPIKeyID, accesscontrol.SubjectKindAPIKey, testNow).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(insertAPIKeyQuery)).
		WithArgs(
			key.ID, key.NamespaceID, key.Name, testUserID, nil, nil,
			key.Status, nil, key.PolicyEpoch, key.DelegationEpoch, key.Revision,
			key.CreatedAt, key.UpdatedAt,
		).
		WillReturnRows(apiKeyRows().AddRow(
			key.ID, key.NamespaceID, key.Name, testUserID, nil, nil,
			key.Status, nil, int64(1), int64(1), int64(1),
			nil, key.CreatedAt, key.UpdatedAt, nil,
		))
	expectCredentialInsert(mock, credential)
	expectOutbox(
		mock, "api_key", string(key.ID), 1, 4, outboxCreated,
		map[string]string{"credentialId": string(credential.ID)},
	)
	mock.ExpectCommit()

	result, err := store.CreateAPIKey(context.Background(), key, credential, testMutationMeta())
	if err != nil {
		t.Fatalf("create API key: %v", err)
	}
	if result.Value.ID != key.ID || result.Value.Revision != 1 || result.Receipt.DesiredRevision != 4 {
		t.Fatalf("unexpected API-key result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestRotateCredentialAdvancesLogicalKeyCASAndPublishesOnlyReferences(t *testing.T) {
	store, mock := newMockStore(t)
	newCredential := testCredential(testCredentialID)
	oldCredentialID := accesscontrol.CredentialVersionID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
	retireAt := testNow.Add(10 * time.Minute)
	rotation := CredentialRotation{
		Credential: newCredential, RetireCredentialID: &oldCredentialID, RetireAt: &retireAt,
	}

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(advanceAPIKeyRevisionQuery)).
		WithArgs(testNamespaceID, testAPIKeyID, int64(1)).
		WillReturnRows(apiKeyRows().AddRow(
			testAPIKeyID, testNamespaceID, "key", testUserID, nil, nil,
			accesscontrol.APIKeyStatusActive, nil, int64(1), int64(1), int64(2),
			nil, testNow, testNow.Add(time.Second), nil,
		))
	mock.ExpectExec(queryPattern(retireCredentialQuery)).
		WithArgs(testNamespaceID, testAPIKeyID, oldCredentialID, retireAt).
		WillReturnResult(sqlmock.NewResult(0, 1))
	expectCredentialInsert(mock, newCredential)
	expectOutbox(
		mock, "api_key", string(testAPIKeyID), 2, 5, outboxCredentialRotated,
		map[string]string{
			"credentialId":        string(newCredential.ID),
			"retiredCredentialId": string(oldCredentialID),
		},
	)
	mock.ExpectCommit()

	result, err := store.RotateCredential(
		context.Background(), testNamespaceID, testAPIKeyID, 1, rotation, testMutationMeta(),
	)
	if err != nil {
		t.Fatalf("rotate credential: %v", err)
	}
	if result.Value.Revision != 2 || result.Receipt.DesiredRevision != 5 {
		t.Fatalf("unexpected rotation result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestSoftDeleteAPIKeyRevokesCredentialsAndCryptoErases(t *testing.T) {
	store, mock := newMockStore(t)
	deletedAt := testNow.Add(time.Minute)

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(softDeleteAPIKeyQuery)).
		WithArgs(testNamespaceID, testAPIKeyID, int64(1)).
		WillReturnRows(apiKeyRows().AddRow(
			testAPIKeyID, testNamespaceID, "key", testUserID, nil, nil,
			accesscontrol.APIKeyStatusDisabled, nil, int64(2), int64(2), int64(2),
			nil, testNow, deletedAt, deletedAt,
		))
	mock.ExpectExec(queryPattern(revokeAllCredentialsQuery)).
		WithArgs(testNamespaceID, testAPIKeyID).
		WillReturnResult(sqlmock.NewResult(0, 2))
	expectOutbox(mock, "api_key", string(testAPIKeyID), 2, 10, outboxDeleted, nil)
	mock.ExpectCommit()

	result, err := store.SoftDeleteAPIKey(
		context.Background(), testNamespaceID, testAPIKeyID, 1, testMutationMeta(),
	)
	if err != nil {
		t.Fatalf("soft-delete API key: %v", err)
	}
	if result.Value.Status != accesscontrol.APIKeyStatusDeleted || result.Value.DeletedAt == nil {
		t.Fatalf("expected logical-key tombstone, got %#v", result.Value)
	}
	if result.Value.PolicyEpoch != 2 || result.Value.DelegationEpoch != 2 {
		t.Fatalf("expected invalidation epochs to advance, got %#v", result.Value)
	}
	assertExpectations(t, mock)
}

func testAPIKey(revision accesscontrol.Revision) accesscontrol.APIKey {
	return accesscontrol.APIKey{
		NamespaceID: testNamespaceID,
		ID:          testAPIKeyID,
		Name:        "key",
		Owner: accesscontrol.SubjectRef{
			NamespaceID: testNamespaceID,
			ID:          accesscontrol.SubjectID(testUserID),
			Kind:        accesscontrol.SubjectKindUser,
		},
		Status:          accesscontrol.APIKeyStatusActive,
		PolicyEpoch:     1,
		DelegationEpoch: 1,
		Revision:        revision,
		CreatedAt:       testNow,
		UpdatedAt:       testNow,
	}
}

func testCredential(id accesscontrol.CredentialVersionID) accesscontrol.CredentialVersion {
	return accesscontrol.CredentialVersion{
		ID:               id,
		APIKeyID:         testAPIKeyID,
		KID:              "kid-123456789",
		SecretHMAC:       []byte("hmac-material"),
		PepperVersion:    "pepper-v1",
		SecretCiphertext: []byte("wrapped-secret"),
		CiphertextNonce:  []byte("nonce"),
		KEKVersion:       "kek-v1",
		Status:           accesscontrol.CredentialStatusActive,
		NotBefore:        testNow,
		CreatedAt:        testNow,
	}
}

func expectCredentialInsert(mock sqlmock.Sqlmock, credential accesscontrol.CredentialVersion) {
	mock.ExpectExec(queryPattern(insertCredentialQuery)).
		WithArgs(
			credential.ID, testNamespaceID, credential.APIKeyID, credential.KID,
			credential.SecretHMAC, credential.PepperVersion,
			credential.SecretCiphertext, credential.CiphertextNonce, credential.KEKVersion,
			credential.Status, credential.NotBefore, nil, nil, credential.CreatedAt,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
}

func apiKeyRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "name", "owner_user_id", "owner_team_id",
		"context_team_id", "status", "expires_at", "policy_epoch", "delegation_epoch",
		"revision", "last_used_at", "created_at", "updated_at", "deleted_at",
	})
}
