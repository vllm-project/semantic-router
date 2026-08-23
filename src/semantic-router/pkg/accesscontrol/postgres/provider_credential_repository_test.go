package postgres

import (
	"context"
	"database/sql"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testProviderCredentialID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
	testProviderVersionOne   = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
	testProviderVersionTwo   = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
	testProviderCatalog      = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

func TestCreateProviderCredentialPublishesOnlyVersionReference(t *testing.T) {
	store, mock := newMockStore(t)
	credential := testProviderCredential(testProviderVersionOne, 1)
	version := testProviderCredentialVersion(testProviderVersionOne)
	command := testProviderCredentialCommand(t, "/management/v1/provider-credentials", "create-key-0123456789", []byte(`{"name":"Provider"}`))

	mock.ExpectBegin()
	expectNewProviderCredentialCommand(mock, command)
	mock.ExpectQuery(queryPattern(insertProviderCredentialQuery)).
		WithArgs(
			credential.ID, credential.NamespaceID, credential.Name, credential.ProviderID,
			credential.CredentialMode, credential.CredentialAdapterID,
			credential.CatalogRevision, credential.NormalizedOrigin,
			credential.Status, testProviderVersionOne,
			credential.CreatedAt, credential.UpdatedAt,
		).
		WillReturnRows(providerCredentialRows().AddRow(
			credential.ID, credential.NamespaceID, credential.Name, credential.ProviderID,
			credential.CredentialMode, credential.CredentialAdapterID,
			credential.CatalogRevision, credential.NormalizedOrigin,
			credential.Status, testProviderVersionOne,
			int64(1), credential.CreatedAt, credential.UpdatedAt, nil,
		))
	expectProviderCredentialVersionInsert(mock, version)
	expectOutbox(mock, "provider_credential", credential.ID, 1, 20, outboxCreated,
		map[string]string{"versionId": testProviderVersionOne})
	expectProviderCredentialCommandCompletion(mock, command, credential.ID, 1, 201)
	mock.ExpectCommit()

	result, err := store.CreateProviderCredential(context.Background(), credential, version, command, testMutationMeta())
	if err != nil {
		t.Fatalf("create provider credential: %v", err)
	}
	if result.Value.ID != credential.ID || result.Receipt.DesiredRevision != 20 {
		t.Fatalf("unexpected create result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestRotateProviderCredentialPinsBoundedPreviousVersion(t *testing.T) {
	store, mock := newMockStore(t)
	version := testProviderCredentialVersion(testProviderVersionTwo)
	retireAt := testNow.Add(5 * time.Minute)
	command := testProviderCredentialCommand(t, "/management/v1/provider-credentials/"+testProviderCredentialID+":rotate", "rotate-key-0123456789", []byte(`{"secret":"next"}`))

	mock.ExpectBegin()
	expectNewProviderCredentialCommand(mock, command)
	expectProviderCredentialVersionInsert(mock, version)
	mock.ExpectExec(queryPattern(retireProviderCredentialVersionQuery)).
		WithArgs(testNamespaceID, testProviderCredentialID, testProviderVersionOne, retireAt).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(rotateProviderCredentialQuery)).
		WithArgs(testNamespaceID, testProviderCredentialID, int64(1), testProviderVersionOne, testProviderVersionTwo).
		WillReturnRows(providerCredentialRows().AddRow(
			testProviderCredentialID, testNamespaceID, "Provider", "openai", providercredential.ModeRequired, "bearer", testProviderCatalog,
			"https://api.example.com/v1", providercredential.StatusActive, testProviderVersionTwo,
			int64(2), testNow, testNow.Add(time.Second), nil,
		))
	expectOutbox(mock, "provider_credential", testProviderCredentialID, 2, 21, outboxCredentialRotated,
		map[string]string{"versionId": testProviderVersionTwo, "retiredVersionId": testProviderVersionOne})
	expectProviderCredentialCommandCompletion(mock, command, testProviderCredentialID, 2, 200)
	mock.ExpectCommit()

	result, err := store.RotateProviderCredential(
		context.Background(), testNamespaceID, testProviderCredentialID, 1,
		ProviderCredentialRotation{
			Version: version, PreviousVersionID: testProviderVersionOne, RetireAt: retireAt,
		},
		command, testMutationMeta(),
	)
	if err != nil {
		t.Fatalf("rotate provider credential: %v", err)
	}
	if result.Value.ActiveVersionID == nil || *result.Value.ActiveVersionID != testProviderVersionTwo {
		t.Fatalf("unexpected active pointer: %#v", result.Value)
	}
	assertExpectations(t, mock)
}

func TestDisableProviderCredentialCryptoErasesAllVersions(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(disableProviderCredentialQuery)).
		WithArgs(testNamespaceID, testProviderCredentialID, int64(1)).
		WillReturnRows(providerCredentialRows().AddRow(
			testProviderCredentialID, testNamespaceID, "Provider", "openai", providercredential.ModeRequired, "bearer", testProviderCatalog,
			"https://api.example.com/v1", providercredential.StatusDisabled, nil,
			int64(2), testNow, testNow.Add(time.Second), nil,
		))
	mock.ExpectExec(queryPattern(revokeProviderCredentialVersionsQuery)).
		WithArgs(testNamespaceID, testProviderCredentialID).
		WillReturnResult(sqlmock.NewResult(0, 2))
	expectOutbox(mock, "provider_credential", testProviderCredentialID, 2, 22, outboxUpdated, nil)
	mock.ExpectCommit()

	result, err := store.DisableProviderCredential(
		context.Background(), testNamespaceID, testProviderCredentialID, 1, testMutationMeta(),
	)
	if err != nil {
		t.Fatalf("disable provider credential: %v", err)
	}
	if result.Value.Status != providercredential.StatusDisabled || result.Value.ActiveVersionID != nil {
		t.Fatalf("unexpected disabled credential: %#v", result.Value)
	}
	assertExpectations(t, mock)
}

func TestLoadPinnedProviderCredentialNeverFallsBackToActiveVersion(t *testing.T) {
	store, mock := newMockStore(t)
	credential := testProviderCredential(testProviderVersionTwo, 2)
	retireAt := testNow.Add(5 * time.Minute)
	version := testProviderCredentialVersion(testProviderVersionOne)
	version.Status = providercredential.VersionRetiring

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(getProviderCredentialByIDQuery)).
		WithArgs(testProviderCredentialID).
		WillReturnRows(providerCredentialRows().AddRow(
			credential.ID, credential.NamespaceID, credential.Name, credential.ProviderID,
			credential.CredentialMode, credential.CredentialAdapterID, credential.CatalogRevision,
			credential.NormalizedOrigin, credential.Status, testProviderVersionTwo,
			int64(2), credential.CreatedAt, credential.UpdatedAt, nil,
		))
	mock.ExpectQuery(queryPattern(getProviderCredentialVersionQuery)).
		WithArgs(testProviderCredentialID, testProviderVersionOne).
		WillReturnRows(providerCredentialVersionRows().AddRow(
			version.ID, version.NamespaceID, version.CredentialID,
			version.Envelope.Ciphertext, version.Envelope.Nonce, version.Envelope.KeyVersion,
			version.Status, version.NotBefore, retireAt, nil, version.CreatedAt,
		))
	mock.ExpectCommit()

	loadedCredential, loadedVersion, err := store.LoadPinnedProviderCredential(
		context.Background(), testProviderCredentialID, testProviderVersionOne,
	)
	if err != nil {
		t.Fatalf("load pinned provider credential: %v", err)
	}
	if loadedCredential.ActiveVersionID == nil || *loadedCredential.ActiveVersionID != testProviderVersionTwo ||
		loadedVersion.ID != testProviderVersionOne {
		t.Fatalf("loader substituted active version: credential=%#v version=%#v", loadedCredential, loadedVersion)
	}
	assertExpectations(t, mock)
}

func TestListProviderCredentialsUsesStableKeysetAndBoundedPage(t *testing.T) {
	store, mock := newMockStore(t)
	rows := providerCredentialRows()
	for _, id := range []string{
		"dddddddd-dddd-4ddd-8ddd-dddddddddddd",
		"eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
		"ffffffff-ffff-4fff-8fff-ffffffffffff",
	} {
		credential := testProviderCredential(testProviderVersionOne, 1)
		credential.ID = id
		rows.AddRow(
			credential.ID, credential.NamespaceID, credential.Name, credential.ProviderID,
			credential.CredentialMode, credential.CredentialAdapterID,
			credential.CatalogRevision, credential.NormalizedOrigin,
			credential.Status, testProviderVersionOne, int64(1),
			credential.CreatedAt, credential.UpdatedAt, nil,
		)
	}
	mock.ExpectQuery(queryPattern(listProviderCredentialsQuery)).
		WithArgs(
			testNamespaceID, "openai", providercredential.StatusActive,
			true, sqlmock.AnyArg(), providercredential.StatusActive, testProviderCredentialID, 3,
		).
		WillReturnRows(rows)
	page, err := store.ListProviderCredentials(context.Background(), testNamespaceID, ProviderCredentialListRequest{
		ProviderID: "openai", Status: providercredential.StatusActive,
		AfterStatus: providercredential.StatusActive, AfterID: testProviderCredentialID, PageSize: 2,
		Scope: accesscontrol.ResultScope{NamespaceID: testNamespaceID, All: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Credentials) != 2 || !page.HasMore ||
		page.Credentials[0].ID != "dddddddd-dddd-4ddd-8ddd-dddddddddddd" ||
		page.Credentials[1].ID != "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee" {
		t.Fatalf("page = %#v", page)
	}
	assertExpectations(t, mock)
}

func testProviderCredential(activeVersion string, revision uint64) providercredential.Credential {
	return providercredential.Credential{
		ID: testProviderCredentialID, NamespaceID: string(testNamespaceID), Name: "Provider",
		ProviderID: "openai", CredentialMode: providercredential.ModeRequired,
		CredentialAdapterID: "bearer", CatalogRevision: testProviderCatalog,
		NormalizedOrigin: "https://api.example.com/v1",
		Status:           providercredential.StatusActive, ActiveVersionID: &activeVersion, Revision: revision,
		CreatedAt: testNow, UpdatedAt: testNow,
	}
}

func testProviderCredentialVersion(id string) providercredential.Version {
	return providercredential.Version{
		ID: id, NamespaceID: string(testNamespaceID), CredentialID: testProviderCredentialID,
		Envelope: accesscredential.Envelope{
			Ciphertext: []byte("encrypted-provider-secret"), Nonce: []byte("nonce"), KeyVersion: "provider-kek-v1",
		},
		Status: providercredential.VersionActive, NotBefore: testNow, CreatedAt: testNow,
	}
}

func expectProviderCredentialVersionInsert(mock sqlmock.Sqlmock, version providercredential.Version) {
	mock.ExpectExec(queryPattern(insertProviderCredentialVersionQuery)).
		WithArgs(
			version.ID, version.NamespaceID, version.CredentialID,
			version.Envelope.Ciphertext, version.Envelope.Nonce, version.Envelope.KeyVersion,
			version.Status, version.NotBefore, nil, nil, version.CreatedAt,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
}

func providerCredentialRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "name", "provider_id", "credential_mode", "credential_adapter_id", "provider_catalog_revision", "normalized_origin",
		"status", "active_version_id", "revision", "created_at", "updated_at", "deleted_at",
	})
}

func testProviderCredentialCommand(t *testing.T, endpoint, key string, request []byte) managementcommand.Command {
	t.Helper()
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{"command-v1": []byte(strings.Repeat("i", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	command, err := codec.Bind(managementcommand.NamespaceCommandScope(string(testNamespaceID)), string(testActorID), endpoint, key, request, now, now.Add(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	return command
}

func expectNewProviderCredentialCommand(mock sqlmock.Sqlmock, command managementcommand.Command) {
	active := command.ActiveDigest()
	mock.ExpectQuery(`(?s)SELECT clock_timestamp\(\).*pg_advisory_xact_lock`).
		WithArgs(sqlmock.AnyArg()).
		WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(time.Now().UTC()))
	mock.ExpectQuery(`(?s)SELECT request_digest.*FROM management_idempotency.*FOR UPDATE`).
		WithArgs(string(managementcommand.ScopeNamespace), command.Scope.NamespaceID, command.PrincipalID, command.Endpoint, active.HMACVersion, active.KeyDigest[:]).
		WillReturnError(sql.ErrNoRows)
}

func expectProviderCredentialCommandCompletion(mock sqlmock.Sqlmock, command managementcommand.Command, resourceID string, revision uint64, status int) {
	active := command.ActiveDigest()
	mock.ExpectExec(`(?s)INSERT INTO management_idempotency.*VALUES`).
		WithArgs(
			string(managementcommand.ScopeNamespace), command.Scope.NamespaceID, command.PrincipalID, command.Endpoint,
			active.HMACVersion, active.KeyDigest[:], active.RequestDigest[:], "provider_credential",
			resourceID, revision, status, command.ExpiresAt,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
}

func providerCredentialVersionRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "provider_credential_id", "secret_ciphertext",
		"ciphertext_nonce", "kek_version", "status", "not_before",
		"expires_at", "revoked_at", "created_at",
	})
}

var _ ProviderCredentialRepository = (*Store)(nil)
