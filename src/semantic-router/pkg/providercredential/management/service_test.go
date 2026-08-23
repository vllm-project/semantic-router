package management

import (
	"bytes"
	"context"
	"errors"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	serviceNamespaceID  = "11111111-1111-4111-8111-111111111111"
	servicePrincipalID  = "22222222-2222-4222-8222-222222222222"
	serviceCredentialID = "33333333-3333-4333-8333-333333333333"
	serviceVersionOne   = "44444444-4444-4444-8444-444444444444"
	serviceVersionTwo   = "55555555-5555-4555-8555-555555555555"
	serviceVersionThree = "66666666-6666-4666-8666-666666666666"
	serviceCatalogRev   = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

func TestServiceReadinessFailsClosedForMissingCommandKeyVersion(t *testing.T) {
	service, repository, _ := newTestService(t, fixedProvider())
	repository.readinessErr = managementcommand.ErrHMACVersionUnavailable
	if err := service.Ready(context.Background()); !errors.Is(err, ErrUnavailable) || !errors.Is(err, managementcommand.ErrHMACVersionUnavailable) {
		t.Fatalf("Ready() error = %v", err)
	}
	repository.readinessErr = nil
	if err := service.Ready(context.Background()); err != nil {
		t.Fatalf("Ready() after key retention = %v", err)
	}
}

func TestServiceOwnsCompleteCredentialLifecycle(t *testing.T) {
	service, repository, catalog := newTestService(t, fixedProvider())
	ctx := context.Background()
	created, err := service.Create(ctx, createServiceRequest())
	if err != nil {
		t.Fatal(err)
	}
	if created.CredentialID != serviceCredentialID || created.Revision != 1 || created.Replayed {
		t.Fatalf("create receipt = %#v", created)
	}
	if catalog.calls != 1 || repository.credential.CredentialMode != providercredential.ModeRequired {
		t.Fatalf("catalog calls=%d credential=%#v", catalog.calls, repository.credential)
	}
	initialVersion := repository.versions[serviceVersionOne]
	if len(initialVersion.Envelope.Ciphertext) == 0 ||
		bytes.Contains(initialVersion.Envelope.Ciphertext, []byte("provider-secret")) {
		t.Fatal("provider secret was not envelope encrypted")
	}

	renamed, err := service.Rename(ctx, RenameRequest{
		NamespaceID: serviceNamespaceID, CredentialID: serviceCredentialID,
		ExpectedRevision: 1, Name: "Primary provider", Actor: serviceActor(),
	})
	if err != nil || renamed.Revision != 2 {
		t.Fatalf("Rename() = %#v, %v", renamed, err)
	}
	disabled, err := service.Disable(ctx, LifecycleRequest{
		NamespaceID: serviceNamespaceID, CredentialID: serviceCredentialID,
		ExpectedRevision: 2, Actor: serviceActor(),
	})
	if err != nil || disabled.Revision != 3 || repository.credential.Status != providercredential.StatusDisabled {
		t.Fatalf("Disable() = %#v, %v, credential=%#v", disabled, err, repository.credential)
	}
	reactivated, err := service.Reactivate(ctx, LifecycleRequest{
		NamespaceID: serviceNamespaceID, CredentialID: serviceCredentialID,
		ExpectedRevision: 3, Secret: []byte("fresh-secret"), Actor: serviceActor(),
	})
	if err != nil || reactivated.Revision != 4 || repository.credential.Status != providercredential.StatusActive {
		t.Fatalf("Reactivate() = %#v, %v, credential=%#v", reactivated, err, repository.credential)
	}
	rotated, err := service.Rotate(ctx, RotateRequest{
		NamespaceID: serviceNamespaceID, CredentialID: serviceCredentialID,
		ExpectedRevision: 4, Secret: []byte("rotated-secret"),
		IdempotencyKey: "rotate-key-0123456789", Actor: serviceActor(),
	})
	if err != nil || rotated.Revision != 5 {
		t.Fatalf("Rotate() = %#v, %v", rotated, err)
	}
	if repository.lastRotation.RetireAt.Sub(repository.lastRotation.Version.NotBefore) != 30*time.Second {
		t.Fatalf("rotation overlap = %s", repository.lastRotation.RetireAt.Sub(repository.lastRotation.Version.NotBefore))
	}
	deleted, err := service.Delete(ctx, LifecycleRequest{
		NamespaceID: serviceNamespaceID, CredentialID: serviceCredentialID,
		ExpectedRevision: 5, Actor: serviceActor(),
	})
	if err != nil || deleted.Revision != 6 || repository.credential.Status != providercredential.StatusDeleted {
		t.Fatalf("Delete() = %#v, %v, credential=%#v", deleted, err, repository.credential)
	}
}

func TestCreateReplayAfterPatchReturnsOriginalImmutableReceipt(t *testing.T) {
	service, _, catalog := newTestService(t, fixedProvider())
	request := createServiceRequest()
	created, createErr := service.Create(context.Background(), request)
	if createErr != nil {
		t.Fatal(createErr)
	}
	if _, err := service.Rename(context.Background(), RenameRequest{
		NamespaceID: serviceNamespaceID, CredentialID: serviceCredentialID,
		ExpectedRevision: 1, Name: "Renamed after create", Actor: serviceActor(),
	}); err != nil {
		t.Fatal(err)
	}
	replayed, createErr := service.Create(context.Background(), request)
	if createErr != nil {
		t.Fatal(createErr)
	}
	if !replayed.Replayed || replayed.CredentialID != created.CredentialID || replayed.Revision != 1 {
		t.Fatalf("replayed receipt = %#v; created = %#v", replayed, created)
	}
	if catalog.calls != 1 {
		t.Fatalf("replay consulted mutable catalog %d times", catalog.calls)
	}
}

func TestServiceRejectsUnsafeOriginAndCatalogBindingDrift(t *testing.T) {
	userSupplied := fixedProvider()
	userSupplied.Provider.Origin = providercatalog.Origin{
		Mode: providercatalog.OriginUserSupplied, Label: "Base URL",
	}
	service, _, _ := newTestService(t, userSupplied)
	service.egress = egressStub{denied: true}
	request := createServiceRequest()
	request.BaseURL = "https://denied.example.com/v1"
	if _, err := service.Create(context.Background(), request); !errors.Is(err, ErrUnsafeOrigin) {
		t.Fatalf("unsafe origin error = %v", err)
	}

	service, repository, catalog := newTestService(t, fixedProvider())
	if _, err := service.Create(context.Background(), createServiceRequest()); err != nil {
		t.Fatal(err)
	}
	drifted := fixedProvider()
	drifted.Provider.Credential.Mode = providercatalog.CredentialOptional
	catalog.detail = drifted
	if _, err := service.Rotate(context.Background(), RotateRequest{
		NamespaceID: serviceNamespaceID, CredentialID: repository.credential.ID,
		ExpectedRevision: 1, Secret: []byte("rotated-secret"),
		IdempotencyKey: "mode-drift-0123456789", Actor: serviceActor(),
	}); !errors.Is(err, ErrProviderMismatch) {
		t.Fatalf("credential-mode drift error = %v", err)
	}

	request = createServiceRequest()
	request.IdempotencyKey = "fixed-override-0123456789"
	request.BaseURL = "https://other.example.com"
	if _, err := service.Create(context.Background(), request); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("fixed origin override error = %v", err)
	}

	mismatched := fixedProvider()
	mismatched.Provider.ID = "provider-b"
	service, _, _ = newTestService(t, mismatched)
	request = createServiceRequest()
	request.IdempotencyKey = "provider-mismatch-0123456789"
	if _, err := service.Create(context.Background(), request); !errors.Is(err, ErrProviderMismatch) {
		t.Fatalf("catalog provider mismatch error = %v", err)
	}
}

type catalogStub struct {
	detail providercatalog.DetailResult
	calls  int
}

func (catalog *catalogStub) Get(_ context.Context, _ string) (providercatalog.DetailResult, error) {
	catalog.calls++
	return catalog.detail, nil
}

type egressStub struct{ denied bool }

func (egress egressStub) AuthorizeOrigin(origin string) (backendegress.Target, error) {
	if egress.denied {
		return backendegress.Target{}, errors.New("denied")
	}
	return backendegress.Target{Origin: origin}, nil
}

type storedCommand struct {
	requestDigest [32]byte
	result        managementcommand.ResourceResult
}

type commandIdentity struct {
	version string
	digest  [32]byte
}

type memoryRepository struct {
	credential   providercredential.Credential
	versions     map[string]providercredential.Version
	commands     map[commandIdentity]storedCommand
	lastRotation accesspostgres.ProviderCredentialRotation
	readinessErr error
}

func (repository *memoryRepository) ValidateManagementCommandHMACVersions(_ context.Context, _ *managementcommand.Codec) error {
	return repository.readinessErr
}

func (repository *memoryRepository) GetProviderCredential(_ context.Context, namespaceID accesscontrol.NamespaceID, id string) (providercredential.Credential, error) {
	if repository.credential.ID != id || repository.credential.NamespaceID != string(namespaceID) {
		return providercredential.Credential{}, accesspostgres.ErrNotFound
	}
	return repository.credential, nil
}

func (repository *memoryRepository) ListProviderCredentials(_ context.Context, namespaceID accesscontrol.NamespaceID, request accesspostgres.ProviderCredentialListRequest) (accesspostgres.ProviderCredentialListResult, error) {
	if repository.credential.ID == "" || repository.credential.NamespaceID != string(namespaceID) ||
		request.ProviderID != "" && request.ProviderID != repository.credential.ProviderID ||
		request.Status != "" && request.Status != repository.credential.Status {
		return accesspostgres.ProviderCredentialListResult{}, nil
	}
	return accesspostgres.ProviderCredentialListResult{Credentials: []providercredential.Credential{repository.credential}}, nil
}

func (repository *memoryRepository) ReplayProviderCredentialCommand(_ context.Context, command managementcommand.Command) (accesspostgres.MutationResult[providercredential.Credential], bool, error) {
	for _, candidate := range command.CandidateDigests() {
		stored, found := repository.commands[commandIdentity{version: candidate.HMACVersion, digest: candidate.KeyDigest}]
		if !found {
			continue
		}
		if !command.SameRequest(candidate.HMACVersion, stored.requestDigest[:]) {
			return accesspostgres.MutationResult[providercredential.Credential]{}, false, managementcommand.ErrConflict
		}
		return replayMutation(stored.result), true, nil
	}
	return accesspostgres.MutationResult[providercredential.Credential]{}, false, nil
}

func (repository *memoryRepository) CreateProviderCredential(_ context.Context, credential providercredential.Credential, version providercredential.Version, command managementcommand.Command, _ accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error) {
	if replay, found, err := repository.ReplayProviderCredentialCommand(context.Background(), command); found || err != nil {
		return replay, err
	}
	repository.credential = credential
	repository.versions[version.ID] = version
	stored := managementcommand.ResourceResult{
		ResourceType: "provider_credential", ResourceID: credential.ID,
		ResourceRevision: credential.Revision, ResponseStatus: 201,
	}
	active := command.ActiveDigest()
	repository.commands[commandIdentity{version: active.HMACVersion, digest: active.KeyDigest}] = storedCommand{requestDigest: active.RequestDigest, result: stored}
	return accesspostgres.MutationResult[providercredential.Credential]{
		Value: credential, ResourceID: credential.ID,
		ResourceRevision: accesscontrol.Revision(credential.Revision), ResponseStatus: 201,
	}, nil
}

func (repository *memoryRepository) RenameProviderCredential(_ context.Context, namespaceID accesscontrol.NamespaceID, id string, expected accesscontrol.Revision, name string, _ accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error) {
	if err := repository.checkCAS(namespaceID, id, expected); err != nil {
		return accesspostgres.MutationResult[providercredential.Credential]{}, err
	}
	repository.credential.Name = name
	return repository.advance(), nil
}

func (repository *memoryRepository) RotateProviderCredential(_ context.Context, namespaceID accesscontrol.NamespaceID, id string, expected accesscontrol.Revision, rotation accesspostgres.ProviderCredentialRotation, command managementcommand.Command, _ accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error) {
	if replay, found, err := repository.ReplayProviderCredentialCommand(context.Background(), command); found || err != nil {
		return replay, err
	}
	if err := repository.checkCAS(namespaceID, id, expected); err != nil {
		return accesspostgres.MutationResult[providercredential.Credential]{}, err
	}
	repository.lastRotation = rotation
	repository.versions[rotation.Version.ID] = rotation.Version
	repository.credential.ActiveVersionID = &rotation.Version.ID
	result := repository.advance()
	stored := managementcommand.ResourceResult{
		ResourceType: "provider_credential", ResourceID: id,
		ResourceRevision: uint64(result.ResourceRevision), ResponseStatus: 200,
	}
	active := command.ActiveDigest()
	repository.commands[commandIdentity{version: active.HMACVersion, digest: active.KeyDigest}] = storedCommand{requestDigest: active.RequestDigest, result: stored}
	result.ResponseStatus = 200
	return result, nil
}

func (repository *memoryRepository) ReactivateProviderCredential(_ context.Context, namespaceID accesscontrol.NamespaceID, id string, expected accesscontrol.Revision, version providercredential.Version, _ accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error) {
	if err := repository.checkCAS(namespaceID, id, expected); err != nil {
		return accesspostgres.MutationResult[providercredential.Credential]{}, err
	}
	repository.versions[version.ID] = version
	repository.credential.Status = providercredential.StatusActive
	repository.credential.ActiveVersionID = &version.ID
	return repository.advance(), nil
}

func (repository *memoryRepository) DisableProviderCredential(_ context.Context, namespaceID accesscontrol.NamespaceID, id string, expected accesscontrol.Revision, _ accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error) {
	if err := repository.checkCAS(namespaceID, id, expected); err != nil {
		return accesspostgres.MutationResult[providercredential.Credential]{}, err
	}
	repository.credential.Status = providercredential.StatusDisabled
	repository.credential.ActiveVersionID = nil
	return repository.advance(), nil
}

func (repository *memoryRepository) DeleteProviderCredential(_ context.Context, namespaceID accesscontrol.NamespaceID, id string, expected accesscontrol.Revision, _ accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error) {
	if err := repository.checkCAS(namespaceID, id, expected); err != nil {
		return accesspostgres.MutationResult[providercredential.Credential]{}, err
	}
	repository.credential.Status = providercredential.StatusDeleted
	repository.credential.ActiveVersionID = nil
	now := time.Now().UTC()
	repository.credential.DeletedAt = &now
	return repository.advance(), nil
}

func (repository *memoryRepository) checkCAS(namespaceID accesscontrol.NamespaceID, id string, expected accesscontrol.Revision) error {
	if repository.credential.NamespaceID != string(namespaceID) || repository.credential.ID != id ||
		repository.credential.Revision != uint64(expected) {
		return accesspostgres.ErrRevisionConflict
	}
	return nil
}

func (repository *memoryRepository) advance() accesspostgres.MutationResult[providercredential.Credential] {
	repository.credential.Revision++
	repository.credential.UpdatedAt = time.Now().UTC()
	return accesspostgres.MutationResult[providercredential.Credential]{
		Value: repository.credential, ResourceID: repository.credential.ID,
		ResourceRevision: accesscontrol.Revision(repository.credential.Revision),
	}
}

func replayMutation(stored managementcommand.ResourceResult) accesspostgres.MutationResult[providercredential.Credential] {
	return accesspostgres.MutationResult[providercredential.Credential]{
		ResourceID: stored.ResourceID, ResourceRevision: accesscontrol.Revision(stored.ResourceRevision),
		Replayed: true, ResponseStatus: stored.ResponseStatus,
	}
}

func newTestService(t *testing.T, detail providercatalog.DetailResult) (*Service, *memoryRepository, *catalogStub) {
	t.Helper()
	repository := &memoryRepository{
		versions: make(map[string]providercredential.Version), commands: make(map[commandIdentity]storedCommand),
	}
	catalog := &catalogStub{detail: detail}
	ids := []string{serviceCredentialID, serviceVersionOne, serviceVersionTwo, serviceVersionThree}
	index := 0
	now := time.Now().UTC().Truncate(time.Second)
	commandCodec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{"command-v1": []byte(strings.Repeat("i", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		Repository: repository, Catalog: catalog, Egress: egressStub{},
		CredentialCodec: providercredential.Codec{Keyring: accesscredential.KEKKeyring{
			ActiveVersion: "provider-kek-v1", Keys: map[string][]byte{"provider-kek-v1": []byte(strings.Repeat("p", 32))},
		}},
		CommandCodec: commandCodec, CursorKeyring: securitykeyring.Symmetric{
			ActiveVersion: "cursor-v1", Keys: map[string][]byte{"cursor-v1": []byte(strings.Repeat("c", 32))},
		},
		IdempotencyTTL: time.Hour, RetiringOverlap: 30 * time.Second,
		Now: func() time.Time { return now }, NewID: func() string {
			if index >= len(ids) {
				t.Fatal("test exhausted deterministic IDs")
			}
			value := ids[index]
			index++
			return value
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return service, repository, catalog
}

func fixedProvider() providercatalog.DetailResult {
	return providercatalog.DetailResult{
		CatalogRevision: serviceCatalogRev,
		Provider: providercatalog.Definition{
			ID: "provider-a",
			Credential: providercatalog.Credential{
				Mode: providercatalog.CredentialRequired, AdapterID: "bearer",
			},
			Origin: providercatalog.Origin{
				Mode: providercatalog.OriginFixed, DefaultURL: "https://api.example.com/v1",
			},
		},
	}
}

func createServiceRequest() CreateRequest {
	return CreateRequest{
		NamespaceID: serviceNamespaceID, Name: "Provider key", ProviderID: "provider-a",
		Secret: []byte("provider-secret"), IdempotencyKey: "create-key-0123456789", Actor: serviceActor(),
	}
}

func serviceActor() Actor {
	return Actor{
		PrincipalID: servicePrincipalID, RequestID: "request-1",
		SourceIP: netip.MustParseAddr("192.0.2.10"),
	}
}
