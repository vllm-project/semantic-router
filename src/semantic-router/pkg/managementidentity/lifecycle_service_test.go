package managementidentity

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/issuerverifier"
)

const (
	lifecycleTestIssuerID  = "80000000-0000-4000-8000-000000000001"
	lifecycleTestSessionID = "80000000-0000-4000-8000-000000000002"
	lifecycleTestActorID   = "80000000-0000-4000-8000-000000000003"
)

type lifecycleRepositoryStub struct {
	LifecycleRepository
	steps   *[]string
	current TrustedIdentityIssuer
	updated IssuerMutation
}

func (repository lifecycleRepositoryStub) GetTrustedIdentityIssuer(context.Context, string) (TrustedIdentityIssuer, error) {
	*repository.steps = append(*repository.steps, "get")
	return repository.current, nil
}

func (repository lifecycleRepositoryStub) UpdateTrustedIdentityIssuer(context.Context, UpdateTrustedIdentityIssuer) (IssuerMutation, error) {
	*repository.steps = append(*repository.steps, "update")
	return repository.updated, nil
}

type lifecycleBarrierStub struct{ steps *[]string }

func (barrier lifecycleBarrierStub) Check(context.Context, managementauth.BarrierCheck) (managementauth.BarrierState, error) {
	return managementauth.BarrierState{Ready: true}, nil
}

func (barrier lifecycleBarrierStub) InstallDeny(_ context.Context, kind managementauth.BarrierKind, id string) error {
	*barrier.steps = append(*barrier.steps, "install:"+string(kind)+":"+id)
	return nil
}

func (barrier lifecycleBarrierStub) RemoveDeny(_ context.Context, kind managementauth.BarrierKind, id string) error {
	*barrier.steps = append(*barrier.steps, "remove:"+string(kind)+":"+id)
	return nil
}
func (lifecycleBarrierStub) Ready(context.Context) error { return nil }

type lifecycleKeyCacheStub struct{ steps *[]string }

func (cache lifecycleKeyCacheStub) Invalidate(string) {
	*cache.steps = append(*cache.steps, "invalidate")
}

func (cache lifecycleKeyCacheStub) Refresh(context.Context, issuerverifier.TrustedIssuer) error {
	*cache.steps = append(*cache.steps, "refresh")
	return nil
}

type lifecycleSessionRepositoryStub struct {
	managementauth.SessionRepository
}
type lifecyclePolicyLoaderStub struct {
	managementauth.SessionPolicyLoader
}

type lifecycleLogoutVerifierStub struct{}

func (lifecycleLogoutVerifierStub) VerifyBackchannelLogout(context.Context, string, string, time.Time) (managementauth.BackchannelLogoutIdentity, error) {
	return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
}

func TestLifecycleIssuerUpdateMaintainsFailClosedBarrierOrder(t *testing.T) {
	steps := []string{}
	current := lifecycleTestIssuer(managementauth.ResourceActive, 7)
	updated := current
	updated.DiscoveryURL = "https://issuer.example/new-discovery"
	updated.Revision = 8
	repository := lifecycleRepositoryStub{
		steps: &steps, current: current,
		updated: IssuerMutation{
			Issuer: updated, Sessions: []string{lifecycleTestSessionID},
			Result: MutationResult{Kind: "trusted_identity_issuer", ID: lifecycleTestIssuerID, Revision: 8, ResponseStatus: 200},
		},
	}
	barriers := lifecycleBarrierStub{steps: &steps}
	service, err := NewLifecycleService(
		repository,
		managementauth.SessionRuntime{
			Sessions: lifecycleSessionRepositoryStub{}, Barriers: barriers,
			PolicyLoader: lifecyclePolicyLoaderStub{},
		},
		barriers, lifecycleKeyCacheStub{steps: &steps}, lifecycleLogoutVerifierStub{},
	)
	if err != nil {
		t.Fatal(err)
	}
	newDiscovery := updated.DiscoveryURL
	result, err := service.UpdateTrustedIdentityIssuer(context.Background(), UpdateTrustedIdentityIssuer{
		ID: lifecycleTestIssuerID, ExpectedRevision: 7, DiscoveryURL: &newDiscovery,
		Actor: MutationActor{PrincipalID: lifecycleTestActorID, RequestID: "request-1", Reason: "Rotate issuer metadata"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Issuer.Revision != 8 {
		t.Fatalf("issuer result = %+v", result.Issuer)
	}
	want := []string{
		"get",
		"install:authentication_source:issuer:" + lifecycleTestIssuerID,
		"update",
		"invalidate",
		"install:management_session:" + lifecycleTestSessionID,
		"refresh",
		"remove:authentication_source:issuer:" + lifecycleTestIssuerID,
	}
	if !reflect.DeepEqual(steps, want) {
		t.Fatalf("lifecycle steps = %v, want %v", steps, want)
	}
}

func lifecycleTestIssuer(status managementauth.ResourceStatus, revision uint64) TrustedIdentityIssuer {
	now := time.Date(2026, 8, 23, 4, 5, 6, 0, time.UTC)
	return TrustedIdentityIssuer{
		ID: lifecycleTestIssuerID, Issuer: "https://issuer.example", Kind: issuerverifier.IssuerOIDC,
		DiscoveryURL: "https://issuer.example/.well-known/openid-configuration",
		Audiences:    []string{issuerverifier.ManagementAudience},
		ClaimMapping: map[string]string{}, AssuranceMapping: map[string]string{},
		Status: status, Revision: revision, CreatedAt: now, UpdatedAt: now,
	}
}
