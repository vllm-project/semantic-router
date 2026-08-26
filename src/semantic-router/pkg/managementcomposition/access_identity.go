package managementcomposition

import (
	"fmt"
	"time"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	invitationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingruntime"
)

// accessIdentityComposition owns the credential and invitation services that
// share the Router's identity transaction boundary. Keeping this construction
// separate makes the top-level Factory an orchestrator rather than another
// domain implementation.
type accessIdentityComposition struct {
	apiKeys     *apikeymanagement.Service
	delegations *delegationmanagement.Service
	invitations *invitationmanagement.Service
	exchanges   *invitationmanagement.IdentityExchangeCoordinator
}

func composeAccessIdentity(
	dependencies routingruntime.ManagementDependencies,
	commands *managementcommand.Codec,
	defaultRevealable bool,
	keyPrefix string,
	now func() time.Time,
) (*accessIdentityComposition, error) {
	repository, err := accesspostgres.NewAPIKeyManagementRepository(dependencies.AccessStore)
	if err != nil {
		return nil, fmt.Errorf("compose API-key repository: %w", err)
	}
	publicationWaiter, err := delegationmanagement.NewRedisPublicationWaiter(dependencies.Redis, keyPrefix)
	if err != nil {
		return nil, fmt.Errorf("compose access publication waiter: %w", err)
	}
	apiKeys, err := apikeymanagement.NewService(apikeymanagement.Options{
		Repository: repository, Waiter: publicationWaiter, Commands: commands,
		CursorKeyring:      dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		APIKeyPeppers:      dependencies.Keyrings.APIKeyPeppers,
		ResponseKEK:        dependencies.Keyrings.ResponseKEK,
		RevealKEK:          dependencies.Keyrings.RevealKEK,
		DefaultRevealable:  defaultRevealable,
		IdempotencyTTL:     defaultIdempotencyTTL,
		SecretDeliveryTTL:  defaultSecretDeliveryTTL,
		PublicationTimeout: defaultPublicationTimeout,
		Now:                now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose API-key Management: %w", err)
	}
	delegationRepository, err := accesspostgres.NewDelegationManagementRepository(dependencies.AccessStore)
	if err != nil {
		apiKeys.Close()
		return nil, fmt.Errorf("compose delegation repository: %w", err)
	}
	delegations, err := delegationmanagement.NewService(delegationmanagement.Options{
		Repository: delegationRepository, Waiter: publicationWaiter, Commands: commands,
		CursorKeyring:     dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		DelegationPeppers: dependencies.Keyrings.DelegationPeppers,
		ResponseKEK:       dependencies.Keyrings.ResponseKEK, Audience: dependencies.DelegationAudience,
		IdempotencyTTL: defaultIdempotencyTTL, SecretDeliveryTTL: defaultSecretDeliveryTTL, Now: now,
	})
	if err != nil {
		apiKeys.Close()
		return nil, fmt.Errorf("compose delegation Management: %w", err)
	}

	invitations, err := composeInvitationManagement(dependencies, commands, publicationWaiter, now)
	if err != nil {
		delegations.Close()
		apiKeys.Close()
		return nil, err
	}
	exchanges, err := invitationmanagement.NewIdentityExchangeCoordinator(invitations)
	if err != nil {
		invitations.Close()
		delegations.Close()
		apiKeys.Close()
		return nil, fmt.Errorf("compose invitation identity exchange: %w", err)
	}
	return &accessIdentityComposition{
		apiKeys: apiKeys, delegations: delegations, invitations: invitations, exchanges: exchanges,
	}, nil
}

func composeInvitationManagement(
	dependencies routingruntime.ManagementDependencies,
	commands *managementcommand.Codec,
	publicationWaiter invitationmanagement.FirstKeyPublicationWaiter,
	now func() time.Time,
) (*invitationmanagement.Service, error) {
	firstKeys, err := invitationmanagement.NewAPIKeyFirstKeyPreparer(
		dependencies.Keyrings.APIKeyPeppers,
		dependencies.Keyrings.RevealKEK,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("compose invitation first-key issuer: %w", err)
	}
	invitationStore, err := invitationpostgres.New(dependencies.Database)
	if err != nil {
		firstKeys.Close()
		return nil, fmt.Errorf("compose invitation repository: %w", err)
	}
	atomicStore, err := invitationpostgres.NewAtomicExchangeStore(
		invitationStore,
		dependencies.SessionStore,
	)
	if err != nil {
		firstKeys.Close()
		return nil, fmt.Errorf("compose atomic invitation exchange: %w", err)
	}
	invitations, err := invitationmanagement.NewService(invitationmanagement.Options{
		Repository: atomicStore, Commands: commands,
		CursorKeyring:      dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		InvitationPeppers:  dependencies.Keyrings.Invitations,
		ResponseKEK:        dependencies.Keyrings.ResponseKEK,
		FirstKeys:          firstKeys,
		PublicationWaiter:  publicationWaiter,
		IdempotencyTTL:     defaultIdempotencyTTL,
		SecretDeliveryTTL:  defaultSecretDeliveryTTL,
		PublicationTimeout: defaultPublicationTimeout,
		Now:                now,
	})
	if err != nil {
		firstKeys.Close()
		return nil, fmt.Errorf("compose invitation Management: %w", err)
	}
	return invitations, nil
}

func (composition *accessIdentityComposition) Close() error {
	if composition == nil {
		return nil
	}
	if composition.invitations != nil {
		composition.invitations.Close()
		composition.invitations = nil
	}
	if composition.delegations != nil {
		composition.delegations.Close()
		composition.delegations = nil
	}
	if composition.apiKeys != nil {
		composition.apiKeys.Close()
		composition.apiKeys = nil
	}
	composition.exchanges = nil
	return nil
}
