// Package application composes the production Router-native Management
// identity and authentication vertical without depending on a Dashboard or on
// the managed-runtime process entrypoint.
package application

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
	redisclient "github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	authpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	authredis "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/redis"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	authorizationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	identitypostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type Options struct {
	Database                  *sql.DB
	Valkey                    *redisclient.Client
	Barriers                  managementidentity.BarrierAdmin
	Challenges                managementauth.ExchangeChallengeStore
	SessionStore              *authpostgres.Store
	KeyPrefix                 string
	CommandCodec              *managementcommand.Codec
	SessionTokenCodec         managementauth.TokenCodec
	ServiceCredentialPeppers  securitykeyring.Symmetric
	BootstrapToken            []byte
	BootstrapTokenPresent     func() (bool, error)
	RecoveryToken             []byte
	BootstrapIdempotencyKeys  securitykeyring.Symmetric
	BootstrapResponseKEKs     accesscredential.KEKKeyring
	WorkloadCursorKeyring     securitykeyring.Symmetric
	WorkloadResponseKEKs      accesscredential.KEKKeyring
	WorkloadIdempotencyTTL    time.Duration
	WorkloadSecretDeliveryTTL time.Duration
	MTLSListenerEnabled       bool
	AssertionVerifier         managementauth.SubjectAssertionVerifier
	BackchannelLogoutVerifier managementauth.BackchannelLogoutVerifier
	IssuerKeyCache            managementidentity.IssuerKeyCache
	Exchanges                 managementauth.IdentityExchangeCoordinator
	AllowPlaintextAuth        bool
	Now                       func() time.Time
}

// IdentityApplication is the narrow production composition seam consumed by
// routingruntime. Its registrars authenticate through the Management identity
// runtime; listener middleware cannot substitute another credential authority.
type IdentityApplication struct {
	repository     *identitypostgres.Store
	barriers       managementidentity.BarrierAdmin
	credentials    *identitypostgres.ServiceCredentialVerifier
	authentication *managementauth.AuthService
	lifecycle      *managementidentity.LifecycleService
	workloads      *managementidentity.WorkloadIdentityService
	bootstrap      *identitypostgres.BootstrapService
	recovery       *identitypostgres.RecoveryService
	sessions       managementauth.SessionRuntime
	authorizer     *managementserver.IdentityRuntimeAuthorizer
	routes         []managementserver.RouteRegistrar
}

func New(ctx context.Context, options Options) (*IdentityApplication, error) {
	if options.Database == nil || options.SessionStore == nil || options.CommandCodec == nil ||
		options.AssertionVerifier == nil || options.BackchannelLogoutVerifier == nil || options.IssuerKeyCache == nil ||
		options.Exchanges == nil || (options.Valkey != nil && options.KeyPrefix == "") {
		return nil, errors.New("management identity application dependencies are required")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	application := &IdentityApplication{}
	complete := false
	defer func() {
		if !complete {
			_ = application.Close()
		}
	}()
	identityService, newErr := application.composeFoundation(ctx, options, now)
	if newErr != nil {
		return nil, newErr
	}
	recovery, newErr := application.composeAuthentication(options, now)
	if newErr != nil {
		return nil, newErr
	}
	if newErr := application.composeRoutes(options, identityService, recovery, now); newErr != nil {
		return nil, newErr
	}
	if err := application.Ready(ctx); err != nil {
		return nil, err
	}
	complete = true
	return application, nil
}

func (application *IdentityApplication) composeFoundation(
	ctx context.Context, options Options, now func() time.Time,
) (*managementidentity.Service, error) {
	repository, err := identitypostgres.New(options.Database, options.CommandCodec)
	if err != nil {
		return nil, err
	}
	application.repository = repository
	barriers := options.Barriers
	if barriers == nil {
		if options.Valkey != nil {
			var redisBarriers *authredis.Store
			redisBarriers, err = authredis.New(authredis.Options{
				Client: options.Valkey, KeyPrefix: options.KeyPrefix, Loader: options.SessionStore,
			})
			if err == nil {
				err = redisBarriers.Rebuild(ctx)
			}
			barriers = redisBarriers
		} else {
			barriers, err = authpostgres.NewBarrierStore(options.Database)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("compose Management revocation barriers: %w", err)
	}
	application.barriers = barriers
	if readyErr := barriers.Ready(ctx); readyErr != nil {
		return nil, fmt.Errorf("initialize Management revocation barriers: %w", readyErr)
	}
	identityService, err := managementidentity.NewService(repository, barriers)
	if err != nil {
		return nil, err
	}
	sessions := managementauth.SessionRuntime{
		Codec: options.SessionTokenCodec, Sessions: options.SessionStore, Barriers: barriers,
		PolicyLoader: options.SessionStore, NewTokenID: func() (string, error) { return uuid.NewString(), nil },
	}
	application.sessions = sessions
	application.lifecycle, err = managementidentity.NewLifecycleService(
		repository, sessions, barriers, options.IssuerKeyCache, options.BackchannelLogoutVerifier,
	)
	if err != nil {
		return nil, err
	}
	application.credentials, err = identitypostgres.NewServiceCredentialVerifier(options.Database, options.ServiceCredentialPeppers)
	if err != nil {
		return nil, err
	}
	application.workloads, err = managementidentity.NewWorkloadIdentityService(managementidentity.WorkloadIdentityOptions{
		Repository: repository, Commands: options.CommandCodec,
		CursorKeyring: options.WorkloadCursorKeyring, CredentialPeppers: options.ServiceCredentialPeppers,
		ResponseKEK: options.WorkloadResponseKEKs, Barriers: barriers, SessionPolicy: options.SessionStore,
		IdempotencyTTL: options.WorkloadIdempotencyTTL, SecretDeliveryTTL: options.WorkloadSecretDeliveryTTL,
		MTLSListenerEnabled: options.MTLSListenerEnabled, Now: now,
	})
	return identityService, err
}

func (application *IdentityApplication) composeAuthentication(
	options Options, now func() time.Time,
) (*identitypostgres.RecoveryService, error) {
	var err error
	application.bootstrap, err = identitypostgres.NewBootstrapService(identitypostgres.BootstrapOptions{
		Database: options.Database, BootstrapToken: options.BootstrapToken,
		BootstrapTokenPresent: options.BootstrapTokenPresent, IdempotencyKeys: options.BootstrapIdempotencyKeys,
		ResponseKEKs: options.BootstrapResponseKEKs, ServiceCredentialPeppers: options.ServiceCredentialPeppers,
		Now: now,
	})
	if err != nil {
		return nil, err
	}
	if len(options.RecoveryToken) != 0 {
		application.recovery, err = identitypostgres.NewRecoveryService(identitypostgres.RecoveryOptions{
			Database: options.Database, RecoveryToken: options.RecoveryToken,
			IdempotencyKeys: options.BootstrapIdempotencyKeys, Now: now,
		})
		if err != nil {
			return nil, err
		}
	}
	challenges := options.Challenges
	if challenges == nil {
		if options.Valkey != nil {
			challenges, err = authredis.NewChallengeStore(authredis.ChallengeOptions{
				Client: options.Valkey, KeyPrefix: options.KeyPrefix,
			})
		} else {
			challenges, err = authpostgres.NewChallengeStore(authpostgres.ChallengeOptions{Database: options.Database})
		}
		if err != nil {
			return nil, err
		}
	}
	application.authentication, err = managementauth.NewAuthService(managementauth.AuthServiceOptions{
		Challenges: challenges, Assertions: options.AssertionVerifier, Exchanges: options.Exchanges,
		ServiceCredentials: application.credentials, MTLSIdentities: application.workloads,
		Sessions: options.SessionStore, Runtime: application.sessions, Now: now,
	})
	if err != nil {
		return nil, err
	}
	authorityStore, err := authorizationpostgres.New(options.Database)
	if err != nil {
		return nil, err
	}
	application.authorizer, err = managementserver.NewIdentityRuntimeAuthorizer(
		managementauthorization.Runtime{Loader: authorityStore},
	)
	return application.recovery, err
}

func (application *IdentityApplication) composeRoutes(
	options Options,
	identityService *managementidentity.Service,
	recovery *identitypostgres.RecoveryService,
	now func() time.Time,
) error {
	authRoutes, err := managementserver.NewIdentityAuthRoutes(managementserver.IdentityAuthRoutesOptions{
		Service: application.authentication, Bootstrap: application.bootstrap,
		Recovery: optionalRecoveryService(recovery), AllowPlaintextForTests: options.AllowPlaintextAuth,
	})
	if err != nil {
		return err
	}
	resourceRoutes, err := managementserver.NewIdentityResourceRoutes(managementserver.IdentityResourceRoutesOptions{
		Service: identityService, Sessions: application.sessions, Authorization: application.authorizer,
		Commands: options.CommandCodec, Now: now,
	})
	if err != nil {
		return err
	}
	lifecycleRoutes, err := managementserver.NewIdentityLifecycleRoutes(managementserver.IdentityLifecycleRoutesOptions{
		Service: application.lifecycle, Sessions: application.sessions, Authorization: application.authorizer,
		Commands: options.CommandCodec, AllowPlaintextForTests: options.AllowPlaintextAuth, Now: now,
	})
	if err != nil {
		return err
	}
	workloadRoutes, err := managementserver.NewWorkloadIdentityRoutes(managementserver.WorkloadIdentityRoutesOptions{
		Service: application.workloads, Sessions: application.sessions, Authorization: application.authorizer, Now: now,
	})
	if err != nil {
		return err
	}
	application.routes = []managementserver.RouteRegistrar{authRoutes, resourceRoutes, lifecycleRoutes, workloadRoutes}
	return nil
}

func optionalRecoveryService(service *identitypostgres.RecoveryService) managementserver.RecoveryService {
	if service == nil {
		return nil
	}
	return service
}

func (application *IdentityApplication) Register(mux *http.ServeMux) {
	if application == nil || mux == nil {
		panic("Management identity application and mux are required")
	}
	for _, route := range application.routes {
		route.Register(mux)
	}
}

func (application *IdentityApplication) Ready(ctx context.Context) error {
	if application == nil || application.repository == nil || application.barriers == nil ||
		application.credentials == nil || application.authentication == nil || application.lifecycle == nil ||
		application.workloads == nil || application.bootstrap == nil {
		return errors.New("management identity application is unavailable")
	}
	if err := application.repository.Ready(ctx); err != nil {
		return err
	}
	if err := application.barriers.Ready(ctx); err != nil {
		return err
	}
	if err := application.credentials.Ready(ctx); err != nil {
		return err
	}
	if err := application.workloads.Ready(ctx); err != nil {
		return err
	}
	if err := application.bootstrap.Ready(ctx); err != nil {
		return err
	}
	if application.recovery != nil {
		if err := application.recovery.Ready(ctx); err != nil {
			return err
		}
	}
	if err := application.lifecycle.Ready(ctx); err != nil {
		return err
	}
	return application.authentication.Ready(ctx)
}

func (application *IdentityApplication) SessionAuthenticator() managementserver.SessionAuthenticator {
	return application.sessions
}

func (application *IdentityApplication) Authorizer() managementserver.Authorizer {
	return application.authorizer
}

// Close erases the identity application's cloned authentication authorities.
func (application *IdentityApplication) Close() error {
	if application == nil {
		return nil
	}
	if application.credentials != nil {
		application.credentials.Close()
	}
	if application.bootstrap != nil {
		application.bootstrap.Close()
	}
	if application.recovery != nil {
		application.recovery.Close()
	}
	if application.workloads != nil {
		application.workloads.Close()
	}
	zeroSigning(&application.sessions.Codec.Keyring)
	application.routes = nil
	application.credentials = nil
	application.bootstrap = nil
	application.recovery = nil
	application.authentication = nil
	application.lifecycle = nil
	application.workloads = nil
	return nil
}

func zeroSigning(keyring *securitykeyring.Signing) {
	if keyring == nil {
		return
	}
	for _, key := range keyring.Private {
		for index := range key {
			key[index] = 0
		}
	}
	for _, key := range keyring.Public {
		for index := range key {
			key[index] = 0
		}
	}
	*keyring = securitykeyring.Signing{}
}

var _ managementserver.RouteRegistrar = (*IdentityApplication)(nil)
