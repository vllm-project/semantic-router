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
// managedruntime. Its registrars authenticate through the Management identity
// runtime; listener middleware cannot substitute another credential authority.
type IdentityApplication struct {
	repository     *identitypostgres.Store
	barriers       *authredis.Store
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
	if options.Database == nil || options.Valkey == nil || options.SessionStore == nil || options.CommandCodec == nil ||
		options.AssertionVerifier == nil || options.BackchannelLogoutVerifier == nil || options.IssuerKeyCache == nil ||
		options.Exchanges == nil || options.KeyPrefix == "" {
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
	repository, newErr := identitypostgres.New(options.Database, options.CommandCodec)
	if newErr != nil {
		return nil, newErr
	}
	application.repository = repository
	sessionStore := options.SessionStore
	barriers, newErr := authredis.New(authredis.Options{Client: options.Valkey, KeyPrefix: options.KeyPrefix, Loader: sessionStore})
	if newErr != nil {
		return nil, newErr
	}
	application.barriers = barriers
	if err := barriers.Rebuild(ctx); err != nil {
		return nil, fmt.Errorf("rebuild Management revocation barriers: %w", err)
	}
	identityService, newErr := managementidentity.NewService(repository, barriers)
	if newErr != nil {
		return nil, newErr
	}
	sessions := managementauth.SessionRuntime{
		Codec: options.SessionTokenCodec, Sessions: sessionStore, Barriers: barriers,
		PolicyLoader: sessionStore, NewTokenID: func() (string, error) { return uuid.NewString(), nil },
	}
	application.sessions = sessions
	lifecycle, newErr := managementidentity.NewLifecycleService(
		repository, sessions, barriers, options.IssuerKeyCache, options.BackchannelLogoutVerifier,
	)
	if newErr != nil {
		return nil, newErr
	}
	application.lifecycle = lifecycle
	credentials, newErr := identitypostgres.NewServiceCredentialVerifier(options.Database, options.ServiceCredentialPeppers)
	if newErr != nil {
		return nil, newErr
	}
	application.credentials = credentials
	workloads, newErr := managementidentity.NewWorkloadIdentityService(managementidentity.WorkloadIdentityOptions{
		Repository: repository, Commands: options.CommandCodec,
		CursorKeyring: options.WorkloadCursorKeyring, CredentialPeppers: options.ServiceCredentialPeppers,
		ResponseKEK: options.WorkloadResponseKEKs, Barriers: barriers, SessionPolicy: sessionStore,
		IdempotencyTTL: options.WorkloadIdempotencyTTL, SecretDeliveryTTL: options.WorkloadSecretDeliveryTTL,
		MTLSListenerEnabled: options.MTLSListenerEnabled, Now: now,
	})
	if newErr != nil {
		return nil, newErr
	}
	application.workloads = workloads
	bootstrap, newErr := identitypostgres.NewBootstrapService(identitypostgres.BootstrapOptions{
		Database: options.Database, BootstrapToken: options.BootstrapToken,
		BootstrapTokenPresent:    options.BootstrapTokenPresent,
		IdempotencyKeys:          options.BootstrapIdempotencyKeys,
		ResponseKEKs:             options.BootstrapResponseKEKs,
		ServiceCredentialPeppers: options.ServiceCredentialPeppers,
		Now:                      now,
	})
	if newErr != nil {
		return nil, newErr
	}
	application.bootstrap = bootstrap
	var recovery *identitypostgres.RecoveryService
	if len(options.RecoveryToken) != 0 {
		recovery, newErr = identitypostgres.NewRecoveryService(identitypostgres.RecoveryOptions{
			Database: options.Database, RecoveryToken: options.RecoveryToken,
			IdempotencyKeys: options.BootstrapIdempotencyKeys, Now: now,
		})
		if newErr != nil {
			return nil, newErr
		}
		application.recovery = recovery
	}
	challenges, newErr := authredis.NewChallengeStore(authredis.ChallengeOptions{Client: options.Valkey, KeyPrefix: options.KeyPrefix})
	if newErr != nil {
		return nil, newErr
	}
	authentication, newErr := managementauth.NewAuthService(managementauth.AuthServiceOptions{
		Challenges: challenges, Assertions: options.AssertionVerifier, Exchanges: options.Exchanges,
		ServiceCredentials: credentials, MTLSIdentities: workloads,
		Sessions: sessionStore, Runtime: sessions, Now: now,
	})
	if newErr != nil {
		return nil, newErr
	}
	application.authentication = authentication
	authorityStore, newErr := authorizationpostgres.New(options.Database)
	if newErr != nil {
		return nil, newErr
	}
	authorizer, newErr := managementserver.NewIdentityRuntimeAuthorizer(managementauthorization.Runtime{Loader: authorityStore})
	if newErr != nil {
		return nil, newErr
	}
	application.authorizer = authorizer
	authRoutes, newErr := managementserver.NewIdentityAuthRoutes(managementserver.IdentityAuthRoutesOptions{
		Service: authentication, Bootstrap: bootstrap, Recovery: optionalRecoveryService(recovery),
		AllowPlaintextForTests: options.AllowPlaintextAuth,
	})
	if newErr != nil {
		return nil, newErr
	}
	resourceRoutes, newErr := managementserver.NewIdentityResourceRoutes(managementserver.IdentityResourceRoutesOptions{
		Service: identityService, Sessions: sessions, Authorization: authorizer,
		Commands: options.CommandCodec, Now: now,
	})
	if newErr != nil {
		return nil, newErr
	}
	lifecycleRoutes, newErr := managementserver.NewIdentityLifecycleRoutes(managementserver.IdentityLifecycleRoutesOptions{
		Service: lifecycle, Sessions: sessions, Authorization: authorizer,
		Commands: options.CommandCodec, AllowPlaintextForTests: options.AllowPlaintextAuth, Now: now,
	})
	if newErr != nil {
		return nil, newErr
	}
	workloadRoutes, newErr := managementserver.NewWorkloadIdentityRoutes(managementserver.WorkloadIdentityRoutesOptions{
		Service: workloads, Sessions: sessions, Authorization: authorizer, Now: now,
	})
	if newErr != nil {
		return nil, newErr
	}
	application.routes = []managementserver.RouteRegistrar{authRoutes, resourceRoutes, lifecycleRoutes, workloadRoutes}
	if err := application.Ready(ctx); err != nil {
		return nil, err
	}
	complete = true
	return application, nil
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
