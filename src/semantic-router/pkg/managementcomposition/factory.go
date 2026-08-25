// Package managementcomposition binds Router-native Management domains to
// process-owned routingruntime resources. It has no Dashboard dependency and
// exposes one Management authentication authority.
package managementcomposition

import (
	"context"
	"crypto/ed25519"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	defaultIdempotencyTTL       = time.Hour
	defaultSecretDeliveryTTL    = 10 * time.Minute
	defaultCredentialRetirement = 30 * time.Second
	// #nosec G101 -- this is a public JWT issuer label, not a credential.
	managementTokenIssuer = "vllm-sr"
	// #nosec G101 -- this is a public JWT audience label, not a credential.
	managementTokenAudience     = "vllm-sr-management"
	managementTokenClockSkew    = 5 * time.Second
	policyWorkerConcurrency     = 4
	policyWorkerPollInterval    = 250 * time.Millisecond
	policyWorkerClaimLease      = 30 * time.Second
	policyWorkerMaximumAttempts = 8
	quotaReconciliationWorkers  = 2
)

// Options exposes test and embedding seams. Production composes the Router's
// dynamic trusted-issuer verifier from composed Management dependencies when no verifier
// is injected.
type Options struct {
	AssertionVerifier         managementauth.SubjectAssertionVerifier
	IssuerKeyCache            managementidentity.IssuerKeyCache
	BackchannelLogoutVerifier managementauth.BackchannelLogoutVerifier
	ModelProber               routingmanagement.Prober
	BuiltInRecipes            *routingmanagement.BuiltInRecipeDistribution
	Now                       func() time.Time
}

type Factory struct {
	nativeAccess               bool
	keyPrefix                  string
	agentInferenceEndpoint     string
	assertionVerifier          managementauth.SubjectAssertionVerifier
	issuerKeyCache             managementidentity.IssuerKeyCache
	backchannelLogoutVerifier  managementauth.BackchannelLogoutVerifier
	modelProber                routingmanagement.Prober
	builtInRecipes             *routingmanagement.BuiltInRecipeDistribution
	builtInRecipeDirectory     string
	defaultRevealable          bool
	maxUsageBacklog            int64
	mtlsListenerEnabled        bool
	validateRoutingPublication routingmanagement.PublicationValidator
	now                        func() time.Time
}

func NewFactory(cfg *config.RouterConfig, options Options) (*Factory, error) {
	capabilities, err := runtimecapabilities.Derive(cfg)
	if err != nil {
		return nil, fmt.Errorf("derive runtime capabilities: %w", err)
	}
	if !capabilities.DurableRouting {
		return nil, errors.New("management composition requires durable Management")
	}
	if cfg.ManagementAPI.Auth.Mode != config.ManagementAuthModeRouter {
		return nil, errors.New("management composition requires Router-native authentication")
	}
	if len(cfg.ManagementAPI.Auth.Tokens) != 0 || len(cfg.ManagementAPI.Auth.Roles) != 0 {
		return nil, errors.New("management composition rejects static tokens and roles")
	}
	keyPrefix := ""
	if capabilities.NativeAccess {
		if !capabilities.DistributedState || cfg.AccessRuntimeStore == nil || cfg.AccessRuntimeStore.Redis.KeyPrefix == "" {
			return nil, errors.New("native access composition requires a durable runtime key prefix")
		}
		keyPrefix = cfg.AccessRuntimeStore.Redis.KeyPrefix
		if strings.TrimSpace(cfg.Agent.PublicInferenceEndpoint) == "" {
			return nil, errors.New("native access composition requires the Agent public inference endpoint")
		}
	}
	identityOverrides := 0
	if options.AssertionVerifier != nil {
		identityOverrides++
	}
	if options.IssuerKeyCache != nil {
		identityOverrides++
	}
	if options.BackchannelLogoutVerifier != nil {
		identityOverrides++
	}
	if identityOverrides != 0 && identityOverrides != 3 {
		return nil, errors.New("management identity verifier overrides must be supplied as one complete set")
	}
	var builtInRecipes *routingmanagement.BuiltInRecipeDistribution
	if options.BuiltInRecipes != nil {
		cloned := cloneBuiltInRecipeDistribution(*options.BuiltInRecipes)
		if err := cloned.Validate(); err != nil {
			return nil, fmt.Errorf("management built-in Recipe distribution is invalid: %w", err)
		}
		builtInRecipes = &cloned
	}
	builtInRecipeDirectory := ""
	if cfg.ConfigBaseDir != "" {
		builtInRecipeDirectory = filepath.Join(
			cfg.ConfigBaseDir, routingmanagement.BuiltInRecipeDistributionRelativeDirectory,
		)
	}
	bootstrap := *cfg
	validateRoutingPublication := func(snapshot *routingsnapshot.Snapshot) error {
		_, err := config.CompileDurableRoutingSnapshot(&bootstrap, snapshot)
		return err
	}
	return &Factory{
		nativeAccess:               capabilities.NativeAccess,
		keyPrefix:                  keyPrefix,
		agentInferenceEndpoint:     cfg.Agent.PublicInferenceEndpoint,
		assertionVerifier:          options.AssertionVerifier,
		issuerKeyCache:             options.IssuerKeyCache,
		backchannelLogoutVerifier:  options.BackchannelLogoutVerifier,
		modelProber:                options.ModelProber,
		builtInRecipes:             builtInRecipes,
		builtInRecipeDirectory:     builtInRecipeDirectory,
		defaultRevealable:          cfg.Access.Credentials.Reveal.Enabled,
		maxUsageBacklog:            cfg.Access.Enforcement.MaxUsageBacklog,
		mtlsListenerEnabled:        cfg.ManagementAPI.TLS.ClientCABundleFile != "" || cfg.ManagementAPI.TLS.ClientCABundleEnv != "",
		validateRoutingPublication: validateRoutingPublication,
		now:                        options.Now,
	}, nil
}

func (factory *Factory) Build(
	ctx context.Context,
	dependencies routingruntime.ManagementDependencies,
) (routingruntime.ManagementAPI, error) {
	if factory == nil || (factory.nativeAccess && factory.keyPrefix == "") {
		return nil, errors.New("management composition factory is unavailable")
	}
	if !factory.nativeAccess {
		return factory.buildDurableRouting(ctx, dependencies)
	}
	if err := validateDependencies(dependencies); err != nil {
		return nil, err
	}
	builder, err := newNativeCompositionBuilder(ctx, factory, dependencies)
	if err != nil {
		return nil, err
	}
	succeeded := false
	defer func() {
		if !succeeded {
			_ = builder.owned.Close()
		}
	}()
	if err := builder.composeIdentity(); err != nil {
		return nil, err
	}
	if err := builder.composeSubjectsAndCredentials(); err != nil {
		return nil, err
	}
	if err := builder.composeRouting(); err != nil {
		return nil, err
	}
	if err := builder.composePolicies(); err != nil {
		return nil, err
	}
	if err := builder.composeAccessRuntime(); err != nil {
		return nil, err
	}
	if err := builder.composeObservabilityAndAgent(); err != nil {
		return nil, err
	}
	if err := builder.composeServer(); err != nil {
		return nil, err
	}
	succeeded = true
	return builder.owned, nil
}

func (factory *Factory) loadBuiltInRecipes() (routingmanagement.BuiltInRecipeDistribution, error) {
	if factory.builtInRecipes != nil {
		return cloneBuiltInRecipeDistribution(*factory.builtInRecipes), nil
	}
	if factory.builtInRecipeDirectory == "" {
		return routingmanagement.BuiltInRecipeDistribution{}, errors.New(
			"management composition requires a canonical built-in Recipe distribution directory",
		)
	}
	distribution, err := routingmanagement.LoadBuiltInRecipeDistribution(factory.builtInRecipeDirectory)
	if err != nil {
		return routingmanagement.BuiltInRecipeDistribution{}, fmt.Errorf("load built-in Recipe distribution: %w", err)
	}
	return distribution, nil
}

func cloneBuiltInRecipeDistribution(
	source routingmanagement.BuiltInRecipeDistribution,
) routingmanagement.BuiltInRecipeDistribution {
	result := source
	result.Recipes = make([]routingmanagement.BuiltInRecipe, len(source.Recipes))
	for index, member := range source.Recipes {
		result.Recipes[index] = member
		result.Recipes[index].Input.Document = append([]byte(nil), member.Input.Document...)
	}
	return result
}

func validateDependencies(dependencies routingruntime.ManagementDependencies) error {
	if dependencies.Database == nil || dependencies.Redis == nil || dependencies.AccessStore == nil ||
		dependencies.SessionStore == nil || dependencies.Catalog == nil || dependencies.Catalog.Catalog == nil ||
		dependencies.Catalog.Coordinator == nil || dependencies.Catalog.Registry == nil ||
		dependencies.Catalog.Discovery == nil || strings.TrimSpace(dependencies.DelegationAudience) == "" {
		return errors.New("management composition dependencies are incomplete")
	}
	if !validPolicyWorkerID(dependencies.ReplicaID) {
		return errors.New("management composition requires a stable replica identity")
	}
	return nil
}

func validPolicyWorkerID(value string) bool {
	if value == "" || len(value) > 128 || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}

type managementHTTP interface {
	Register(*http.ServeMux)
	Ready(context.Context) error
}

type backgroundWorker interface {
	Run(context.Context) error
}

type application struct {
	server    managementHTTP
	workers   []backgroundWorker
	closers   []func() error
	closeOnce sync.Once
	closeErr  error
}

func (application *application) Register(mux *http.ServeMux) {
	if application == nil || application.server == nil {
		panic("Management composition is unavailable")
	}
	application.server.Register(mux)
}

func (application *application) Ready(ctx context.Context) error {
	if application == nil || application.server == nil {
		return errors.New("management composition is unavailable")
	}
	return application.server.Ready(ctx)
}

func (application *application) Run(ctx context.Context) error {
	if application == nil || len(application.workers) == 0 {
		return errors.New("management composition worker is unavailable")
	}
	workerContext, cancel := context.WithCancel(ctx)
	defer cancel()
	errorsByWorker := make(chan error, len(application.workers))
	for _, worker := range application.workers {
		go func(active backgroundWorker) { errorsByWorker <- active.Run(workerContext) }(worker)
	}
	var failures []error
	for range application.workers {
		err := <-errorsByWorker
		switch {
		case workerContext.Err() != nil && (err == nil || errors.Is(err, context.Canceled)):
			// A sibling already stopped the application, or the owner cancelled it.
		case err == nil:
			failures = append(failures, errors.New("management worker exited before cancellation"))
			cancel()
		case !errors.Is(err, context.Canceled):
			failures = append(failures, err)
			cancel()
		}
	}
	if len(failures) > 0 {
		return errors.Join(failures...)
	}
	return ctx.Err()
}

func (application *application) addCloser(closer func() error) {
	if application == nil || closer == nil {
		panic("Management composition closer is required")
	}
	application.closers = append(application.closers, closer)
}

func (application *application) Close() error {
	if application == nil {
		return nil
	}
	application.closeOnce.Do(func() {
		var closeErrors []error
		for index := len(application.closers) - 1; index >= 0; index-- {
			closeErrors = append(closeErrors, application.closers[index]())
		}
		application.server = nil
		application.workers = nil
		application.closers = nil
		application.closeErr = errors.Join(closeErrors...)
	})
	return application.closeErr
}

func cloneSigning(source securitykeyring.Signing) securitykeyring.Signing {
	result := securitykeyring.Signing{
		ActiveVersion: source.ActiveVersion,
		Private:       make(map[string]ed25519.PrivateKey, len(source.Private)),
		Public:        make(map[string]ed25519.PublicKey, len(source.Public)),
	}
	for version, key := range source.Private {
		result.Private[version] = ed25519.PrivateKey(append([]byte(nil), key...))
	}
	for version, key := range source.Public {
		result.Public[version] = ed25519.PublicKey(append([]byte(nil), key...))
	}
	return result
}

func pepperSymmetric(source accesscredential.PepperKeyring) securitykeyring.Symmetric {
	return securitykeyring.Symmetric{ActiveVersion: source.ActiveVersion, Keys: source.Keys}
}

var (
	_ routingruntime.ManagementFactory = (*Factory)(nil)
	_ routingruntime.ManagementAPI     = (*application)(nil)
)
