package managementcomposition

import (
	"context"
	"errors"
	"fmt"
	"time"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	agentpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentnative"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentpublication"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenttoolsource"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentwebsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentworkflow"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	routingapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	authorizationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingruntime"
)

const (
	agentRegistryRetention = 24 * time.Hour
	agentSessionTTL        = 8 * time.Hour
	agentWorkerConcurrency = 2
)

type agentRuntimeComposition struct {
	routes             *managementserver.AgentRoutes
	workers            []backgroundWorker
	service            *agentmanagement.Service
	secrets            *agentruntime.SecretCodec
	liveEvents         *agentruntime.RedisLiveEventBroker
	webSearchTransport *backendegress.Transport
}

func (composition *agentRuntimeComposition) Close() error {
	if composition == nil {
		return nil
	}
	if composition.service != nil {
		composition.service.Close()
	}
	if composition.liveEvents != nil {
		_ = composition.liveEvents.Close()
	}
	if composition.webSearchTransport != nil {
		composition.webSearchTransport.CloseIdleConnections()
	}
	if composition.secrets != nil {
		composition.secrets.Close()
	}
	return nil
}

// composeAgentRuntime is the only production assembly point for Router-native
// Agent state, tools, delegated inference, and turn execution. PostgreSQL owns
// the durable queue/fence; the public inference client deliberately re-enters
// Envoy so access, quota, usage, and logs cannot be bypassed.
func composeAgentRuntime(
	ctx context.Context,
	dependencies routingruntime.ManagementDependencies,
	commandCodec *managementcommand.Codec,
	authorityStore *authorizationpostgres.Store,
	authorization managementauthorization.Runtime,
	namespaces managementserver.NamespaceResolver,
	sessions managementserver.SessionAuthenticator,
	authorizer managementserver.Authorizer,
	routing *routingapplication.Application,
	distribution routingmanagement.BuiltInRecipeDistribution,
	publicInferenceEndpoint string,
	keyPrefix string,
	now func() time.Time,
) (_ *agentRuntimeComposition, resultErr error) {
	if dependencies.Database == nil || dependencies.Redis == nil || commandCodec == nil ||
		authorityStore == nil || authorization.Loader == nil || namespaces == nil || sessions == nil ||
		authorizer == nil || routing == nil || routing.Service == nil ||
		dependencies.ProtocolCodecs == nil || publicInferenceEndpoint == "" || keyPrefix == "" {
		return nil, errors.New("agent production composition dependencies are incomplete")
	}
	builder := &agentRuntimeBuilder{
		dependencies: dependencies, commandCodec: commandCodec, authorityStore: authorityStore,
		authorization: authorization, namespaces: namespaces, sessions: sessions,
		authorizer: authorizer, routing: routing, distribution: distribution,
		publicInferenceEndpoint: publicInferenceEndpoint, keyPrefix: keyPrefix, now: now,
		composition: &agentRuntimeComposition{},
	}
	defer func() {
		if resultErr != nil {
			_ = builder.composition.Close()
		}
	}()
	if err := builder.composeToolsAndAuthority(); err != nil {
		return nil, err
	}
	if err := builder.composeManagement(ctx); err != nil {
		return nil, err
	}
	if err := builder.composeExecutionAndRoutes(); err != nil {
		return nil, err
	}
	return builder.composition, nil
}

type agentRuntimeBuilder struct {
	dependencies            routingruntime.ManagementDependencies
	commandCodec            *managementcommand.Codec
	authorityStore          *authorizationpostgres.Store
	authorization           managementauthorization.Runtime
	namespaces              managementserver.NamespaceResolver
	sessions                managementserver.SessionAuthenticator
	authorizer              managementserver.Authorizer
	routing                 *routingapplication.Application
	distribution            routingmanagement.BuiltInRecipeDistribution
	publicInferenceEndpoint string
	keyPrefix               string
	now                     func() time.Time
	composition             *agentRuntimeComposition
	store                   *agentpostgres.Store
	remoteTools             *agenttoolsource.ClientFactory
	sessionAuthority        *accesspostgres.AgentSessionAuthority
	registries              *agentruntime.RegistryArchive
	defaults                *agentpostgres.DefaultReconciler
}

func (builder *agentRuntimeBuilder) composeToolsAndAuthority() error {
	var err error
	builder.store, err = agentpostgres.New(builder.dependencies.Database, builder.commandCodec)
	if err != nil {
		return fmt.Errorf("compose Agent PostgreSQL store: %w", err)
	}
	builder.composition.secrets, err = agentruntime.NewSecretCodec(
		builder.dependencies.Keyrings.Routing.AgentSecret.Symmetric(),
	)
	if err != nil {
		return fmt.Errorf("compose Agent secret codec: %w", err)
	}
	vault, err := agentruntime.NewCredentialVault(builder.store, builder.composition.secrets, builder.now)
	if err != nil {
		return fmt.Errorf("compose Agent Tool credential vault: %w", err)
	}
	builder.remoteTools, err = agenttoolsource.NewClientFactory(agenttoolsource.ClientFactoryOptions{
		OperatorGuard: backendegress.Guard{Policy: builder.dependencies.EgressPolicy},
		Credentials:   vault,
	})
	if err != nil {
		return fmt.Errorf("compose Agent remote Tool Source client: %w", err)
	}
	catalog, err := agentnative.NewConfigCatalog()
	if err != nil {
		return fmt.Errorf("compose Agent component catalog: %w", err)
	}
	examples, err := agentnative.NewDistributionExamples(builder.distribution)
	if err != nil {
		return fmt.Errorf("compose Agent Recipe examples: %w", err)
	}
	nativeTools, err := agentnative.New(agentnative.Options{
		Store: builder.store, Routing: builder.routing.Service, Scopes: builder.authorization,
		Catalog: catalog, Examples: examples,
	})
	if err != nil {
		return fmt.Errorf("compose Router-native Agent read tools: %w", err)
	}
	workflowTools, err := agentworkflow.New(agentworkflow.Options{
		Store: builder.store, Routing: builder.routing.Service, Authorization: builder.authorization,
		Commands: builder.commandCodec, Now: builder.now,
	})
	if err != nil {
		return fmt.Errorf("compose Router-native Agent workflow tools: %w", err)
	}
	webSearchTransport, err := backendegress.NewTransport(backendegress.TransportOptions{
		Guard: backendegress.Guard{Policy: builder.dependencies.EgressPolicy},
	})
	if err != nil {
		return fmt.Errorf("compose Agent web search egress: %w", err)
	}
	builder.composition.webSearchTransport = webSearchTransport
	webSearchTools, err := agentwebsearch.New(agentwebsearch.Options{
		Client: backendegress.NewHTTPClient(webSearchTransport, false),
	})
	if err != nil {
		return fmt.Errorf("compose Router-native Agent web search: %w", err)
	}
	allNativeTools, err := agentruntime.NewCompositeNativeToolProvider(
		[]agentruntime.NativeToolProvider{nativeTools, workflowTools, webSearchTools},
		agentmanagement.BuiltinBuilderToolNames(),
	)
	if err != nil {
		return fmt.Errorf("compose complete Router-native Agent Tool set: %w", err)
	}
	return builder.composeAuthorityAndRegistry(allNativeTools)
}

func (builder *agentRuntimeBuilder) composeAuthorityAndRegistry(
	nativeTools *agentruntime.CompositeNativeToolProvider,
) error {
	waiter, err := delegationmanagement.NewRedisPublicationWaiter(
		builder.dependencies.Redis, builder.keyPrefix,
	)
	if err != nil {
		return fmt.Errorf("compose Agent delegated inference publication waiter: %w", err)
	}
	builder.sessionAuthority, err = accesspostgres.NewAgentSessionAuthority(
		accesspostgres.AgentSessionAuthorityOptions{
			Store: builder.dependencies.AccessStore, Management: builder.authorityStore,
			Peppers: builder.dependencies.Keyrings.DelegationPeppers,
			Secrets: builder.composition.secrets, Waiter: waiter,
			Audience: builder.dependencies.DelegationAudience, Now: builder.now,
		},
	)
	if err != nil {
		return fmt.Errorf("compose Agent session authority: %w", err)
	}
	toolAuthorizer, err := agentruntime.NewLiveToolAuthorizer(agentruntime.LiveToolAuthorizerOptions{
		Store: builder.store, Sessions: builder.sessionAuthority, Authority: builder.authorityStore,
	})
	if err != nil {
		return fmt.Errorf("compose Agent Tool authorization: %w", err)
	}
	assembler, err := agentruntime.NewRegistryAssembler(agentruntime.RegistryAssemblerOptions{
		Store: builder.store, Native: nativeTools, Remote: builder.remoteTools,
	})
	if err != nil {
		return fmt.Errorf("compose Agent Tool Registry assembler: %w", err)
	}
	builder.registries, err = agentruntime.NewRegistryArchive(agentruntime.RegistryArchiveOptions{
		Store: builder.store, Assembler: assembler, Authorizer: toolAuthorizer,
		Retention: agentRegistryRetention, Now: builder.now,
	})
	if err != nil {
		return fmt.Errorf("compose Agent Tool Registry archive: %w", err)
	}
	return nil
}

func (builder *agentRuntimeBuilder) composeManagement(ctx context.Context) error {
	definitions, err := agentmanagement.NewRegistryDefinitionValidator(builder.store)
	if err != nil {
		return fmt.Errorf("compose Agent definition validator: %w", err)
	}
	service, err := agentmanagement.NewService(agentmanagement.ServiceOptions{
		Store: builder.store, SessionAuthority: builder.sessionAuthority,
		TargetVisibility: builder.sessionAuthority, Definitions: definitions,
		SourcePolicies: agenttoolsource.PolicyCompiler{}, ToolSources: builder.remoteTools,
		Registries: builder.registries, SecretCodec: builder.composition.secrets,
		CommandCodec:  builder.commandCodec,
		CursorKeyring: builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		SessionTTL:    agentSessionTTL, Now: builder.now,
	})
	if err != nil {
		return fmt.Errorf("compose Agent Management service: %w", err)
	}
	builder.composition.service = service
	builder.defaults, err = agentpostgres.NewDefaultReconciler(builder.store, 0, builder.now)
	if err != nil {
		return fmt.Errorf("compose Agent defaults: %w", err)
	}
	if err := builder.defaults.Reconcile(ctx); err != nil {
		return fmt.Errorf("install default Agent Profiles and Skills: %w", err)
	}
	return nil
}

func (builder *agentRuntimeBuilder) composeExecutionAndRoutes() error {
	inference, err := agentruntime.NewHTTPPublicInferenceClient(agentruntime.HTTPPublicInferenceOptions{
		Endpoint: builder.publicInferenceEndpoint, Codecs: builder.dependencies.ProtocolCodecs,
	})
	if err != nil {
		return fmt.Errorf("compose Agent public inference client: %w", err)
	}
	liveEvents, err := agentruntime.NewRedisLiveEventBroker(agentruntime.RedisLiveEventBrokerOptions{
		Client: builder.dependencies.Redis, KeyPrefix: builder.keyPrefix,
	})
	if err != nil {
		return fmt.Errorf("compose Agent live event broker: %w", err)
	}
	builder.composition.liveEvents = liveEvents
	worker, err := agentruntime.NewWorker(agentruntime.WorkerOptions{
		Store: builder.store, Authority: builder.sessionAuthority, Registries: builder.registries,
		Inference: inference, LiveEvents: liveEvents, WorkerID: builder.dependencies.ReplicaID,
		Concurrency: agentWorkerConcurrency, Now: builder.now,
	})
	if err != nil {
		return fmt.Errorf("compose Agent turn worker: %w", err)
	}
	publications, err := agentpublication.New(agentpublication.Options{
		Store: builder.store, Publisher: builder.routing.Service,
		Commands: builder.commandCodec, Now: builder.now,
	})
	if err != nil {
		return fmt.Errorf("compose Agent publication committer: %w", err)
	}
	routes, err := managementserver.NewAgentRoutes(managementserver.AgentRoutesOptions{
		Service: builder.composition.service, Defaults: builder.defaults,
		Publications: publications, LiveEvents: liveEvents,
		Namespaces: builder.namespaces, Sessions: builder.sessions,
		Authorization: builder.authorizer, Scopes: builder.authorization, Now: builder.now,
	})
	if err != nil {
		return fmt.Errorf("compose Agent Management routes: %w", err)
	}
	builder.composition.routes = routes
	builder.composition.workers = []backgroundWorker{builder.defaults, worker}
	return nil
}
