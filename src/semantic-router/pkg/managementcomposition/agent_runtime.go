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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentworkflow"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	routingmanaged "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/managed"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	authorizationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
)

const (
	agentRegistryRetention = 24 * time.Hour
	agentSessionTTL        = 8 * time.Hour
	agentWorkerConcurrency = 2
)

type agentRuntimeComposition struct {
	routes     *managementserver.AgentRoutes
	workers    []backgroundWorker
	service    *agentmanagement.Service
	secrets    *agentruntime.SecretCodec
	liveEvents *agentruntime.RedisLiveEventBroker
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
	dependencies managedruntime.ManagementDependencies,
	commandCodec *managementcommand.Codec,
	authorityStore *authorizationpostgres.Store,
	authorization managementauthorization.Runtime,
	namespaces managementserver.NamespaceResolver,
	sessions managementserver.SessionAuthenticator,
	authorizer managementserver.Authorizer,
	routing *routingmanaged.Application,
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
	store, composeAgentRuntimeErr := agentpostgres.New(dependencies.Database, commandCodec)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent PostgreSQL store: %w", composeAgentRuntimeErr)
	}
	secrets, composeAgentRuntimeErr := agentruntime.NewSecretCodec(
		dependencies.Keyrings.ControlPlane.AgentSecret.Symmetric(),
	)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent secret codec: %w", composeAgentRuntimeErr)
	}
	composition := &agentRuntimeComposition{secrets: secrets}
	defer func() {
		if resultErr != nil {
			_ = composition.Close()
		}
	}()

	vault, composeAgentRuntimeErr := agentruntime.NewCredentialVault(store, secrets, now)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent Tool credential vault: %w", composeAgentRuntimeErr)
	}
	remoteTools, composeAgentRuntimeErr := agenttoolsource.NewClientFactory(agenttoolsource.ClientFactoryOptions{
		OperatorGuard: backendegress.Guard{Policy: dependencies.EgressPolicy},
		Credentials:   vault,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent remote Tool Source client: %w", composeAgentRuntimeErr)
	}
	catalog, composeAgentRuntimeErr := agentnative.NewConfigCatalog()
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent component catalog: %w", composeAgentRuntimeErr)
	}
	examples, composeAgentRuntimeErr := agentnative.NewDistributionExamples(distribution)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent Recipe examples: %w", composeAgentRuntimeErr)
	}
	nativeTools, composeAgentRuntimeErr := agentnative.New(agentnative.Options{
		Store: store, Routing: routing.Service, Scopes: authorization,
		Catalog: catalog, Examples: examples,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Router-native Agent read tools: %w", composeAgentRuntimeErr)
	}
	workflowTools, composeAgentRuntimeErr := agentworkflow.New(agentworkflow.Options{
		Store: store, Routing: routing.Service, Authorization: authorization,
		Commands: commandCodec, Now: now,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Router-native Agent workflow tools: %w", composeAgentRuntimeErr)
	}
	allNativeTools, composeAgentRuntimeErr := agentruntime.NewCompositeNativeToolProvider(
		[]agentruntime.NativeToolProvider{nativeTools, workflowTools},
		agentmanagement.BuiltinBuilderToolNames(),
	)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose complete Router-native Agent Tool set: %w", composeAgentRuntimeErr)
	}
	waiter, composeAgentRuntimeErr := delegationmanagement.NewRedisPublicationWaiter(dependencies.Redis, keyPrefix)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent delegated inference publication waiter: %w", composeAgentRuntimeErr)
	}
	sessionAuthority, composeAgentRuntimeErr := accesspostgres.NewAgentSessionAuthority(
		accesspostgres.AgentSessionAuthorityOptions{
			Store: dependencies.AccessStore, Management: authorityStore,
			Peppers: dependencies.Keyrings.DelegationPeppers, Secrets: secrets,
			Waiter: waiter, Audience: dependencies.DelegationAudience, Now: now,
		},
	)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent session authority: %w", composeAgentRuntimeErr)
	}
	toolAuthorizer, composeAgentRuntimeErr := agentruntime.NewLiveToolAuthorizer(agentruntime.LiveToolAuthorizerOptions{
		Store: store, Sessions: sessionAuthority, Authority: authorityStore,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent Tool authorization: %w", composeAgentRuntimeErr)
	}
	assembler, composeAgentRuntimeErr := agentruntime.NewRegistryAssembler(agentruntime.RegistryAssemblerOptions{
		Store: store, Native: allNativeTools, Remote: remoteTools,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent Tool Registry assembler: %w", composeAgentRuntimeErr)
	}
	registries, composeAgentRuntimeErr := agentruntime.NewRegistryArchive(agentruntime.RegistryArchiveOptions{
		Store: store, Assembler: assembler, Authorizer: toolAuthorizer,
		Retention: agentRegistryRetention, Now: now,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent Tool Registry archive: %w", composeAgentRuntimeErr)
	}
	definitions, composeAgentRuntimeErr := agentmanagement.NewRegistryDefinitionValidator(store)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent definition validator: %w", composeAgentRuntimeErr)
	}
	service, composeAgentRuntimeErr := agentmanagement.NewService(agentmanagement.ServiceOptions{
		Store: store, SessionAuthority: sessionAuthority, TargetVisibility: sessionAuthority,
		Definitions: definitions, SourcePolicies: agenttoolsource.PolicyCompiler{},
		ToolSources: remoteTools, Registries: registries, SecretCodec: secrets,
		CommandCodec:  commandCodec,
		CursorKeyring: dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		SessionTTL:    agentSessionTTL, Now: now,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent Management service: %w", composeAgentRuntimeErr)
	}
	composition.service = service
	defaults, composeAgentRuntimeErr := agentpostgres.NewDefaultReconciler(store, 0, now)
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent defaults: %w", composeAgentRuntimeErr)
	}
	// Defaults are a listener startup gate, not eventual client-side setup.
	if err := defaults.Reconcile(ctx); err != nil {
		return nil, fmt.Errorf("install default Agent Profiles and Skills: %w", err)
	}
	inference, composeAgentRuntimeErr := agentruntime.NewHTTPPublicInferenceClient(agentruntime.HTTPPublicInferenceOptions{
		Endpoint: publicInferenceEndpoint, Codecs: dependencies.ProtocolCodecs,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent public inference client: %w", composeAgentRuntimeErr)
	}
	liveEvents, composeAgentRuntimeErr := agentruntime.NewRedisLiveEventBroker(agentruntime.RedisLiveEventBrokerOptions{
		Client: dependencies.Redis, KeyPrefix: keyPrefix,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent live event broker: %w", composeAgentRuntimeErr)
	}
	composition.liveEvents = liveEvents
	worker, composeAgentRuntimeErr := agentruntime.NewWorker(agentruntime.WorkerOptions{
		Store: store, Authority: sessionAuthority, Registries: registries,
		Inference: inference, LiveEvents: liveEvents, WorkerID: dependencies.ReplicaID,
		Concurrency: agentWorkerConcurrency, Now: now,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent turn worker: %w", composeAgentRuntimeErr)
	}
	publications, composeAgentRuntimeErr := agentpublication.New(agentpublication.Options{
		Store: store, Publisher: routing.Service, Commands: commandCodec, Now: now,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent publication committer: %w", composeAgentRuntimeErr)
	}
	routes, composeAgentRuntimeErr := managementserver.NewAgentRoutes(managementserver.AgentRoutesOptions{
		Service: service, Defaults: defaults, Publications: publications,
		LiveEvents: liveEvents,
		Namespaces: namespaces, Sessions: sessions, Authorization: authorizer,
		Scopes: authorization, Now: now,
	})
	if composeAgentRuntimeErr != nil {
		return nil, fmt.Errorf("compose Agent Management routes: %w", composeAgentRuntimeErr)
	}
	composition.routes = routes
	composition.workers = []backgroundWorker{defaults, worker}
	return composition, nil
}
