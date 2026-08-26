package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcomposition"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
)

const managementIssuerEgressPolicyFileEnv = "VLLM_SR_INTERNAL_MANAGEMENT_ISSUER_EGRESS_POLICY_FILE"

// processRuntime owns process composition and immutable Router generations as
// one lifecycle. The ExtProc listener borrows only the request resolver and
// never closes shared resources.
type processRuntime struct {
	core         *routingruntime.Runtime
	capabilities runtimecapabilities.RuntimeCapabilities
	routers      *extproc.DurableRoutingRegistry
	requests     *extproc.DurableRoutingRequestRuntime
	fileRequests *extproc.FileRequestRuntime
	dispatch     extproc.DispatchCapabilityRuntime
	outcomes     extproc.OutcomeFeedbackRuntime
	projections  extproc.OutcomeLearningProjectionRuntime
	terminals    backendinvoker.ResponseTerminalReader
	codecs       *protocolcodec.Registry

	closeOnce sync.Once
	closeErr  error
}

// resolveProductionManagementFactory is the sole production connection point
// for Management identity, authorization, and domain routes. The concrete
// factory is linked here; durable startup never falls back to legacy or
// Dashboard state when it is unavailable.
var resolveProductionManagementFactory = func(cfg *config.RouterConfig) (routingruntime.ManagementFactory, error) {
	factory, err := managementcomposition.NewFactory(cfg, managementcomposition.Options{})
	if err != nil {
		return nil, err
	}
	return factory, nil
}

func composeProcessRuntime(
	ctx context.Context,
	cfg *config.RouterConfig,
) (*processRuntime, error) {
	capabilities, err := runtimecapabilities.Derive(cfg)
	if err != nil {
		return nil, fmt.Errorf("derive runtime capabilities: %w", err)
	}
	options := routingruntime.Options{}
	if capabilities.DurableRouting {
		options.ProviderIntegrations, options.ProviderBackendCompilers = productionProviderIntegrations()
		options.ReplicaID = strings.TrimSpace(os.Getenv("VLLM_SR_REPLICA_ID"))
		if options.ReplicaID == "" {
			return nil, errors.New("VLLM_SR_REPLICA_ID is required for durable routing")
		}
		if capabilities.ManagementAPI {
			options.ManagementIssuerEgressPolicy, err = resolveManagementIssuerEgressPolicy()
			if err != nil {
				return nil, err
			}
			factory, factoryErr := resolveProductionManagementFactory(cfg)
			if factoryErr != nil {
				return nil, factoryErr
			}
			if factory == nil {
				return nil, errors.New("management API factory is unavailable")
			}
			options.ManagementFactory = factory
		}
	}
	core, err := routingruntime.New(ctx, cfg, options)
	if err != nil {
		return nil, fmt.Errorf("compose process runtime: %w", err)
	}
	result := &processRuntime{
		core: core, capabilities: capabilities,
		terminals: core.ResponseTerminals(), codecs: core.ProtocolCodecs(),
	}
	dispatch, ok := core.DispatchCapabilities().(extproc.DispatchCapabilityRuntime)
	if !ok || dispatch == nil {
		_ = core.Close()
		return nil, errors.New("backend dispatch authority is unavailable")
	}
	result.dispatch = dispatch
	if capabilities.DurableRouting {
		routers, composeProcessRuntimeErr := extproc.NewDurableRoutingRegistry(extproc.DurableRoutingRegistryOptions{
			BootstrapConfig: cfg,
			Dependencies: extproc.RuntimeDependencies{
				InferenceAccess: core.InferenceAccess(), DispatchCapabilities: dispatch,
				OutcomeFeedback:   core.OutcomeFeedback(),
				OutcomeProjection: core.OutcomeProjection(),
				ResponseTerminals: core.ResponseTerminals(),
				ProtocolCodecs:    core.ProtocolCodecs(),
			},
		})
		if composeProcessRuntimeErr != nil {
			_ = core.Close()
			return nil, fmt.Errorf("compose durable Router generations: %w", composeProcessRuntimeErr)
		}
		if err := core.AttachRoutingSnapshots(routers); err != nil {
			_ = routers.Close()
			_ = core.Close()
			return nil, fmt.Errorf("attach durable Router generations: %w", err)
		}
		requests, composeProcessRuntimeErr := extproc.NewDurableRoutingRequestRuntime(extproc.DurableRoutingRequestRuntimeOptions{
			Access: core.InferenceAccess(), PublicNamespaceID: core.PublicRoutingNamespace(),
			Publications: core, Routers: routers, Dispatch: dispatch,
		})
		if composeProcessRuntimeErr != nil {
			_ = routers.Close()
			_ = core.Close()
			return nil, fmt.Errorf("compose durable request routing: %w", composeProcessRuntimeErr)
		}
		result.routers = routers
		result.requests = requests
		result.outcomes = core.OutcomeFeedback()
		result.projections = core.OutcomeProjection()
	} else {
		if cfg == nil || cfg.RoutingSnapshot == nil {
			_ = core.Close()
			return nil, errors.New("file routing snapshot is unavailable")
		}
		requests, err := extproc.NewFileRequestRuntime(extproc.FileRequestRuntimeOptions{
			NamespaceID:  cfg.RoutingSnapshot.NamespaceID,
			Publications: core,
			Dispatch:     dispatch,
		})
		if err != nil {
			_ = core.Close()
			return nil, fmt.Errorf("compose file request routing: %w", err)
		}
		result.fileRequests = requests
	}
	return result, nil
}

func resolveManagementIssuerEgressPolicy() (*backendegress.Policy, error) {
	path := strings.TrimSpace(os.Getenv(managementIssuerEgressPolicyFileEnv))
	if path == "" {
		return nil, nil
	}
	policy, err := backendegress.LoadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load system Management issuer egress policy: %w", err)
	}
	return &policy, nil
}

func (runtime *processRuntime) Start(ctx context.Context) error {
	if runtime == nil || runtime.core == nil {
		return errors.New("process runtime is unavailable")
	}
	if err := runtime.core.Start(ctx); err != nil {
		return fmt.Errorf("start process runtime: %w", err)
	}
	return nil
}

// Ready is the complete serving contract for the operational probe. Durable
// routing remains unready until publication, access, provider, and dispatch
// dependencies are all usable, independently of Management API enablement.
func (runtime *processRuntime) Ready(ctx context.Context) error {
	if runtime == nil || runtime.core == nil {
		return errors.New("process runtime is unavailable")
	}
	if runtime.capabilities.ManagementAPI {
		management := runtime.core.ManagementAPI()
		if management == nil {
			return errors.New("management runtime is unavailable")
		}
		return management.Ready(ctx)
	}
	return runtime.core.Ready(ctx)
}

func (runtime *processRuntime) ManagementAPI() routingruntime.ManagementAPI {
	if runtime == nil || runtime.core == nil || !runtime.capabilities.ManagementAPI {
		return nil
	}
	return runtime.core.ManagementAPI()
}

func (runtime *processRuntime) DurableRoutingRequests() *extproc.DurableRoutingRequestRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.requests
}

func (runtime *processRuntime) FileRequests() *extproc.FileRequestRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.fileRequests
}

func (runtime *processRuntime) DispatchCapabilities() extproc.DispatchCapabilityRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.dispatch
}

func (runtime *processRuntime) OutcomeFeedback() extproc.OutcomeFeedbackRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.outcomes
}

func (runtime *processRuntime) OutcomeProjection() extproc.OutcomeLearningProjectionRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.projections
}

func (runtime *processRuntime) ResponseTerminals() backendinvoker.ResponseTerminalReader {
	if runtime == nil {
		return nil
	}
	return runtime.terminals
}

func (runtime *processRuntime) ProtocolCodecs() *protocolcodec.Registry {
	if runtime == nil {
		return nil
	}
	return runtime.codecs
}

func (runtime *processRuntime) Close() error {
	if runtime == nil {
		return nil
	}
	runtime.closeOnce.Do(func() {
		var closeErrors []error
		if runtime.core != nil {
			closeErrors = append(closeErrors, runtime.core.Close())
		}
		if runtime.routers != nil {
			closeErrors = append(closeErrors, runtime.routers.Close())
		}
		runtime.closeErr = errors.Join(closeErrors...)
	})
	return runtime.closeErr
}
