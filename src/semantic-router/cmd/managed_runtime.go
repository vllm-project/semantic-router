package main

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcomposition"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// processRuntime owns the managed control plane and its immutable Router
// generations as one lifecycle. The ExtProc listener borrows only the request
// resolver and never closes shared resources.
type processRuntime struct {
	core               *managedruntime.Runtime
	routers            *extproc.ManagedRouterRegistry
	requests           *extproc.ManagedRequestRuntime
	standaloneRequests *extproc.StandaloneRequestRuntime
	dispatch           extproc.DispatchCapabilityRuntime
	outcomes           extproc.OutcomeFeedbackRuntime
	projections        extproc.OutcomeLearningProjectionRuntime
	terminals          backendinvoker.ResponseTerminalReader
	codecs             *protocolcodec.Registry

	closeOnce sync.Once
	closeErr  error
}

// resolveProductionManagementFactory is the sole production connection point
// for Management identity, authorization, and domain routes. The concrete
// factory is linked here; managed startup never falls back to legacy or
// Dashboard state when it is unavailable.
var resolveProductionManagementFactory = func(cfg *config.RouterConfig) (managedruntime.ManagementFactory, error) {
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
	options := managedruntime.Options{}
	if cfg != nil && cfg.ControlPlane.Mode == config.ControlPlaneModeManaged {
		options.ProviderIntegrations, options.ProviderBackendCompilers = productionProviderIntegrations()
		factory, err := resolveProductionManagementFactory(cfg)
		if err != nil {
			return nil, err
		}
		if factory == nil {
			return nil, errors.New("managed Management factory is unavailable")
		}
		options.ManagementFactory = factory
	}
	core, err := managedruntime.New(ctx, cfg, options)
	if err != nil {
		return nil, fmt.Errorf("compose process runtime: %w", err)
	}
	result := &processRuntime{
		core: core, terminals: core.ResponseTerminals(), codecs: core.ProtocolCodecs(),
	}
	dispatch, ok := core.DispatchCapabilities().(extproc.DispatchCapabilityRuntime)
	if !ok || dispatch == nil {
		_ = core.Close()
		return nil, errors.New("backend dispatch authority is unavailable")
	}
	result.dispatch = dispatch
	if cfg != nil && cfg.ControlPlane.Mode == config.ControlPlaneModeManaged {
		routers, composeProcessRuntimeErr := extproc.NewManagedRouterRegistry(extproc.ManagedRouterRegistryOptions{
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
			return nil, fmt.Errorf("compose managed Router generations: %w", composeProcessRuntimeErr)
		}
		if err := core.AttachRoutingSnapshots(routers); err != nil {
			_ = routers.Close()
			_ = core.Close()
			return nil, fmt.Errorf("attach managed Router generations: %w", err)
		}
		requests, composeProcessRuntimeErr := extproc.NewManagedRequestRuntime(extproc.ManagedRequestRuntimeOptions{
			Access: core.InferenceAccess(), PublicNamespaceID: cfg.ControlPlane.PublicNamespaceID,
			Publications: core, Routers: routers, Dispatch: dispatch,
		})
		if composeProcessRuntimeErr != nil {
			_ = routers.Close()
			_ = core.Close()
			return nil, fmt.Errorf("compose managed request routing: %w", composeProcessRuntimeErr)
		}
		result.routers = routers
		result.requests = requests
		result.outcomes = core.OutcomeFeedback()
		result.projections = core.OutcomeProjection()
	} else {
		if cfg == nil || cfg.RoutingSnapshot == nil {
			_ = core.Close()
			return nil, errors.New("standalone routing snapshot is unavailable")
		}
		requests, err := extproc.NewStandaloneRequestRuntime(extproc.StandaloneRequestRuntimeOptions{
			NamespaceID:  cfg.RoutingSnapshot.NamespaceID,
			Publications: core,
			Dispatch:     dispatch,
		})
		if err != nil {
			_ = core.Close()
			return nil, fmt.Errorf("compose standalone request routing: %w", err)
		}
		result.standaloneRequests = requests
	}
	return result, nil
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

func (runtime *processRuntime) ManagedAPI() managedruntime.ManagedAPI {
	if runtime == nil || runtime.core == nil {
		return nil
	}
	return runtime.core.ManagedAPI()
}

func (runtime *processRuntime) ManagedRequests() *extproc.ManagedRequestRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.requests
}

func (runtime *processRuntime) StandaloneRequests() *extproc.StandaloneRequestRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.standaloneRequests
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
