package config

import "fmt"

// ValidateRuntimeBootstrap validates the process composition implied by the
// configured services and stores. Authority is derived from store presence;
// the public configuration has no deployment-mode selector.
func (cfg *RouterConfig) ValidateRuntimeBootstrap() error {
	if cfg == nil {
		return nil
	}

	durableRouting := cfg.AccessStore != nil
	distributedRuntime := cfg.AccessRuntimeStore != nil
	nativeAccess := cfg.Access.Enabled

	if cfg.ManagementAPI.Enabled && !durableRouting {
		return fmt.Errorf("global.services.management_api.enabled requires global.stores.management.postgres")
	}
	if distributedRuntime && !durableRouting {
		return fmt.Errorf("global.stores.runtime.redis requires global.stores.management.postgres")
	}
	if nativeAccess && (!durableRouting || !distributedRuntime) {
		return fmt.Errorf("global.services.access.enabled requires global.stores.management.postgres and global.stores.runtime.redis")
	}

	if err := validateAccessStore(cfg.AccessStore); err != nil {
		return err
	}
	if err := validateAccessRuntimeStore(cfg.AccessRuntimeStore); err != nil {
		return err
	}
	if err := validateAccessService(cfg.Access); err != nil {
		return err
	}
	if err := validateAgentService(nativeAccess, cfg.Agent); err != nil {
		return err
	}
	if err := cfg.ManagementAPI.validateAuthMode(); err != nil {
		return err
	}
	if err := cfg.ManagementAPI.validateAuthModeForAuthority(durableRouting); err != nil {
		return err
	}

	backendDispatch := cfg.BackendDispatch
	if backendDispatch == (BackendDispatchConfig{}) {
		backendDispatch = DefaultBackendDispatchConfig()
	}
	if err := validateBackendDispatch(backendDispatch); err != nil {
		return err
	}
	if err := validateRoutingSecurity(durableRouting, cfg.RoutingSecurity); err != nil {
		return err
	}
	if err := validateBackendBootstrap(
		durableRouting, cfg.BackendCredentials, cfg.BackendEgress,
	); err != nil {
		return err
	}
	if err := validateBackendCredentialRefs(
		durableRouting, cfg.BackendCredentials, cfg.RoutingSnapshot,
	); err != nil {
		return err
	}
	return validateManagementBootstrapSecurity(cfg.ManagementAPI)
}
