package config

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
)

const (
	ControlPlaneModeStandalone          = "standalone"
	ControlPlaneModeManaged             = "managed"
	defaultProviderCatalogLease         = "45s"
	defaultProviderCatalogRenewInterval = "15s"
)

// ControlPlaneConfig selects the single authority for routing and access
// desired state. Standalone reads one immutable routing manifest; managed mode
// obtains mutable state from the Router-owned control plane.
type ControlPlaneConfig struct {
	Mode string `yaml:"mode,omitempty"`
	// PublicNamespaceID selects the one published namespace exposed by a
	// managed routing-only deployment. Managed access derives the namespace
	// from the authenticated key and therefore forbids this value.
	PublicNamespaceID string                         `yaml:"public_namespace_id,omitempty"`
	ProviderCatalog   ProviderCatalogBootstrapConfig `yaml:"provider_catalog,omitempty"`
}

type ProviderCatalogBootstrapConfig struct {
	ReplicaIDEnv          string                              `yaml:"replica_id_env,omitempty"`
	Lease                 string                              `yaml:"lease,omitempty"`
	RenewInterval         string                              `yaml:"renew_interval,omitempty"`
	RolloutGroups         []ProviderCatalogRolloutGroupConfig `yaml:"rollout_groups,omitempty"`
	RequiredRolloutGroups []ProviderCatalogRolloutGroupConfig `yaml:"required_rollout_groups,omitempty"`
}

type ProviderCatalogRolloutGroupConfig struct {
	Plane string `yaml:"plane"`
	ID    string `yaml:"id"`
}

var providerCatalogRolloutGroupPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)

func DefaultControlPlaneConfig() ControlPlaneConfig {
	return ControlPlaneConfig{Mode: ControlPlaneModeStandalone}
}

func cloneControlPlaneConfig(source ControlPlaneConfig) ControlPlaneConfig {
	cloned := source
	cloned.ProviderCatalog.RolloutGroups = append(
		[]ProviderCatalogRolloutGroupConfig(nil), source.ProviderCatalog.RolloutGroups...,
	)
	cloned.ProviderCatalog.RequiredRolloutGroups = append(
		[]ProviderCatalogRolloutGroupConfig(nil), source.ProviderCatalog.RequiredRolloutGroups...,
	)
	return cloned
}

func applyControlPlaneBootstrapDefaults(global *CanonicalGlobal) {
	if global == nil {
		return
	}
	if global.ControlPlane.Mode == "" {
		global.ControlPlane.Mode = ControlPlaneModeStandalone
	}
	if global.ControlPlane.Mode == ControlPlaneModeManaged {
		if global.ControlPlane.ProviderCatalog.Lease == "" {
			global.ControlPlane.ProviderCatalog.Lease = defaultProviderCatalogLease
		}
		if global.ControlPlane.ProviderCatalog.RenewInterval == "" {
			global.ControlPlane.ProviderCatalog.RenewInterval = defaultProviderCatalogRenewInterval
		}
	}
	applyAccessStoreDefaults(global.Stores.Access, global.Stores.AccessRuntime)
	applyAccessServiceDefaults(&global.Services.Access)
	global.Services.ManagementAPI.applyV04SecurityDefaults()
}

// ValidateControlPlaneBootstrap validates the independent v0.4 bootstrap
// layer without starting stores or reading secret files.
func (cfg *RouterConfig) ValidateControlPlaneBootstrap() error {
	if cfg == nil {
		return nil
	}
	mode := strings.TrimSpace(cfg.ControlPlane.Mode)
	if mode == "" {
		mode = ControlPlaneModeStandalone
	}
	if mode != cfg.ControlPlane.Mode && cfg.ControlPlane.Mode != "" {
		return fmt.Errorf("global.control_plane.mode must not contain surrounding whitespace")
	}
	if mode != ControlPlaneModeStandalone && mode != ControlPlaneModeManaged {
		return fmt.Errorf("global.control_plane.mode must be standalone or managed")
	}
	if err := validatePublicNamespace(mode, cfg.Access.Enabled, cfg.ControlPlane.PublicNamespaceID); err != nil {
		return err
	}
	if err := validateProviderCatalogBootstrap(mode, cfg.ControlPlane.ProviderCatalog); err != nil {
		return err
	}
	backendDispatch := cfg.BackendDispatch
	if backendDispatch == (BackendDispatchConfig{}) {
		backendDispatch = DefaultBackendDispatchConfig()
	}
	if err := validateBackendDispatch(backendDispatch); err != nil {
		return err
	}

	if mode == ControlPlaneModeStandalone {
		return validateStandaloneBootstrap(cfg)
	}
	return validateManagedBootstrap(cfg)
}

func validatePublicNamespace(mode string, accessEnabled bool, value string) error {
	if value != strings.TrimSpace(value) {
		return fmt.Errorf("global.control_plane.public_namespace_id must not contain surrounding whitespace")
	}
	if mode == ControlPlaneModeStandalone {
		if value != "" {
			return fmt.Errorf("global.control_plane.public_namespace_id is managed routing-only")
		}
		return nil
	}
	if accessEnabled {
		if value != "" {
			return fmt.Errorf("global.control_plane.public_namespace_id is forbidden when managed access is enabled")
		}
		return nil
	}
	parsed, err := uuid.Parse(value)
	if err != nil || parsed == uuid.Nil || parsed.String() != value {
		return fmt.Errorf("managed routing-only requires a canonical global.control_plane.public_namespace_id UUID")
	}
	return nil
}

func validateProviderCatalogBootstrap(mode string, catalog ProviderCatalogBootstrapConfig) error {
	configured := catalog.ReplicaIDEnv != "" || catalog.Lease != "" || catalog.RenewInterval != "" ||
		len(catalog.RolloutGroups) != 0 || len(catalog.RequiredRolloutGroups) != 0
	if mode == ControlPlaneModeStandalone {
		if configured {
			return fmt.Errorf("global.control_plane.provider_catalog is managed-only")
		}
		return nil
	}
	if !bootstrapEnvNamePattern.MatchString(catalog.ReplicaIDEnv) {
		return fmt.Errorf("global.control_plane.provider_catalog.replica_id_env must name an environment variable")
	}
	lease, err := time.ParseDuration(catalog.Lease)
	if err != nil || lease < time.Second || lease > 5*time.Minute {
		return fmt.Errorf("global.control_plane.provider_catalog.lease must be between 1s and 5m")
	}
	renew, err := time.ParseDuration(catalog.RenewInterval)
	if err != nil || renew < time.Second || renew >= lease {
		return fmt.Errorf("global.control_plane.provider_catalog.renew_interval must be at least 1s and shorter than lease")
	}
	if err := validateProviderCatalogGroups("rollout_groups", catalog.RolloutGroups); err != nil {
		return err
	}
	return validateProviderCatalogGroups("required_rollout_groups", catalog.RequiredRolloutGroups)
}

func validateProviderCatalogGroups(path string, groups []ProviderCatalogRolloutGroupConfig) error {
	if len(groups) == 0 {
		return fmt.Errorf("global.control_plane.provider_catalog.%s requires at least one group", path)
	}
	seen := make(map[string]struct{}, len(groups))
	for index, group := range groups {
		if (group.Plane != "control" && group.Plane != "data") || !providerCatalogRolloutGroupPattern.MatchString(group.ID) {
			return fmt.Errorf("global.control_plane.provider_catalog.%s[%d] is invalid", path, index)
		}
		key := group.Plane + "/" + group.ID
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("global.control_plane.provider_catalog.%s contains duplicate group %q", path, key)
		}
		seen[key] = struct{}{}
	}
	return nil
}

func validateStandaloneBootstrap(cfg *RouterConfig) error {
	if err := validateAgentService(ControlPlaneModeStandalone, cfg.Agent); err != nil {
		return err
	}
	if err := validateManagementAuthControlPlane(ControlPlaneModeStandalone, cfg.ManagementAPI); err != nil {
		return err
	}
	if cfg.AccessStore != nil || cfg.AccessRuntimeStore != nil {
		return fmt.Errorf("global.control_plane.mode standalone rejects global.stores.access and global.stores.access_runtime")
	}
	if cfg.Access.Enabled {
		return fmt.Errorf("global.control_plane.mode standalone requires global.services.access.enabled=false")
	}
	if err := validateAccessService(cfg.Access); err != nil {
		return err
	}
	if err := validateBackendBootstrap(ControlPlaneModeStandalone, cfg.BackendCredentials, cfg.BackendEgress); err != nil {
		return err
	}
	if err := validateBackendCredentialRefs(ControlPlaneModeStandalone, cfg.BackendCredentials, cfg.RoutingSnapshot); err != nil {
		return err
	}
	return validateManagementBootstrapSecurity(ControlPlaneModeStandalone, cfg.ManagementAPI)
}

func validateManagedBootstrap(cfg *RouterConfig) error {
	if err := validateAgentService(ControlPlaneModeManaged, cfg.Agent); err != nil {
		return err
	}
	if err := validateManagementAuthControlPlane(ControlPlaneModeManaged, cfg.ManagementAPI); err != nil {
		return err
	}
	if cfg.AccessStore == nil || cfg.AccessStore.Type != AccessStoreTypePostgres {
		return fmt.Errorf("managed mode requires global.stores.access.type=postgres")
	}
	if cfg.AccessRuntimeStore == nil || cfg.AccessRuntimeStore.Type != AccessRuntimeStoreTypeRedis {
		return fmt.Errorf("managed mode requires global.stores.access_runtime.type=redis")
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
	if err := validateBackendBootstrap(ControlPlaneModeManaged, cfg.BackendCredentials, cfg.BackendEgress); err != nil {
		return err
	}
	if err := validateBackendCredentialRefs(ControlPlaneModeManaged, cfg.BackendCredentials, cfg.RoutingSnapshot); err != nil {
		return err
	}
	return validateManagementBootstrapSecurity(ControlPlaneModeManaged, cfg.ManagementAPI)
}

func validateManagementAuthControlPlane(mode string, management ManagementAPIConfig) error {
	if err := management.validateAuthMode(); err != nil {
		return err
	}
	return management.validateAuthModeForControlPlane(mode)
}
