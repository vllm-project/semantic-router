// Package runtimecapabilities derives process composition from configured
// services and stores. It deliberately has no serialized mode selector: the
// presence of an authority or store is the capability contract.
package runtimecapabilities

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RuntimeCapabilities is immutable process-composition state derived from one
// validated Router bootstrap. It is not part of the public YAML contract.
type RuntimeCapabilities struct {
	FileRouting      bool
	DurableRouting   bool
	ManagementAPI    bool
	DistributedState bool
	NativeAccess     bool
}

// Derive resolves the supported capability matrix from configured services and
// stores. It validates cross-component requirements before a caller opens any
// durable resource.
func Derive(cfg *config.RouterConfig) (RuntimeCapabilities, error) {
	if cfg == nil {
		return RuntimeCapabilities{}, errors.New("router configuration is required")
	}

	capabilities := RuntimeCapabilities{
		DurableRouting:   cfg.AccessStore != nil,
		ManagementAPI:    cfg.ManagementAPI.Enabled,
		DistributedState: cfg.AccessRuntimeStore != nil,
		NativeAccess:     cfg.Access.Enabled,
	}
	capabilities.FileRouting = !capabilities.DurableRouting

	if capabilities.ManagementAPI && !capabilities.DurableRouting {
		return RuntimeCapabilities{}, errors.New(
			"global.services.management_api.enabled requires global.stores.management.postgres")
	}
	if capabilities.DistributedState && !capabilities.DurableRouting {
		return RuntimeCapabilities{}, errors.New(
			"global.stores.runtime.redis requires global.stores.management.postgres")
	}
	if capabilities.NativeAccess &&
		(!capabilities.DurableRouting || !capabilities.DistributedState) {
		return RuntimeCapabilities{}, errors.New(
			"global.services.access.enabled requires global.stores.management.postgres and global.stores.runtime.redis")
	}
	return capabilities, nil
}
