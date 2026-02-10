package config

import "fmt"

// GetEnabledAdapters returns a list of enabled adapters.
// If no adapters are configured, returns default adapters for backward compatibility.
func (c *RouterConfig) GetEnabledAdapters() []AdapterConfig {
	// If no adapters configured, use defaults for backward compatibility
	if len(c.Adapters) == 0 {
		return []AdapterConfig{
			{
				Type:    "envoy",
				Enabled: true,
				Port:    50051,
			},
		}
	}

	var enabled []AdapterConfig
	for _, adapter := range c.Adapters {
		if adapter.Enabled {
			enabled = append(enabled, adapter)
		}
	}
	return enabled
}

// GetAdapterByType returns the adapter configuration for a specific type
func (c *RouterConfig) GetAdapterByType(adapterType string) *AdapterConfig {
	for _, adapter := range c.Adapters {
		if adapter.Type == adapterType {
			return &adapter
		}
	}
	return nil
}

// HasEnabledAdapter checks if at least one adapter is enabled
func (c *RouterConfig) HasEnabledAdapter() bool {
	for _, adapter := range c.Adapters {
		if adapter.Enabled {
			return true
		}
	}
	return false
}

// ValidateAdapters validates adapter configuration
func (c *RouterConfig) ValidateAdapters() error {
	seenPorts := make(map[int]string)

	for _, adapter := range c.Adapters {
		if !adapter.Enabled {
			continue
		}

		// Validate adapter type
		if adapter.Type != "envoy" && adapter.Type != "http" && adapter.Type != "nginx" {
			return ErrInvalidAdapterType{Type: adapter.Type}
		}

		// Validate port
		if adapter.Port <= 0 || adapter.Port > 65535 {
			return ErrInvalidAdapterPort{Type: adapter.Type, Port: adapter.Port}
		}

		// Check for port conflicts
		if existingType, exists := seenPorts[adapter.Port]; exists {
			return ErrAdapterPortConflict{
				Port:  adapter.Port,
				Type1: existingType,
				Type2: adapter.Type,
			}
		}
		seenPorts[adapter.Port] = adapter.Type
	}

	return nil
}

// Adapter validation errors
type ErrInvalidAdapterType struct {
	Type string
}

func (e ErrInvalidAdapterType) Error() string {
	return "invalid adapter type: " + e.Type + " (must be 'envoy', 'http', or 'nginx')"
}

type ErrInvalidAdapterPort struct {
	Type string
	Port int
}

func (e ErrInvalidAdapterPort) Error() string {
	return fmt.Sprintf("invalid port %d for adapter %s (must be 1-65535)", e.Port, e.Type)
}

type ErrAdapterPortConflict struct {
	Port  int
	Type1 string
	Type2 string
}

func (e ErrAdapterPortConflict) Error() string {
	return fmt.Sprintf("port %d is used by both %s and %s adapters", e.Port, e.Type1, e.Type2)
}
