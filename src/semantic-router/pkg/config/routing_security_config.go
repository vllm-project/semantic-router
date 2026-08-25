package config

import "fmt"

// RoutingSecurityConfig references the versioned root used to derive
// publication, Provider Catalog, and backend-dispatch HMAC domains. It belongs
// to the durable routing runtime, independently of whether the Management API
// listener is enabled.
type RoutingSecurityConfig struct {
	HMACKeyringFile string `yaml:"hmac_keyring_file,omitempty"`
	HMACKeyringEnv  string `yaml:"hmac_keyring_env,omitempty"`
}

func validateRoutingSecurity(durableRouting bool, security RoutingSecurityConfig) error {
	configured := security.HMACKeyringFile != "" || security.HMACKeyringEnv != ""
	if err := validateSecretSource(
		"global.services.routing_security.hmac_keyring",
		security.HMACKeyringFile,
		security.HMACKeyringEnv,
		durableRouting,
	); err != nil {
		return err
	}
	if !durableRouting && configured {
		return fmt.Errorf(
			"global.services.routing_security.hmac_keyring requires global.stores.management.postgres",
		)
	}
	return nil
}
