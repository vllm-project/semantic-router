package config

import (
	"fmt"
	"strings"
)

// ManagementAPITLSConfig contains references to Router-terminated Management
// listener TLS material. Literal certificates and keys are not accepted.
type ManagementAPITLSConfig struct {
	CertificateFile    string `yaml:"certificate_file,omitempty"`
	CertificateEnv     string `yaml:"certificate_env,omitempty"`
	PrivateKeyFile     string `yaml:"private_key_file,omitempty"`
	PrivateKeyEnv      string `yaml:"private_key_env,omitempty"`
	ClientCABundleFile string `yaml:"client_ca_bundle_file,omitempty"`
	ClientCABundleEnv  string `yaml:"client_ca_bundle_env,omitempty"`
}

type ManagementAPIBootstrapConfig struct {
	TokenFile                     string `yaml:"token_file,omitempty"`
	TokenEnv                      string `yaml:"token_env,omitempty"`
	DisableAfterFirstClusterAdmin bool   `yaml:"disable_after_first_cluster_admin"`
}

type ManagementAPIRecoveryConfig struct {
	Enabled      bool   `yaml:"enabled"`
	TokenFile    string `yaml:"token_file,omitempty"`
	TokenEnv     string `yaml:"token_env,omitempty"`
	LoopbackOnly bool   `yaml:"loopback_only"`
}

func (c *ManagementAPIConfig) applySecurityDefaults() {
	if c == nil {
		return
	}
	// These booleans are safe invariants, not secret-bearing environment
	// guesses. Secret references intentionally have no implicit path.
	if !c.Auth.Bootstrap.DisableAfterFirstClusterAdmin &&
		c.Auth.Bootstrap.TokenFile == "" && c.Auth.Bootstrap.TokenEnv == "" {
		c.Auth.Bootstrap.DisableAfterFirstClusterAdmin = true
	}
	if !c.Auth.Recovery.Enabled && !c.Auth.Recovery.LoopbackOnly &&
		c.Auth.Recovery.TokenFile == "" && c.Auth.Recovery.TokenEnv == "" {
		c.Auth.Recovery.LoopbackOnly = true
	}
}

func validateManagementBootstrapSecurity(management ManagementAPIConfig) error {
	if err := validateManagementTLS(management.TLS, management.Enabled); err != nil {
		return err
	}
	auth := management.Auth
	keyrings := []struct {
		path     string
		file     string
		env      string
		required bool
	}{
		{"global.services.management_api.auth.token_signing_keyring", auth.TokenSigningKeyringFile, auth.TokenSigningKeyringEnv, management.Enabled},
		{"global.services.management_api.auth.service_account_hmac_keyring", auth.ServiceAccountHMACKeyringFile, auth.ServiceAccountHMACKeyringEnv, management.Enabled},
		{"global.services.management_api.auth.invitation_hmac_keyring", auth.InvitationHMACKeyringFile, auth.InvitationHMACKeyringEnv, management.Enabled},
		{"global.services.management_api.auth.response_kek_keyring", auth.ResponseKEKKeyringFile, auth.ResponseKEKKeyringEnv, management.Enabled},
	}
	for _, keyring := range keyrings {
		if err := validateSecretSource(keyring.path, keyring.file, keyring.env, keyring.required); err != nil {
			return err
		}
	}
	bootstrapConfigured := auth.Bootstrap.TokenFile != "" || auth.Bootstrap.TokenEnv != ""
	if err := validateSecretSource(
		"global.services.management_api.auth.bootstrap.token",
		auth.Bootstrap.TokenFile,
		auth.Bootstrap.TokenEnv,
		false,
	); err != nil {
		return err
	}
	if bootstrapConfigured && !auth.Bootstrap.DisableAfterFirstClusterAdmin {
		return fmt.Errorf("global.services.management_api.auth.bootstrap.disable_after_first_cluster_admin must be true")
	}
	if bootstrapConfigured && !management.Enabled {
		return fmt.Errorf("management bootstrap token requires management_api.enabled=true")
	}

	recoveryConfigured := auth.Recovery.TokenFile != "" || auth.Recovery.TokenEnv != ""
	if err := validateSecretSource(
		"global.services.management_api.auth.recovery.token",
		auth.Recovery.TokenFile,
		auth.Recovery.TokenEnv,
		auth.Recovery.Enabled,
	); err != nil {
		return err
	}
	if !auth.Recovery.Enabled && recoveryConfigured {
		return fmt.Errorf("global.services.management_api.auth.recovery token requires enabled=true")
	}
	if auth.Recovery.Enabled && !management.Enabled {
		return fmt.Errorf("management recovery requires management_api.enabled=true")
	}
	if auth.Recovery.Enabled && !auth.Recovery.LoopbackOnly {
		return fmt.Errorf("global.services.management_api.auth.recovery.loopback_only must be true")
	}
	if bootstrapConfigured && recoveryConfigured && sameSecretSource(
		auth.Bootstrap.TokenFile,
		auth.Bootstrap.TokenEnv,
		auth.Recovery.TokenFile,
		auth.Recovery.TokenEnv,
	) {
		return fmt.Errorf("management bootstrap and recovery must use separate token references")
	}
	return nil
}

func validateManagementTLS(tls ManagementAPITLSConfig, required bool) error {
	if err := validateSecretSource("global.services.management_api.tls.certificate", tls.CertificateFile, tls.CertificateEnv, required); err != nil {
		return err
	}
	if err := validateSecretSource("global.services.management_api.tls.private_key", tls.PrivateKeyFile, tls.PrivateKeyEnv, required); err != nil {
		return err
	}
	if err := validateSecretSource("global.services.management_api.tls.client_ca_bundle", tls.ClientCABundleFile, tls.ClientCABundleEnv, false); err != nil {
		return err
	}
	certificateConfigured := tls.CertificateFile != "" || tls.CertificateEnv != ""
	privateKeyConfigured := tls.PrivateKeyFile != "" || tls.PrivateKeyEnv != ""
	if certificateConfigured != privateKeyConfigured {
		return fmt.Errorf("global.services.management_api.tls certificate and private key must be configured together")
	}
	return nil
}

func sameSecretSource(leftFile, leftEnv, rightFile, rightEnv string) bool {
	leftFile, leftEnv = strings.TrimSpace(leftFile), strings.TrimSpace(leftEnv)
	rightFile, rightEnv = strings.TrimSpace(rightFile), strings.TrimSpace(rightEnv)
	return (leftFile != "" && leftFile == rightFile) || (leftEnv != "" && leftEnv == rightEnv)
}
