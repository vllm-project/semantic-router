package routingruntime

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const maximumScalarSecretBytes = 64 << 10

func loadDeploymentKeyrings(cfg *config.RouterConfig) (DeploymentKeyrings, error) {
	capabilities, err := runtimecapabilities.Derive(cfg)
	if err != nil {
		return DeploymentKeyrings{}, err
	}
	if !capabilities.DurableRouting {
		return DeploymentKeyrings{}, errors.New("deployment keyrings require durable routing")
	}
	var result DeploymentKeyrings
	fail := func(domain string, err error) (DeploymentKeyrings, error) {
		result.zero()
		return DeploymentKeyrings{}, fmt.Errorf("load %s keyring: %w", domain, err)
	}

	if cfg.Access.Enabled {
		apiKeys, loadErr := loadSymmetricKeyring(
			cfg.Access.Credentials.APIKeyHMACKeyringFile,
			cfg.Access.Credentials.APIKeyHMACKeyringEnv,
		)
		if loadErr != nil {
			return fail("API-key HMAC", loadErr)
		}
		result.APIKeyPeppers = pepperKeyring(apiKeys)
		delegation, loadErr := loadSymmetricKeyring(
			cfg.Access.Credentials.DelegationHMACKeyringFile,
			cfg.Access.Credentials.DelegationHMACKeyringEnv,
		)
		if loadErr != nil {
			return fail("delegation HMAC", loadErr)
		}
		result.DelegationPeppers = pepperKeyring(delegation)
		tenantSigning, loadErr := loadSigningKeyring(
			cfg.Access.TenantContext.SigningKeyFile,
			cfg.Access.TenantContext.SigningKeyEnv,
		)
		if loadErr != nil {
			return fail("tenant-context signing", loadErr)
		}
		result.TenantContextSigning = tenantSigning
		if cfg.Access.Credentials.Reveal.Enabled {
			reveal, revealErr := loadSymmetricKeyring(
				cfg.Access.Credentials.Reveal.KEKKeyringFile,
				cfg.Access.Credentials.Reveal.KEKKeyringEnv,
			)
			if revealErr != nil {
				return fail("inference-key reveal KEK", revealErr)
			}
			value := kekKeyring(reveal)
			result.RevealKEK = &value
		}
	}

	provider, err := loadSymmetricKeyring(
		cfg.BackendCredentials.ProviderKEKKeyringFile,
		cfg.BackendCredentials.ProviderKEKKeyringEnv,
	)
	if err != nil {
		return fail("provider-credential KEK", err)
	}
	result.ProviderKEK = kekKeyring(provider)

	if capabilities.ManagementAPI {
		auth := cfg.ManagementAPI.Auth
		managementSigning, signingErr := loadSigningKeyring(auth.TokenSigningKeyringFile, auth.TokenSigningKeyringEnv)
		if signingErr != nil {
			return fail("Management token signing", signingErr)
		}
		result.ManagementSigning = managementSigning
		serviceAccounts, serviceErr := loadSymmetricKeyring(auth.ServiceAccountHMACKeyringFile, auth.ServiceAccountHMACKeyringEnv)
		if serviceErr != nil {
			return fail("Management service-account HMAC", serviceErr)
		}
		result.ServiceAccounts = pepperKeyring(serviceAccounts)
		invitations, invitationErr := loadSymmetricKeyring(auth.InvitationHMACKeyringFile, auth.InvitationHMACKeyringEnv)
		if invitationErr != nil {
			return fail("Management invitation HMAC", invitationErr)
		}
		result.Invitations = pepperKeyring(invitations)
		responses, responseErr := loadSymmetricKeyring(auth.ResponseKEKKeyringFile, auth.ResponseKEKKeyringEnv)
		if responseErr != nil {
			return fail("Management response KEK", responseErr)
		}
		result.ResponseKEK = kekKeyring(responses)
	}

	root, err := loadSymmetricKeyring(
		cfg.RoutingSecurity.HMACKeyringFile,
		cfg.RoutingSecurity.HMACKeyringEnv,
	)
	if err != nil {
		return fail("routing HMAC root", err)
	}
	derived, err := deriveRoutingKeyrings(root)
	zeroBytesMap(root.Keys)
	if err != nil {
		return fail("routing HMAC root", err)
	}
	result.Routing = derived
	return result, nil
}

func loadSymmetricKeyring(file, environment string) (securitykeyring.Symmetric, error) {
	payload, err := (securitykeyring.Source{File: file, Env: environment}).Read()
	if err != nil {
		return securitykeyring.Symmetric{}, err
	}
	defer zero(payload)
	return securitykeyring.ParseSymmetric(payload, 32)
}

func loadSigningKeyring(file, environment string) (securitykeyring.Signing, error) {
	payload, err := (securitykeyring.Source{File: file, Env: environment}).Read()
	if err != nil {
		return securitykeyring.Signing{}, err
	}
	defer zero(payload)
	return securitykeyring.ParseSigning(payload)
}

func pepperKeyring(source securitykeyring.Symmetric) accesscredential.PepperKeyring {
	return accesscredential.PepperKeyring{ActiveVersion: source.ActiveVersion, Keys: cloneBytesMap(source.Keys)}
}

func kekKeyring(source securitykeyring.Symmetric) accesscredential.KEKKeyring {
	return accesscredential.KEKKeyring{ActiveVersion: source.ActiveVersion, Keys: cloneBytesMap(source.Keys)}
}

func readScalarSecret(file, environment string) (string, error) {
	if (file == "") == (environment == "") {
		return "", errors.New("exactly one secret source is required")
	}
	var payload []byte
	var err error
	if file != "" {
		payload, err = os.ReadFile(file)
		if err != nil {
			return "", errors.New("read secret file")
		}
	} else {
		value, found := os.LookupEnv(environment)
		if !found {
			return "", errors.New("secret environment source is not set")
		}
		payload = []byte(value)
	}
	defer zero(payload)
	if len(payload) == 0 || len(payload) > maximumScalarSecretBytes || strings.ContainsRune(string(payload), '\x00') {
		return "", errors.New("secret value is empty or invalid")
	}
	value := strings.TrimSpace(string(payload))
	if value == "" || strings.ContainsAny(value, "\r\n\t") {
		return "", errors.New("secret value must contain one canonical line")
	}
	return value, nil
}

func readOptionalScalarSecret(file, environment string) ([]byte, error) {
	if file == "" && environment == "" {
		return nil, nil
	}
	value, err := readScalarSecret(file, environment)
	if err != nil {
		return nil, err
	}
	return []byte(value), nil
}

// readBootstrapToken returns both the initial credential and the exact source
// restarting the Router. A missing or empty file is a valid finalized state;
// BootstrapService readiness decides whether that state is legal for the
// durable installation marker. Environment-backed authorities remain present
// for the lifetime of the process and therefore require a deployment rollout
// to finalize.
func readBootstrapToken(file, environment string) ([]byte, func() (bool, error), error) {
	if file == "" && environment == "" {
		return nil, func() (bool, error) { return false, nil }, nil
	}
	if file != "" && environment != "" {
		return nil, nil, errors.New("bootstrap token requires exactly one source")
	}
	if environment != "" {
		value, found := os.LookupEnv(environment)
		if !found || strings.TrimSpace(value) == "" {
			return nil, func() (bool, error) { return false, nil }, nil
		}
		payload, err := readOptionalScalarSecret("", environment)
		return payload, func() (bool, error) { return true, nil }, err
	}
	present := func() (bool, error) {
		info, err := os.Lstat(file)
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		if err != nil {
			return false, fmt.Errorf("inspect bootstrap token file: %w", err)
		}
		if !info.Mode().IsRegular() {
			return false, errors.New("bootstrap token path is not a regular file")
		}
		if info.Mode().Perm()&0o077 != 0 {
			return false, errors.New("bootstrap token file must be owner-readable only")
		}
		return info.Size() > 0, nil
	}
	exists, err := present()
	if err != nil || !exists {
		return nil, present, err
	}
	payload, err := readOptionalScalarSecret(file, "")
	return payload, present, err
}

func readRecoveryToken(enabled bool, file, environment string) ([]byte, error) {
	if !enabled {
		return nil, nil
	}
	value, err := readOptionalScalarSecret(file, environment)
	if err != nil {
		return nil, err
	}
	if len(value) < 32 {
		zero(value)
		return nil, errors.New("recovery token must contain at least 32 bytes")
	}
	return value, nil
}
