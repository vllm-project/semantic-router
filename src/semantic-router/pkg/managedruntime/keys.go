// Package managedruntime owns the process-wide resources used by a managed
// Router replica. It deliberately contains no Dashboard integration.
package managedruntime

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const derivedHMACKeyBytes = sha256.Size

const controlPlaneHKDFSalt = "vllm-sr/managed-control-plane/hkdf-sha256/v1"

type HMACDomain string

const (
	HMACDomainCatalogCursor        HMACDomain = "catalog-cursor"
	HMACDomainDiscoveryClaim       HMACDomain = "discovery-claim"
	HMACDomainManagementCommand    HMACDomain = "management-command"
	HMACDomainManagementCursor     HMACDomain = "management-cursor"
	HMACDomainAgentSecret          HMACDomain = "agent-secret"
	HMACDomainBootstrapIdempotency HMACDomain = "bootstrap-idempotency"
	HMACDomainBackendDispatch      HMACDomain = "backend-dispatch"
)

var controlPlaneHMACDomains = []HMACDomain{
	HMACDomainCatalogCursor,
	HMACDomainDiscoveryClaim,
	HMACDomainManagementCommand,
	HMACDomainManagementCursor,
	HMACDomainAgentSecret,
	HMACDomainBootstrapIdempotency,
	HMACDomainBackendDispatch,
}

// HMACKeyring is a domain-separated, versioned signer/verifier keyring.
// ActiveVersion is the only version used to sign. Retained versions may only
// verify wire values that explicitly carry their version.
type HMACKeyring struct {
	activeVersion string
	keys          map[string][]byte
}

func (keyring HMACKeyring) ActiveVersion() string { return keyring.activeVersion }

func (keyring HMACKeyring) Versions() []string {
	versions := make([]string, 0, len(keyring.keys))
	for version := range keyring.keys {
		versions = append(versions, version)
	}
	sort.Strings(versions)
	return versions
}

// Symmetric adapts the derived domain to codecs that consume the repository's
// standard versioned keyring value. Returned bytes are a defensive copy.
func (keyring HMACKeyring) Symmetric() securitykeyring.Symmetric {
	return securitykeyring.Symmetric{ActiveVersion: keyring.activeVersion, Keys: cloneBytesMap(keyring.keys)}
}

func (keyring HMACKeyring) activeKey() (string, []byte, error) {
	key, found := keyring.keys[keyring.activeVersion]
	if !found || len(key) != derivedHMACKeyBytes {
		return "", nil, errors.New("active control-plane HMAC key is unavailable")
	}
	return keyring.activeVersion, append([]byte(nil), key...), nil
}

func (keyring HMACKeyring) Sign(payload []byte) (string, []byte, error) {
	version, key, err := keyring.activeKey()
	if err != nil {
		return "", nil, err
	}
	defer zero(key)
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write(payload)
	return version, mac.Sum(nil), nil
}

func (keyring HMACKeyring) Verify(version string, payload, signature []byte) bool {
	key, found := keyring.keys[version]
	if !found || len(key) != derivedHMACKeyBytes || len(signature) != sha256.Size {
		return false
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write(payload)
	return subtle.ConstantTimeCompare(mac.Sum(nil), signature) == 1
}

func (keyring HMACKeyring) clone() HMACKeyring {
	return HMACKeyring{activeVersion: keyring.activeVersion, keys: cloneBytesMap(keyring.keys)}
}

func (keyring *HMACKeyring) zero() {
	if keyring == nil {
		return
	}
	zeroBytesMap(keyring.keys)
	keyring.activeVersion = ""
	keyring.keys = nil
}

// ControlPlaneKeyrings are independent HKDF domains derived from one dedicated
// deployment root. Root bytes never leave construction and are not reused as
// an application HMAC key.
type ControlPlaneKeyrings struct {
	CatalogCursor        HMACKeyring
	DiscoveryClaim       HMACKeyring
	ManagementCommand    HMACKeyring
	ManagementCursor     HMACKeyring
	AgentSecret          HMACKeyring
	BootstrapIdempotency HMACKeyring
	BackendDispatch      HMACKeyring
}

func deriveControlPlaneKeyrings(root securitykeyring.Symmetric) (ControlPlaneKeyrings, error) {
	if strings.TrimSpace(root.ActiveVersion) == "" || len(root.Keys) == 0 {
		return ControlPlaneKeyrings{}, errors.New("control-plane HMAC root keyring is empty")
	}
	if _, found := root.Keys[root.ActiveVersion]; !found {
		return ControlPlaneKeyrings{}, errors.New("active control-plane HMAC root version is unavailable")
	}
	byDomain := make(map[HMACDomain]HMACKeyring, len(controlPlaneHMACDomains))
	for _, domain := range controlPlaneHMACDomains {
		keyring := HMACKeyring{activeVersion: root.ActiveVersion, keys: make(map[string][]byte, len(root.Keys))}
		for version, rootKey := range root.Keys {
			if len(rootKey) < derivedHMACKeyBytes {
				return ControlPlaneKeyrings{}, fmt.Errorf("control-plane HMAC root %q is weaker than 256 bits", version)
			}
			info := []byte("vllm-sr/managed-control-plane/" + string(domain) + "/v1\x00" + version)
			keyring.keys[version] = hkdfSHA256(rootKey, []byte(controlPlaneHKDFSalt), info, derivedHMACKeyBytes)
		}
		byDomain[domain] = keyring
	}
	return ControlPlaneKeyrings{
		CatalogCursor: byDomain[HMACDomainCatalogCursor], DiscoveryClaim: byDomain[HMACDomainDiscoveryClaim],
		ManagementCommand: byDomain[HMACDomainManagementCommand], ManagementCursor: byDomain[HMACDomainManagementCursor],
		AgentSecret:          byDomain[HMACDomainAgentSecret],
		BootstrapIdempotency: byDomain[HMACDomainBootstrapIdempotency],
		BackendDispatch:      byDomain[HMACDomainBackendDispatch],
	}, nil
}

func (keyrings ControlPlaneKeyrings) clone() ControlPlaneKeyrings {
	return ControlPlaneKeyrings{
		CatalogCursor: keyrings.CatalogCursor.clone(), DiscoveryClaim: keyrings.DiscoveryClaim.clone(),
		ManagementCommand: keyrings.ManagementCommand.clone(), ManagementCursor: keyrings.ManagementCursor.clone(),
		AgentSecret:          keyrings.AgentSecret.clone(),
		BootstrapIdempotency: keyrings.BootstrapIdempotency.clone(),
		BackendDispatch:      keyrings.BackendDispatch.clone(),
	}
}

func (keyrings *ControlPlaneKeyrings) zero() {
	if keyrings == nil {
		return
	}
	keyrings.CatalogCursor.zero()
	keyrings.DiscoveryClaim.zero()
	keyrings.ManagementCommand.zero()
	keyrings.ManagementCursor.zero()
	keyrings.AgentSecret.zero()
	keyrings.BootstrapIdempotency.zero()
	keyrings.BackendDispatch.zero()
}

// DeploymentKeyrings keeps each cryptographic authority in its own domain.
// The Management factory receives a defensive clone and must not retain or
// mutate bytes after the Runtime closes.
type DeploymentKeyrings struct {
	APIKeyPeppers        accesscredential.PepperKeyring
	DelegationPeppers    accesscredential.PepperKeyring
	TenantContextSigning securitykeyring.Signing
	ManagementSigning    securitykeyring.Signing
	ServiceAccounts      accesscredential.PepperKeyring
	Invitations          accesscredential.PepperKeyring
	ResponseKEK          accesscredential.KEKKeyring
	ProviderKEK          accesscredential.KEKKeyring
	RevealKEK            *accesscredential.KEKKeyring
	ControlPlane         ControlPlaneKeyrings
}

func (keyrings DeploymentKeyrings) clone() DeploymentKeyrings {
	cloned := DeploymentKeyrings{
		APIKeyPeppers: clonePepper(keyrings.APIKeyPeppers), DelegationPeppers: clonePepper(keyrings.DelegationPeppers),
		TenantContextSigning: cloneSigning(keyrings.TenantContextSigning), ManagementSigning: cloneSigning(keyrings.ManagementSigning),
		ServiceAccounts: clonePepper(keyrings.ServiceAccounts), Invitations: clonePepper(keyrings.Invitations),
		ResponseKEK: cloneKEK(keyrings.ResponseKEK), ProviderKEK: cloneKEK(keyrings.ProviderKEK),
		ControlPlane: keyrings.ControlPlane.clone(),
	}
	if keyrings.RevealKEK != nil {
		value := cloneKEK(*keyrings.RevealKEK)
		cloned.RevealKEK = &value
	}
	return cloned
}

func (keyrings *DeploymentKeyrings) zero() {
	if keyrings == nil {
		return
	}
	zeroBytesMap(keyrings.APIKeyPeppers.Keys)
	zeroBytesMap(keyrings.DelegationPeppers.Keys)
	zeroSigning(&keyrings.TenantContextSigning)
	zeroSigning(&keyrings.ManagementSigning)
	zeroBytesMap(keyrings.ServiceAccounts.Keys)
	zeroBytesMap(keyrings.Invitations.Keys)
	zeroBytesMap(keyrings.ResponseKEK.Keys)
	zeroBytesMap(keyrings.ProviderKEK.Keys)
	if keyrings.RevealKEK != nil {
		zeroBytesMap(keyrings.RevealKEK.Keys)
	}
	keyrings.ControlPlane.zero()
	*keyrings = DeploymentKeyrings{}
}

func hkdfSHA256(secret, salt, info []byte, length int) []byte {
	extract := hmac.New(sha256.New, salt)
	_, _ = extract.Write(secret)
	prk := extract.Sum(nil)
	defer zero(prk)
	result := make([]byte, 0, length)
	previous := []byte(nil)
	for counter := byte(1); len(result) < length; counter++ {
		expand := hmac.New(sha256.New, prk)
		_, _ = expand.Write(previous)
		_, _ = expand.Write(info)
		_, _ = expand.Write([]byte{counter})
		zero(previous)
		previous = expand.Sum(nil)
		remaining := length - len(result)
		if remaining > len(previous) {
			remaining = len(previous)
		}
		result = append(result, previous[:remaining]...)
	}
	zero(previous)
	return result
}

func clonePepper(source accesscredential.PepperKeyring) accesscredential.PepperKeyring {
	return accesscredential.PepperKeyring{ActiveVersion: source.ActiveVersion, Keys: cloneBytesMap(source.Keys)}
}

func cloneKEK(source accesscredential.KEKKeyring) accesscredential.KEKKeyring {
	return accesscredential.KEKKeyring{ActiveVersion: source.ActiveVersion, Keys: cloneBytesMap(source.Keys)}
}

func cloneSigning(source securitykeyring.Signing) securitykeyring.Signing {
	cloned := securitykeyring.Signing{
		ActiveVersion: source.ActiveVersion,
		Private:       make(map[string]ed25519.PrivateKey, len(source.Private)),
		Public:        make(map[string]ed25519.PublicKey, len(source.Public)),
	}
	for version, key := range source.Private {
		cloned.Private[version] = ed25519.PrivateKey(append([]byte(nil), key...))
	}
	for version, key := range source.Public {
		cloned.Public[version] = ed25519.PublicKey(append([]byte(nil), key...))
	}
	return cloned
}

func zeroSigning(keyring *securitykeyring.Signing) {
	if keyring == nil {
		return
	}
	zeroBytesMap(keyring.Private)
	zeroBytesMap(keyring.Public)
	*keyring = securitykeyring.Signing{}
}

func cloneBytesMap[T ~[]byte](source map[string]T) map[string][]byte {
	if source == nil {
		return nil
	}
	cloned := make(map[string][]byte, len(source))
	for version, key := range source {
		cloned[version] = append([]byte(nil), key...)
	}
	return cloned
}

func zeroBytesMap[T ~[]byte](values map[string]T) {
	for _, value := range values {
		zero(value)
	}
}

func zero(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
