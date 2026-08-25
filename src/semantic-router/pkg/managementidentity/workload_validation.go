package managementidentity

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func validateServiceAccountOwner(owner ServiceAccountOwnerScope, namespaceID string) error {
	switch owner {
	case ServiceAccountOwnerCluster:
		if namespaceID != "" {
			return ErrInvalidWorkloadRequest
		}
	case ServiceAccountOwnerNamespace:
		if !canonicalUUID(namespaceID) {
			return ErrInvalidWorkloadRequest
		}
	default:
		return ErrInvalidWorkloadRequest
	}
	return nil
}

func validateCredentialExpiry(now, expiresAt time.Time) error {
	expiresAt = expiresAt.UTC()
	if now.IsZero() || expiresAt.IsZero() || expiresAt.Before(now.Add(workloadCredentialMinimumTTL)) || expiresAt.After(now.Add(workloadCredentialMaximumTTL)) {
		return ErrInvalidWorkloadRequest
	}
	return nil
}

func validateWorkloadActor(actor WorkloadActor) error {
	if validateActor(actor.MutationActor()) != nil || actor.Session.PrincipalID != actor.PrincipalID {
		return ErrInvalidWorkloadRequest
	}
	return nil
}

func validWorkloadClass(value WorkloadClass) bool {
	return value == WorkloadStandard || value == WorkloadStrong
}

func canonicalWorkloadPageSize(value int) int {
	if value == 0 {
		return workloadIdentityPageSize
	}
	if value < 1 || value > workloadIdentityMaximumPage {
		return 0
	}
	return value
}

func validateServiceAccountList(request ServiceAccountListRequest) (int, string, error) {
	pageSize := canonicalWorkloadPageSize(request.PageSize)
	if pageSize == 0 || (request.Status != "" && request.Status != ServiceAccountActive && request.Status != ServiceAccountDisabled) {
		return 0, "", ErrInvalidWorkloadRequest
	}
	scope := canonicalServiceAccountScope(request.Scope)
	if scope.Cluster {
		if scope.NamespaceID != "" || !scope.All || len(scope.IDs) != 0 {
			return 0, "", ErrInvalidWorkloadRequest
		}
	} else if !canonicalUUID(scope.NamespaceID) || (scope.All && len(scope.IDs) != 0) {
		return 0, "", ErrInvalidWorkloadRequest
	}
	for _, id := range scope.IDs {
		if !canonicalUUID(id) {
			return 0, "", ErrInvalidWorkloadRequest
		}
	}
	payload, _ := json.Marshal(scope)
	digest := sha256.Sum256(payload)
	return pageSize, base64.RawURLEncoding.EncodeToString(digest[:]), nil
}

func canonicalServiceAccountScope(scope ServiceAccountResultScope) ServiceAccountResultScope {
	result := scope
	result.IDs = append([]string(nil), scope.IDs...)
	sort.Strings(result.IDs)
	write := 0
	for _, id := range result.IDs {
		if write > 0 && result.IDs[write-1] == id {
			continue
		}
		result.IDs[write] = id
		write++
	}
	result.IDs = result.IDs[:write]
	return result
}

func workloadDigest(values ...string) string {
	digest := sha256.New()
	_, _ = digest.Write([]byte("vllm-sr/workload-list-scope/v1\x00"))
	for _, value := range values {
		_, _ = digest.Write([]byte(value))
		_, _ = digest.Write([]byte{0})
	}
	return base64.RawURLEncoding.EncodeToString(digest.Sum(nil))
}

var (
	dnsMatcherPattern    = regexp.MustCompile(`^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))*$`)
	subjectDigestPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func validateMTLSMatcher(kind MTLSMatcherKind, value string) error {
	if value == "" || len(value) > 2048 || strings.TrimSpace(value) != value {
		return ErrInvalidWorkloadRequest
	}
	switch kind {
	case MTLSMatcherSPIFFEID, MTLSMatcherSANURI:
		parsed, err := url.Parse(value)
		if err != nil || !parsed.IsAbs() || parsed.String() != value || parsed.User != nil || parsed.Fragment != "" {
			return ErrInvalidWorkloadRequest
		}
		if kind == MTLSMatcherSPIFFEID && (parsed.Scheme != "spiffe" || parsed.Host == "" || parsed.RawQuery != "") {
			return ErrInvalidWorkloadRequest
		}
	case MTLSMatcherSANDNS:
		if value != strings.ToLower(value) || strings.Contains(value, "*") || !dnsMatcherPattern.MatchString(value) {
			return ErrInvalidWorkloadRequest
		}
	case MTLSMatcherSubjectDNDigest:
		if !subjectDigestPattern.MatchString(value) {
			return ErrInvalidWorkloadRequest
		}
	default:
		return ErrInvalidWorkloadRequest
	}
	return nil
}

func validateWorkloadPeppers(peppers securitykeyring.Symmetric) error {
	if !canonicalWorkloadKeyVersion(peppers.ActiveVersion) || len(peppers.Keys) < 1 || len(peppers.Keys) > 8 {
		return ErrWorkloadUnavailable
	}
	for version, key := range peppers.Keys {
		if !canonicalWorkloadKeyVersion(version) || len(key) != sha256.Size {
			return ErrWorkloadUnavailable
		}
	}
	if _, found := peppers.Keys[peppers.ActiveVersion]; !found {
		return ErrWorkloadUnavailable
	}
	return nil
}

func ComputeServiceCredentialHMAC(pepper []byte, credentialID string, secret []byte) [sha256.Size]byte {
	digest := hmac.New(sha256.New, pepper)
	_, _ = digest.Write([]byte("vsm"))
	_, _ = digest.Write([]byte{0})
	_, _ = digest.Write([]byte(credentialID))
	_, _ = digest.Write([]byte{0})
	_, _ = digest.Write(secret)
	var result [sha256.Size]byte
	copy(result[:], digest.Sum(nil))
	return result
}

func workloadResponseAAD(endpoint, accountID string, revision uint64) []byte {
	return []byte(fmt.Sprintf("vllm-sr/workload-secret/v1\x00%s\x00%s\x00%d", endpoint, accountID, revision))
}

func cloneWorkloadSymmetric(source securitykeyring.Symmetric) securitykeyring.Symmetric {
	result := securitykeyring.Symmetric{ActiveVersion: source.ActiveVersion, Keys: make(map[string][]byte, len(source.Keys))}
	for version, key := range source.Keys {
		result.Keys[version] = append([]byte(nil), key...)
	}
	return result
}

func cloneWorkloadKEK(source accesscredential.KEKKeyring) accesscredential.KEKKeyring {
	result := accesscredential.KEKKeyring{ActiveVersion: source.ActiveVersion, Keys: make(map[string][]byte, len(source.Keys))}
	for version, key := range source.Keys {
		result.Keys[version] = append([]byte(nil), key...)
	}
	return result
}

func zeroWorkloadBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

func zeroWorkloadString(value *string) {
	if value != nil {
		*value = ""
	}
}

var _ managementauth.MTLSIdentityResolver = (*WorkloadIdentityService)(nil)
