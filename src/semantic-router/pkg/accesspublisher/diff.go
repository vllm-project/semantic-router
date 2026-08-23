package accesspublisher

import (
	"encoding/json"
	"fmt"
	"math/big"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

type PreviousDocuments struct {
	Manifest            *Manifest
	Access              map[string]AccessDocument
	Credentials         map[string]CredentialDocument
	ProviderCredentials map[string]ProviderCredentialDocument
	Routing             *RoutingDocument
}

func Diff(previous PreviousDocuments, current Publication) ([]Barrier, error) {
	barriers := append([]Barrier(nil), current.BarrierHints...)
	if previous.Manifest == nil {
		return canonicalBarriers(barriers)
	}
	if previous.Manifest.NamespaceID != current.NamespaceID || previous.Manifest.QuotaPartition != current.QuotaPartition {
		return nil, fmt.Errorf("previous manifest belongs to another namespace or partition")
	}
	if previous.Manifest.RuntimeEpoch != current.RuntimeEpoch {
		barriers = append(barriers, Barrier{Kind: "namespace", ResourceID: current.NamespaceID, Reason: "runtime_epoch_changed"})
	}

	currentAccess := make(map[string]AccessDocument, len(current.Access))
	for _, document := range current.Access {
		currentAccess[document.KeyID] = document
	}
	for keyID := range previous.Manifest.Access {
		oldDocument, ok := previous.Access[keyID]
		if !ok {
			return nil, fmt.Errorf("previous access document %s is unavailable", keyID)
		}
		newDocument, exists := currentAccess[keyID]
		if !exists || accessRestricts(oldDocument, newDocument) {
			barriers = append(barriers, Barrier{Kind: "api_key", ResourceID: keyID, Reason: "policy_restricted"})
		}
	}

	currentCredentials := make(map[string]CredentialDocument, len(current.Credentials))
	for _, document := range current.Credentials {
		currentCredentials[credentialIdentity(document.Kind, document.PublicID)] = document
	}
	for identity := range previous.Manifest.Credentials {
		oldDocument, ok := previous.Credentials[identity]
		if !ok {
			return nil, fmt.Errorf("previous credential document %s is unavailable", identity)
		}
		newDocument, exists := currentCredentials[identity]
		if !exists || credentialRestricts(oldDocument, newDocument) {
			kind, publicID, _ := strings.Cut(identity, ":")
			barriers = append(barriers, Barrier{
				Kind: "credential", ResourceID: kind + ":" + publicID, Reason: "credential_restricted",
			})
		}
	}

	currentProviderCredentials := make(map[string]ProviderCredentialDocument, len(current.ProviderCredentials))
	for _, document := range current.ProviderCredentials {
		currentProviderCredentials[document.Credential.ID] = document
	}
	for credentialID := range previous.Manifest.ProviderCredentials {
		oldDocument, ok := previous.ProviderCredentials[credentialID]
		if !ok {
			return nil, fmt.Errorf("previous provider credential document %s is unavailable", credentialID)
		}
		newDocument, exists := currentProviderCredentials[credentialID]
		if exists && providerCredentialRestricts(oldDocument, newDocument) {
			barriers = append(barriers, Barrier{
				Kind: "provider_credential", ResourceID: credentialID, Reason: "provider_credential_restricted",
			})
		}
	}

	for resource, oldDigest := range previous.Manifest.RoutingResources {
		if newDigest, exists := current.Routing.ResourceDigests[resource]; !exists || newDigest != oldDigest {
			kind, resourceID, ok := strings.Cut(resource, ":")
			if !ok {
				return nil, fmt.Errorf("invalid routing resource identity %q", resource)
			}
			barriers = append(barriers, Barrier{Kind: kind, ResourceID: resourceID, Reason: "routing_restricted"})
		}
	}
	return canonicalBarriers(barriers)
}

func providerCredentialRestricts(oldDocument, newDocument ProviderCredentialDocument) bool {
	old, next := oldDocument.Credential, newDocument.Credential
	return next.Status != "active" || old.NamespaceID != next.NamespaceID ||
		old.ProviderID != next.ProviderID || old.CredentialMode != next.CredentialMode ||
		old.CredentialAdapterID != next.CredentialAdapterID || old.CatalogRevision != next.CatalogRevision ||
		old.NormalizedOrigin != next.NormalizedOrigin
}

func credentialRestricts(oldDocument, newDocument CredentialDocument) bool {
	old, next := oldDocument.Projection, newDocument.Projection
	if old.Kind != next.Kind || old.KeyID != next.KeyID || old.KID != next.KID ||
		old.PepperVersion != next.PepperVersion || old.Status != next.Status ||
		old.ManagementSessionID != next.ManagementSessionID || old.PrincipalID != next.PrincipalID ||
		old.DelegationEpoch != next.DelegationEpoch || old.UserID != next.UserID ||
		old.TeamID != next.TeamID || old.Audience != next.Audience ||
		!equalBytes(old.SecretHMAC, next.SecretHMAC) {
		return true
	}
	if next.NotBefore.After(old.NotBefore) {
		return true
	}
	return expiryRestricts(timePointerMillis(old.ExpiresAt), timePointerMillis(next.ExpiresAt))
}

func accessRestricts(oldDocument, newDocument AccessDocument) bool {
	old, next := oldDocument.Projection, newDocument.Projection
	if old.NamespaceID != next.NamespaceID || old.QuotaPartition != next.QuotaPartition || old.KeyID != next.KeyID ||
		old.UserID != next.UserID || old.TeamID != next.TeamID || old.PolicyEpoch != next.PolicyEpoch ||
		old.DelegationEpoch != next.DelegationEpoch || old.AccessSource != next.AccessSource {
		return true
	}
	if next.KeyStatus != "active" || old.KeyStatus != next.KeyStatus ||
		expiryRestricts(timePointerMillis(old.KeyExpiresAt), timePointerMillis(next.KeyExpiresAt)) {
		return true
	}
	if routingClaimsDigest(old.RoutingClaims) != routingClaimsDigest(next.RoutingClaims) {
		return true
	}
	if accessAllowsRemoved(old.Grants, next.Grants) {
		return true
	}
	return quotaRestricts(old.RateBindings, next.RateBindings)
}

func accessAllowsRemoved(oldGrants, newGrants []accessprojection.Grant) bool {
	oldAllowed := allowedGrantSet(oldGrants)
	newAllowed := allowedGrantSet(newGrants)
	for grant := range oldAllowed {
		if _, exists := newAllowed[grant]; !exists {
			return true
		}
	}
	return false
}

func allowedGrantSet(grants []accessprojection.Grant) map[string]struct{} {
	type decision struct{ allow, deny bool }
	decisions := make(map[string]decision)
	for _, grant := range grants {
		key := string(grant.ResourceType) + "\x00" + grant.ResourceID + "\x00" + string(grant.Permission)
		value := decisions[key]
		switch grant.Effect {
		case "allow":
			value.allow = true
		case "deny":
			value.deny = true
		}
		decisions[key] = value
	}
	result := make(map[string]struct{})
	for key, value := range decisions {
		if value.allow && !value.deny {
			result[key] = struct{}{}
		}
	}
	return result
}

func quotaRestricts(oldBindings, newBindings []accessprojection.RateBinding) bool {
	oldRules := flattenedRules(oldBindings)
	newRules := flattenedRules(newBindings)
	for identity, next := range newRules {
		old, existed := oldRules[identity]
		if !existed {
			if next.rule.Enforcement == quota.EnforcementEnforce {
				return true
			}
			continue
		}
		if ruleRestricts(old, next) {
			return true
		}
	}
	return false
}

type flattenedRule struct {
	mode accessprojection.RateBinding
	rule quota.RateLimitRule
}

func flattenedRules(bindings []accessprojection.RateBinding) map[string]flattenedRule {
	result := make(map[string]flattenedRule)
	for _, binding := range bindings {
		for _, projected := range binding.Rules {
			rule := projected.Rule
			result[binding.BindingID+"\x00"+rule.ID] = flattenedRule{mode: binding, rule: rule}
		}
	}
	return result
}

func ruleRestricts(old, next flattenedRule) bool {
	if old.mode.Mode != next.mode.Mode || old.rule.Metric != next.rule.Metric ||
		old.rule.Algorithm != next.rule.Algorithm || old.rule.Accounting != next.rule.Accounting ||
		old.rule.Window != next.rule.Window || old.rule.CalendarPeriod != next.rule.CalendarPeriod ||
		old.rule.CalendarTimezone != next.rule.CalendarTimezone || old.rule.RefillPeriod != next.rule.RefillPeriod ||
		old.rule.GCRAEmissionInterval != next.rule.GCRAEmissionInterval {
		return true
	}
	if old.rule.Enforcement == quota.EnforcementShadow && next.rule.Enforcement == quota.EnforcementEnforce {
		return true
	}
	if next.rule.Enforcement == quota.EnforcementShadow {
		return false
	}
	for _, values := range [][2]*quota.QuotaInteger{
		{old.rule.WholeLimit, next.rule.WholeLimit},
		{old.rule.BucketCapacity, next.rule.BucketCapacity},
		{old.rule.RefillAmount, next.rule.RefillAmount},
		{old.rule.GCRABurstTolerance, next.rule.GCRABurstTolerance},
	} {
		if quantityRestricts(values[0], values[1]) {
			return true
		}
	}
	return currencyRestricts(old.rule.CostLimit, next.rule.CostLimit)
}

func quantityRestricts(old, next *quota.QuotaInteger) bool {
	if old == nil {
		return next != nil
	}
	if next == nil {
		return false
	}
	oldInteger, _ := new(big.Int).SetString(old.String(), 10)
	nextInteger, _ := new(big.Int).SetString(next.String(), 10)
	return nextInteger.Cmp(oldInteger) < 0
}

func currencyRestricts(old, next *quota.CurrencyDecimal) bool {
	if old == nil {
		return next != nil
	}
	if next == nil {
		return false
	}
	oldValue, _ := new(big.Rat).SetString(old.String())
	nextValue, _ := new(big.Rat).SetString(next.String())
	return nextValue.Cmp(oldValue) < 0
}

func expiryRestricts(old, next *int64) bool {
	if old == nil {
		return next != nil
	}
	if next == nil {
		return false
	}
	return *next < *old
}

func timePointerMillis(value *time.Time) *int64 {
	if value == nil {
		return nil
	}
	result := value.UnixMilli()
	return &result
}

func equalBytes(left, right []byte) bool {
	if len(left) != len(right) {
		return false
	}
	var difference byte
	for index := range left {
		difference |= left[index] ^ right[index]
	}
	return difference == 0
}

func routingClaimsDigest(value any) string {
	payload, _ := json.Marshal(value)
	return string(payload)
}

func sortedMapKeys[V any](input map[string]V) []string {
	keys := make([]string, 0, len(input))
	for key := range input {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
