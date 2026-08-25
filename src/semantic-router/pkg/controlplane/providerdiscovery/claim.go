package providerdiscovery

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

const (
	claimVersion          = 1
	minimumClaimKeyBytes  = 32
	maximumClaimKeys      = 8
	defaultClaimTTL       = 5 * time.Minute
	maximumClaimTTL       = 15 * time.Minute
	maximumDiscoveryItems = 200
)

type ClaimKeyset struct {
	ActiveKeyID string
	Keys        map[string][]byte
}

type ClaimCodec struct {
	activeKeyID string
	keys        map[string][]byte
}

type ClaimExpectation struct {
	NamespaceID     string
	AuthorityDigest string
	CatalogRevision string
	ProviderID      string
}

type SelectedModel struct {
	CatalogItemID   string
	ProviderModelID string
}

// VerifiedBinding is the complete immutable control-plane binding covered by
// a discovery claim. Bulk import must compare every field with its canonical
// Model authoring request before compiling a backend.
type VerifiedBinding struct {
	NamespaceID       string
	AuthorityDigest   string
	CatalogRevision   string
	ProviderID        string
	Origin            string
	ConnectionDigest  string
	CredentialID      string
	CredentialVersion string
}

type VerifiedSelection struct {
	Binding VerifiedBinding
	Models  []SelectedModel
}

type claimContext struct {
	NamespaceID       string `json:"namespaceId"`
	AuthorityDigest   string `json:"authorityDigest"`
	CatalogRevision   string `json:"catalogRevision"`
	ProviderID        string `json:"providerId"`
	Origin            string `json:"origin"`
	ConnectionDigest  string `json:"connectionDigest"`
	CredentialID      string `json:"credentialId,omitempty"`
	CredentialVersion string `json:"credentialVersion,omitempty"`
}

type claimItem struct {
	CatalogItemID   string `json:"catalogItemId"`
	ProviderModelID string `json:"providerModelId"`
}

type discoveryClaim struct {
	Version   int          `json:"version"`
	Context   claimContext `json:"context"`
	Items     []claimItem  `json:"items"`
	ExpiresAt int64        `json:"expiresAt"`
}

func NewClaimCodec(keyset ClaimKeyset) (ClaimCodec, error) {
	if !adapterIDPattern.MatchString(keyset.ActiveKeyID) || len(keyset.Keys) == 0 || len(keyset.Keys) > maximumClaimKeys {
		return ClaimCodec{}, fmt.Errorf("%w: discovery claim keyset is invalid", ErrInvalidClaim)
	}
	keys := make(map[string][]byte, len(keyset.Keys))
	for keyID, key := range keyset.Keys {
		if !adapterIDPattern.MatchString(keyID) || len(key) < minimumClaimKeyBytes {
			return ClaimCodec{}, fmt.Errorf("%w: discovery claim key %q is invalid", ErrInvalidClaim, keyID)
		}
		keys[keyID] = append([]byte(nil), key...)
	}
	if _, found := keys[keyset.ActiveKeyID]; !found {
		return ClaimCodec{}, fmt.Errorf("%w: active discovery claim key is unavailable", ErrInvalidClaim)
	}
	return ClaimCodec{activeKeyID: keyset.ActiveKeyID, keys: keys}, nil
}

func (codec ClaimCodec) Issue(
	plan providercatalog.DiscoveryPlan,
	authorityDigest string,
	credentialVersion string,
	models []AdapterModel,
	now time.Time,
	ttl time.Duration,
) ([]Model, string, time.Time, error) {
	if ttl == 0 {
		ttl = defaultClaimTTL
	}
	if now.IsZero() || ttl <= 0 || ttl > maximumClaimTTL || len(models) > maximumDiscoveryItems {
		return nil, "", time.Time{}, fmt.Errorf("%w: discovery claim lifetime or item count is invalid", ErrInvalidClaim)
	}
	context, err := newClaimContext(plan, authorityDigest, credentialVersion)
	if err != nil {
		return nil, "", time.Time{}, err
	}
	key, found := codec.keys[codec.activeKeyID]
	if !found {
		return nil, "", time.Time{}, fmt.Errorf("%w: active discovery claim key is unavailable", ErrInvalidClaim)
	}
	binding, err := json.Marshal(context)
	if err != nil {
		return nil, "", time.Time{}, fmt.Errorf("%w: encode discovery binding", ErrInvalidClaim)
	}
	seenModelIDs := make(map[string]struct{}, len(models))
	seenItemIDs := make(map[string]struct{}, len(models))
	result := make([]Model, len(models))
	items := make([]claimItem, len(models))
	for index, model := range models {
		if !canonicalModelText(model.ProviderModelID, 1, 512) || !canonicalModelText(model.DisplayName, 1, 512) {
			return nil, "", time.Time{}, fmt.Errorf("%w: discovered model %d is invalid", ErrInvalidClaim, index)
		}
		if _, duplicate := seenModelIDs[model.ProviderModelID]; duplicate {
			return nil, "", time.Time{}, fmt.Errorf("%w: duplicate discovered model %q", ErrInvalidClaim, model.ProviderModelID)
		}
		seenModelIDs[model.ProviderModelID] = struct{}{}
		itemID := claimItemID(key, binding, model.ProviderModelID)
		if _, duplicate := seenItemIDs[itemID]; duplicate {
			return nil, "", time.Time{}, fmt.Errorf("%w: discovery item identity collision", ErrInvalidClaim)
		}
		seenItemIDs[itemID] = struct{}{}
		result[index] = Model{
			CatalogItemID: itemID, ProviderModelID: model.ProviderModelID,
			DisplayName:  model.DisplayName,
			Capabilities: append([]string(nil), model.Capabilities...),
		}
		items[index] = claimItem{CatalogItemID: itemID, ProviderModelID: model.ProviderModelID}
	}
	sort.Slice(items, func(left, right int) bool { return items[left].CatalogItemID < items[right].CatalogItemID })
	expiresAt := now.UTC().Add(ttl)
	token, err := codec.sign(discoveryClaim{
		Version: claimVersion, Context: context, Items: items, ExpiresAt: expiresAt.Unix(),
	})
	if err != nil {
		return nil, "", time.Time{}, err
	}
	return result, token, expiresAt, nil
}

func (codec ClaimCodec) VerifySelection(
	token string,
	expectation ClaimExpectation,
	selectedItemIDs []string,
	now time.Time,
) (VerifiedSelection, error) {
	claim, err := codec.verify(token)
	if err != nil {
		return VerifiedSelection{}, err
	}
	if now.IsZero() || !now.Before(time.Unix(claim.ExpiresAt, 0)) {
		return VerifiedSelection{}, ErrExpiredClaim
	}
	if claim.Context.NamespaceID != expectation.NamespaceID ||
		claim.Context.AuthorityDigest != expectation.AuthorityDigest ||
		claim.Context.CatalogRevision != expectation.CatalogRevision ||
		claim.Context.ProviderID != expectation.ProviderID {
		return VerifiedSelection{}, ErrInvalidClaim
	}
	if len(selectedItemIDs) == 0 || len(selectedItemIDs) > len(claim.Items) {
		return VerifiedSelection{}, ErrInvalidClaim
	}
	available := make(map[string]string, len(claim.Items))
	for _, item := range claim.Items {
		if !canonicalItemID(item.CatalogItemID) || !canonicalModelText(item.ProviderModelID, 1, 512) {
			return VerifiedSelection{}, ErrInvalidClaim
		}
		if _, duplicate := available[item.CatalogItemID]; duplicate {
			return VerifiedSelection{}, ErrInvalidClaim
		}
		available[item.CatalogItemID] = item.ProviderModelID
	}
	selected := make([]SelectedModel, len(selectedItemIDs))
	seen := make(map[string]struct{}, len(selectedItemIDs))
	for index, itemID := range selectedItemIDs {
		modelID, found := available[itemID]
		if !found {
			return VerifiedSelection{}, ErrInvalidClaim
		}
		if _, duplicate := seen[itemID]; duplicate {
			return VerifiedSelection{}, ErrInvalidClaim
		}
		seen[itemID] = struct{}{}
		selected[index] = SelectedModel{CatalogItemID: itemID, ProviderModelID: modelID}
	}
	return VerifiedSelection{
		Binding: VerifiedBinding{
			NamespaceID: claim.Context.NamespaceID, AuthorityDigest: claim.Context.AuthorityDigest,
			CatalogRevision: claim.Context.CatalogRevision, ProviderID: claim.Context.ProviderID,
			Origin: claim.Context.Origin, ConnectionDigest: claim.Context.ConnectionDigest,
			CredentialID: claim.Context.CredentialID, CredentialVersion: claim.Context.CredentialVersion,
		},
		Models: selected,
	}, nil
}

func newClaimContext(
	plan providercatalog.DiscoveryPlan,
	authorityDigest string,
	credentialVersion string,
) (claimContext, error) {
	if _, err := uuid.Parse(plan.NamespaceID); err != nil || !digestPattern(plan.CatalogRevision) ||
		!digestPattern(authorityDigest) || !adapterIDPattern.MatchString(plan.ProviderID) {
		return claimContext{}, ErrInvalidClaim
	}
	if plan.CredentialID == "" && credentialVersion != "" || plan.CredentialID != "" && credentialVersion == "" {
		return claimContext{}, ErrInvalidClaim
	}
	if plan.CredentialID != "" {
		if _, err := uuid.Parse(plan.CredentialID); err != nil || !canonicalModelText(credentialVersion, 1, 256) {
			return claimContext{}, ErrInvalidClaim
		}
	}
	connectionDigest, err := providercatalog.CanonicalConnectionDigest(plan.ConnectionFields)
	if err != nil {
		return claimContext{}, err
	}
	return claimContext{
		NamespaceID: plan.NamespaceID, AuthorityDigest: authorityDigest,
		CatalogRevision: plan.CatalogRevision, ProviderID: plan.ProviderID,
		Origin: plan.NormalizedOrigin, ConnectionDigest: connectionDigest,
		CredentialID: plan.CredentialID, CredentialVersion: credentialVersion,
	}, nil
}

func claimItemID(key, binding []byte, providerModelID string) string {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write(binding)
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(providerModelID))
	return "pmi_" + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func (codec ClaimCodec) sign(claim discoveryClaim) (string, error) {
	payload, err := json.Marshal(claim)
	if err != nil {
		return "", fmt.Errorf("%w: encode discovery claim", ErrInvalidClaim)
	}
	encoded := base64.RawURLEncoding.EncodeToString(payload)
	message := codec.activeKeyID + "." + encoded
	mac := hmac.New(sha256.New, codec.keys[codec.activeKeyID])
	_, _ = mac.Write([]byte(message))
	return message + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func (codec ClaimCodec) verify(token string) (discoveryClaim, error) {
	if len(token) == 0 || len(token) > 256*1024 || strings.TrimSpace(token) != token {
		return discoveryClaim{}, ErrInvalidClaim
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 || !adapterIDPattern.MatchString(parts[0]) {
		return discoveryClaim{}, ErrInvalidClaim
	}
	key, found := codec.keys[parts[0]]
	if !found {
		return discoveryClaim{}, ErrInvalidClaim
	}
	providedMAC, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || len(providedMAC) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(providedMAC) != parts[2] {
		return discoveryClaim{}, ErrInvalidClaim
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(parts[0] + "." + parts[1]))
	if !hmac.Equal(providedMAC, mac.Sum(nil)) {
		return discoveryClaim{}, ErrInvalidClaim
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[1] {
		return discoveryClaim{}, ErrInvalidClaim
	}
	var claim discoveryClaim
	if err := json.Unmarshal(payload, &claim); err != nil || claim.Version != claimVersion ||
		claim.ExpiresAt <= 0 || len(claim.Items) > maximumDiscoveryItems {
		return discoveryClaim{}, ErrInvalidClaim
	}
	return claim, nil
}

func digestPattern(value string) bool {
	if len(value) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil && value == strings.ToLower(value)
}

func canonicalItemID(value string) bool {
	if !strings.HasPrefix(value, "pmi_") {
		return false
	}
	encoded := strings.TrimPrefix(value, "pmi_")
	decoded, err := base64.RawURLEncoding.DecodeString(encoded)
	return err == nil && len(decoded) == sha256.Size && base64.RawURLEncoding.EncodeToString(decoded) == encoded
}
