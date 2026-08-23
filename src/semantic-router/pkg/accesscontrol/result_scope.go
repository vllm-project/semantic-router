package accesscontrol

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"sort"
)

var ErrInvalidResultScope = errors.New("invalid authorized result scope")

// ResultScope is the canonical, server-authorized visibility envelope passed
// from authorization into bounded repository queries. It is a value contract,
// not an authorization evaluator; only the Management authorization runtime
// may derive one from grants.
type ResultScope struct {
	NamespaceID NamespaceID
	All         bool
	TeamIDs     []TeamID
	UserIDs     []UserID
	APIKeyIDs   []APIKeyID
	// ResourceIDs carries exact grants for non-subject resources. API keys stay
	// in APIKeyIDs because they also participate in ownership joins.
	ResourceIDs map[ScopeResourceType][]ResourceID
}

func (scope ResultScope) Empty() bool {
	return !scope.All && len(scope.TeamIDs) == 0 && len(scope.UserIDs) == 0 &&
		len(scope.APIKeyIDs) == 0 && len(scope.ResourceIDs) == 0
}

// IDs returns a defensive copy of the exact IDs authorized for resourceType.
func (scope ResultScope) IDs(resourceType ScopeResourceType) []ResourceID {
	return append([]ResourceID(nil), scope.ResourceIDs[resourceType]...)
}

// Canonical validates, sorts, and deduplicates every typed dimension.
func (scope ResultScope) Canonical() (ResultScope, error) {
	if scope.NamespaceID == "" || hasEmptyResultIDs(scope.TeamIDs) ||
		hasEmptyResultIDs(scope.UserIDs) || hasEmptyResultIDs(scope.APIKeyIDs) {
		return ResultScope{}, ErrInvalidResultScope
	}
	for resourceType, ids := range scope.ResourceIDs {
		if !resourceType.Valid() || resourceType == ScopeResourceAPIKey || hasEmptyResultIDs(ids) {
			return ResultScope{}, ErrInvalidResultScope
		}
	}
	if scope.All {
		return ResultScope{NamespaceID: scope.NamespaceID, All: true}, nil
	}
	canonical := ResultScope{
		NamespaceID: scope.NamespaceID,
		TeamIDs:     uniqueSortedResultIDs(scope.TeamIDs),
		UserIDs:     uniqueSortedResultIDs(scope.UserIDs),
		APIKeyIDs:   uniqueSortedResultIDs(scope.APIKeyIDs),
	}
	if len(scope.ResourceIDs) > 0 {
		canonical.ResourceIDs = make(map[ScopeResourceType][]ResourceID, len(scope.ResourceIDs))
		for resourceType, ids := range scope.ResourceIDs {
			normalized := uniqueSortedResultIDs(ids)
			if len(normalized) > 0 {
				canonical.ResourceIDs[resourceType] = normalized
			}
		}
		if len(canonical.ResourceIDs) == 0 {
			canonical.ResourceIDs = nil
		}
	}
	return canonical, nil
}

// Digest binds a cursor to the complete server-authorized visibility envelope.
// Equivalent sets produce the same digest regardless of input ordering.
func (scope ResultScope) Digest() (string, error) {
	canonical, err := scope.Canonical()
	if err != nil {
		return "", err
	}
	type resourceSet struct {
		Type ScopeResourceType `json:"type"`
		IDs  []ResourceID      `json:"ids"`
	}
	resources := make([]resourceSet, 0, len(canonical.ResourceIDs))
	for resourceType, ids := range canonical.ResourceIDs {
		resources = append(resources, resourceSet{Type: resourceType, IDs: ids})
	}
	sort.Slice(resources, func(left, right int) bool { return resources[left].Type < resources[right].Type })
	payload, err := json.Marshal(struct {
		Version     int           `json:"version"`
		NamespaceID NamespaceID   `json:"namespaceId"`
		All         bool          `json:"all"`
		TeamIDs     []TeamID      `json:"teamIds,omitempty"`
		UserIDs     []UserID      `json:"userIds,omitempty"`
		APIKeyIDs   []APIKeyID    `json:"apiKeyIds,omitempty"`
		Resources   []resourceSet `json:"resources,omitempty"`
	}{
		Version:     1,
		NamespaceID: canonical.NamespaceID,
		All:         canonical.All,
		TeamIDs:     canonical.TeamIDs,
		UserIDs:     canonical.UserIDs,
		APIKeyIDs:   canonical.APIKeyIDs,
		Resources:   resources,
	})
	if err != nil {
		return "", ErrInvalidResultScope
	}
	digest := sha256.Sum256(payload)
	return base64.RawURLEncoding.EncodeToString(digest[:]), nil
}

// Covers reports whether every row visible through other is also visible
// through scope.
func (scope ResultScope) Covers(other ResultScope) bool {
	canonical, err := scope.Canonical()
	if err != nil {
		return false
	}
	candidate, err := other.Canonical()
	if err != nil || canonical.NamespaceID != candidate.NamespaceID {
		return false
	}
	if canonical.All {
		return true
	}
	if candidate.All {
		return false
	}
	return containsAllResultIDs(canonical.TeamIDs, candidate.TeamIDs) &&
		containsAllResultIDs(canonical.UserIDs, candidate.UserIDs) &&
		containsAllResultIDs(canonical.APIKeyIDs, candidate.APIKeyIDs) &&
		containsAllResultResources(canonical.ResourceIDs, candidate.ResourceIDs)
}

func hasEmptyResultIDs[T ~string](values []T) bool {
	for _, value := range values {
		if value == "" {
			return true
		}
	}
	return false
}

func uniqueSortedResultIDs[T ~string](values []T) []T {
	if len(values) == 0 {
		return nil
	}
	result := append([]T(nil), values...)
	sort.Slice(result, func(left, right int) bool { return result[left] < result[right] })
	write := 0
	for _, value := range result {
		if write > 0 && result[write-1] == value {
			continue
		}
		result[write] = value
		write++
	}
	return result[:write]
}

func containsAllResultIDs[T ~string](values, candidates []T) bool {
	available := make(map[T]struct{}, len(values))
	for _, value := range values {
		available[value] = struct{}{}
	}
	for _, candidate := range candidates {
		if _, found := available[candidate]; !found {
			return false
		}
	}
	return true
}

func containsAllResultResources(
	values map[ScopeResourceType][]ResourceID,
	candidates map[ScopeResourceType][]ResourceID,
) bool {
	for resourceType, ids := range candidates {
		if !containsAllResultIDs(values[resourceType], ids) {
			return false
		}
	}
	return true
}
