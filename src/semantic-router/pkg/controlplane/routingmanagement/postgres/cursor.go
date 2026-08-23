package postgres

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

type listScope struct {
	all    bool
	ids    []accesscontrol.ResourceID
	digest string
}

func normalizeListScope(
	namespaceID string,
	resourceType accesscontrol.ScopeResourceType,
	request routingmanagement.ListQuery,
) (listScope, error) {
	canonical, err := request.Scope.Canonical()
	if err != nil || canonical.NamespaceID != accesscontrol.NamespaceID(namespaceID) {
		return listScope{}, fmt.Errorf("%w: routing result scope is invalid", routingmanagement.ErrInvalid)
	}
	digest, err := canonical.Digest()
	if err != nil {
		return listScope{}, fmt.Errorf("%w: routing result scope is invalid", routingmanagement.ErrInvalid)
	}
	return listScope{all: canonical.All, ids: canonical.IDs(resourceType), digest: digest}, nil
}

func validateListQuery(request routingmanagement.ListQuery) error {
	normalizedSearch, err := managementsearch.Normalize(request.Search)
	if err != nil || normalizedSearch != request.Search || request.Limit < 1 || request.Limit > 200 {
		return fmt.Errorf("%w: routing list query is invalid", routingmanagement.ErrInvalid)
	}
	if request.Status != "" && request.Status != routingmanagement.StatusDraft &&
		request.Status != routingmanagement.StatusActive && request.Status != routingmanagement.StatusDisabled {
		return fmt.Errorf("%w: routing list status is invalid", routingmanagement.ErrInvalid)
	}
	if request.After != nil && (request.After.CreatedAt.IsZero() || routingmanagement.ValidateResourceID(request.After.ID) != nil) {
		return fmt.Errorf("%w: routing list seek is invalid", routingmanagement.ErrInvalid)
	}
	return nil
}
