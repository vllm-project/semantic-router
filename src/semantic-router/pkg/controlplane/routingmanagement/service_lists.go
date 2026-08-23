package routingmanagement

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

const (
	defaultRoutingPageSize = 50
	maximumRoutingPageSize = 200
)

type routingListBinding struct {
	namespaceID string
	kind        routingResourceKind
	status      Status
	search      string
	scopeDigest string
}

func (service *Service) ListModels(
	ctx context.Context, namespaceID string, request PageRequest,
) (Page[Model], error) {
	query, binding, err := service.prepareList(namespaceID, routingResourceModel, request)
	if err != nil {
		return Page[Model]{}, err
	}
	result, err := service.store.ListModels(ctx, namespaceID, query)
	if err != nil {
		return Page[Model]{}, err
	}
	return finishRoutingList(service, result, binding, func(model Model) ResourceIdentity {
		return model.ResourceIdentity
	})
}

func (service *Service) ListRecipes(
	ctx context.Context, namespaceID string, request PageRequest,
) (Page[Recipe], error) {
	query, binding, err := service.prepareList(namespaceID, routingResourceRecipe, request)
	if err != nil {
		return Page[Recipe]{}, err
	}
	result, err := service.store.ListRecipes(ctx, namespaceID, query)
	if err != nil {
		return Page[Recipe]{}, err
	}
	return finishRoutingList(service, result, binding, func(recipe Recipe) ResourceIdentity {
		return recipe.ResourceIdentity
	})
}

func (service *Service) ListEntrypoints(
	ctx context.Context, namespaceID string, request PageRequest,
) (Page[Entrypoint], error) {
	query, binding, err := service.prepareList(namespaceID, routingResourceEntrypoint, request)
	if err != nil {
		return Page[Entrypoint]{}, err
	}
	result, err := service.store.ListEntrypoints(ctx, namespaceID, query)
	if err != nil {
		return Page[Entrypoint]{}, err
	}
	return finishRoutingList(service, result, binding, func(entrypoint Entrypoint) ResourceIdentity {
		return entrypoint.ResourceIdentity
	})
}

func (service *Service) prepareList(
	namespaceID string, kind routingResourceKind, request PageRequest,
) (ListQuery, routingListBinding, error) {
	if service == nil || !canonicalUUIDText(namespaceID) || !validRoutingListStatus(request.Status) {
		return ListQuery{}, routingListBinding{}, ErrInvalid
	}
	limit := request.PageSize
	if limit == 0 {
		limit = defaultRoutingPageSize
	}
	if limit < 1 || limit > maximumRoutingPageSize {
		return ListQuery{}, routingListBinding{}, ErrInvalid
	}
	search, err := managementsearch.Normalize(request.Search)
	if err != nil {
		return ListQuery{}, routingListBinding{}, ErrInvalid
	}
	scope, err := request.Scope.Canonical()
	if err != nil || scope.NamespaceID != accesscontrol.NamespaceID(namespaceID) {
		return ListQuery{}, routingListBinding{}, ErrInvalid
	}
	scopeDigest, err := scope.Digest()
	if err != nil {
		return ListQuery{}, routingListBinding{}, ErrInvalid
	}
	binding := routingListBinding{
		namespaceID: namespaceID, kind: kind, status: request.Status,
		search: search, scopeDigest: scopeDigest,
	}
	query := ListQuery{Limit: limit, Search: search, Status: request.Status, Scope: scope}
	if request.Cursor == "" {
		return query, binding, nil
	}
	cursor, err := service.cursors.decode(request.Cursor)
	if err != nil || cursor.NamespaceID != binding.namespaceID || cursor.ResourceKind != binding.kind ||
		cursor.Status != binding.status || cursor.Search != binding.search ||
		cursor.ScopeDigest != binding.scopeDigest || cursor.CreatedAt.IsZero() ||
		ValidateResourceID(cursor.ID) != nil {
		return ListQuery{}, routingListBinding{}, ErrInvalid
	}
	query.After = &ListCursor{CreatedAt: cursor.CreatedAt.UTC(), ID: cursor.ID}
	return query, binding, nil
}

func finishRoutingList[T any](
	service *Service,
	result ListResult[T], binding routingListBinding, identity func(T) ResourceIdentity,
) (Page[T], error) {
	page := Page[T]{Items: result.Items, HasMore: result.HasMore}
	if !result.HasMore {
		return page, nil
	}
	if len(result.Items) == 0 {
		return Page[T]{}, fmt.Errorf("routing repository returned an empty continuation")
	}
	last := identity(result.Items[len(result.Items)-1])
	var err error
	page.NextCursor, err = service.cursors.encode(routingCursorPayload{
		NamespaceID: binding.namespaceID, ResourceKind: binding.kind,
		Status: binding.status, Search: binding.search, ScopeDigest: binding.scopeDigest,
		CreatedAt: last.CreatedAt.UTC(), ID: last.ID,
	})
	if err != nil {
		return Page[T]{}, fmt.Errorf("encode routing continuation: %w", err)
	}
	return page, nil
}

func validRoutingListStatus(status Status) bool {
	return status == "" || status == StatusDraft || status == StatusActive || status == StatusDisabled
}
