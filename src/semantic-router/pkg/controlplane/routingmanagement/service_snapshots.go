package routingmanagement

import (
	"context"
	"fmt"
)

func (service *Service) ListSnapshots(
	ctx context.Context,
	namespaceID string,
	request SnapshotPageRequest,
) (Page[SnapshotMetadata], error) {
	if service == nil || !canonicalUUIDText(namespaceID) {
		return Page[SnapshotMetadata]{}, ErrInvalid
	}
	limit := request.PageSize
	if limit == 0 {
		limit = defaultRoutingPageSize
	}
	if limit < 1 || limit > maximumRoutingPageSize {
		return Page[SnapshotMetadata]{}, ErrInvalid
	}
	query := SnapshotListQuery{Limit: limit}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.NamespaceID != namespaceID ||
			cursor.ResourceKind != routingResourceSnapshot || cursor.RoutingRevision <= 0 ||
			cursor.Status != "" || cursor.Search != "" || cursor.ScopeDigest != "" ||
			!cursor.CreatedAt.IsZero() || cursor.ID != "" {
			return Page[SnapshotMetadata]{}, ErrInvalid
		}
		query.BeforeRevision = &cursor.RoutingRevision
	}
	result, err := service.store.ListSnapshots(ctx, namespaceID, query)
	if err != nil {
		return Page[SnapshotMetadata]{}, err
	}
	page := Page[SnapshotMetadata]{Items: result.Items, HasMore: result.HasMore}
	if !result.HasMore {
		return page, nil
	}
	if len(result.Items) == 0 {
		return Page[SnapshotMetadata]{}, fmt.Errorf("routing snapshot repository returned an empty continuation")
	}
	last := result.Items[len(result.Items)-1]
	page.NextCursor, err = service.cursors.encode(routingCursorPayload{
		NamespaceID: namespaceID, ResourceKind: routingResourceSnapshot,
		RoutingRevision: last.RoutingRevision,
	})
	if err != nil {
		return Page[SnapshotMetadata]{}, fmt.Errorf("encode routing snapshot continuation: %w", err)
	}
	return page, nil
}

func (service *Service) GetSnapshot(
	ctx context.Context,
	namespaceID string,
	routingRevision int64,
) (SnapshotDetail, error) {
	if service == nil || !canonicalUUIDText(namespaceID) || routingRevision <= 0 {
		return SnapshotDetail{}, ErrInvalid
	}
	return service.store.GetSnapshot(ctx, namespaceID, routingRevision)
}
