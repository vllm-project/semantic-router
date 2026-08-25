package providercatalog

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	defaultPageSize = 50
	maximumPageSize = 200
)

type ServiceOptions struct {
	CursorKeyring    securitykeyring.Symmetric
	DiscoveryPlugins *DiscoveryRegistry
}

// SnapshotSource resolves the currently active immutable catalog revision.
// A PostgreSQL publication coordinator can implement this without coupling
// reads to provider integration loading mechanics.
type SnapshotSource interface {
	ActiveSnapshot(context.Context) (*Snapshot, error)
}

type SnapshotSourceFunc func(context.Context) (*Snapshot, error)

func (resolve SnapshotSourceFunc) ActiveSnapshot(ctx context.Context) (*Snapshot, error) {
	return resolve(ctx)
}

// Service is the HTTP-neutral application boundary for catalog reads and
// discovery preparation. It owns a defensive copy of one immutable snapshot;
// replacing a catalog creates another Service and another revision.
type Service struct {
	source    SnapshotSource
	cursors   cursorCodec
	discovery *DiscoveryRegistry
}

type ListRequest struct {
	PageSize int
	Cursor   string
	Search   string
	Category string
	// Capability filters Provider transport support, not Model metadata.
	Capability string
}

type ListResult struct {
	CatalogRevision string
	Providers       []Definition
	Categories      []string
	PageSize        int
	NextCursor      string
	HasMore         bool
}

type DetailResult struct {
	CatalogRevision string
	Provider        Definition
}

type normalizedListQuery struct {
	pageSize   int
	search     string
	category   string
	capability string
}

func NewService(source SnapshotSource, options ServiceOptions) (*Service, error) {
	if source == nil {
		return nil, fmt.Errorf("%w: active provider catalog source is required", ErrInvalidRequest)
	}
	codec, err := newCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, err
	}
	discovery := options.DiscoveryPlugins
	if discovery == nil {
		discovery = &DiscoveryRegistry{plugins: map[string]DiscoveryRequestValidator{}}
	}
	return &Service{
		source: source, cursors: codec, discovery: discovery.clone(),
	}, nil
}

// Close erases the Service-owned cursor key copies. The process lifecycle
// owner calls it only after Management HTTP handlers have stopped.
func (service *Service) Close() {
	if service == nil {
		return
	}
	service.cursors.close()
}

func (service *Service) Revision(ctx context.Context) (string, error) {
	if service == nil {
		return "", fmt.Errorf("%w: provider catalog service is required", ErrInvalidRequest)
	}
	snapshot, _, err := service.activeSnapshot(ctx)
	if err != nil {
		return "", err
	}
	return snapshot.Revision(), nil
}

func (service *Service) List(ctx context.Context, request ListRequest) (ListResult, error) {
	if service == nil {
		return ListResult{}, fmt.Errorf("%w: provider catalog service is required", ErrInvalidRequest)
	}
	snapshot, providers, listErr := service.activeSnapshot(ctx)
	if listErr != nil {
		return ListResult{}, listErr
	}
	query, listErr := normalizeListRequest(request)
	if listErr != nil {
		return ListResult{}, listErr
	}
	queryDigest := listQueryDigest(query)
	var after *listCursor
	if request.Cursor != "" {
		decoded, err := service.cursors.decode(request.Cursor)
		if err != nil {
			return ListResult{}, err
		}
		if decoded.CatalogRevision != snapshot.Revision() {
			return ListResult{}, ErrStaleCursor
		}
		if decoded.QueryDigest != queryDigest {
			return ListResult{}, fmt.Errorf("%w: cursor filters do not match the request", ErrInvalidCursor)
		}
		after = &decoded
	}

	selected := make([]Definition, 0, query.pageSize+1)
	for _, provider := range providers {
		if after != nil && !definitionAfter(provider, *after) {
			continue
		}
		if !matchesListQuery(provider, query) {
			continue
		}
		selected = append(selected, cloneDefinition(provider))
		if len(selected) == query.pageSize+1 {
			break
		}
	}
	result := ListResult{
		CatalogRevision: snapshot.Revision(), Categories: providerCategories(providers),
		PageSize: query.pageSize, Providers: selected,
	}
	if len(selected) > query.pageSize {
		last := selected[query.pageSize-1]
		result.Providers = selected[:query.pageSize]
		result.HasMore = true
		result.NextCursor, listErr = service.cursors.encode(listCursor{
			Version: 1, CatalogRevision: snapshot.Revision(), QueryDigest: queryDigest,
			Order: last.Order, ProviderID: last.ID,
		})
		if listErr != nil {
			return ListResult{}, listErr
		}
	}
	return result, nil
}

func (service *Service) Get(ctx context.Context, providerID string) (DetailResult, error) {
	if service == nil || !idPattern.MatchString(providerID) {
		return DetailResult{}, fmt.Errorf("%w: provider ID is invalid", ErrInvalidRequest)
	}
	snapshot, _, err := service.activeSnapshot(ctx)
	if err != nil {
		return DetailResult{}, err
	}
	provider, found := snapshot.Get(providerID)
	if !found {
		return DetailResult{}, ErrNotFound
	}
	return DetailResult{CatalogRevision: snapshot.Revision(), Provider: provider}, nil
}

func (service *Service) activeSnapshot(ctx context.Context) (*Snapshot, []Definition, error) {
	snapshot, err := service.source.ActiveSnapshot(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("read active provider catalog snapshot: %w", err)
	}
	if snapshot == nil || snapshot.Revision() == "" {
		return nil, nil, fmt.Errorf("%w: active provider catalog snapshot is unavailable", ErrInvalidRequest)
	}
	providers := snapshot.List()
	if len(providers) == 0 {
		return nil, nil, fmt.Errorf("%w: active provider catalog snapshot is empty", ErrInvalidRequest)
	}
	return snapshot, providers, nil
}

func providerCategories(providers []Definition) []string {
	categorySet := make(map[string]string)
	for _, provider := range providers {
		key := strings.ToLower(provider.Display.Category)
		if _, exists := categorySet[key]; !exists {
			categorySet[key] = provider.Display.Category
		}
	}
	categories := make([]string, 0, len(categorySet))
	for _, category := range categorySet {
		categories = append(categories, category)
	}
	sort.Slice(categories, func(i, j int) bool {
		left, right := strings.ToLower(categories[i]), strings.ToLower(categories[j])
		if left == right {
			return categories[i] < categories[j]
		}
		return left < right
	})
	return categories
}

func normalizeListRequest(request ListRequest) (normalizedListQuery, error) {
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return normalizedListQuery{}, fmt.Errorf("%w: page size must be between 1 and %d", ErrInvalidRequest, maximumPageSize)
	}
	for label, value := range map[string]string{
		"search": request.Search, "category": request.Category,
	} {
		if value != "" && !canonicalText(value, 1, 128) {
			return normalizedListQuery{}, fmt.Errorf("%w: %s filter is invalid", ErrInvalidRequest, label)
		}
	}
	if request.Capability != "" && !capabilityPattern.MatchString(request.Capability) {
		return normalizedListQuery{}, fmt.Errorf("%w: capability filter is invalid", ErrInvalidRequest)
	}
	return normalizedListQuery{
		pageSize: pageSize, search: strings.ToLower(request.Search),
		category: strings.ToLower(request.Category), capability: request.Capability,
	}, nil
}

func definitionAfter(provider Definition, cursor listCursor) bool {
	return provider.Order > cursor.Order || provider.Order == cursor.Order && provider.ID > cursor.ProviderID
}

func matchesListQuery(provider Definition, query normalizedListQuery) bool {
	if query.category != "" && strings.ToLower(provider.Display.Category) != query.category {
		return false
	}
	if query.capability != "" {
		found := false
		for _, capability := range provider.Capabilities {
			if capability == query.capability {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	if query.search == "" {
		return true
	}
	return strings.Contains(strings.ToLower(provider.ID), query.search) ||
		strings.Contains(strings.ToLower(provider.Display.Name), query.search) ||
		strings.Contains(strings.ToLower(provider.Display.Description), query.search)
}
