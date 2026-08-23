package managementidentity

import (
	"context"
	"strings"
	"unicode"
)

const (
	defaultDirectoryPageSize = 50
	maximumDirectoryPageSize = 200
	maximumDirectorySearch   = 128
)

func (service *Service) GetPrincipalDirectoryEntry(
	ctx context.Context,
	namespaceID string,
	principalID string,
) (PrincipalDirectoryEntry, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(principalID) {
		return PrincipalDirectoryEntry{}, ErrNotFound
	}
	return service.repository.GetPrincipalDirectoryEntry(ctx, namespaceID, principalID)
}

func (service *Service) ListPrincipalDirectory(
	ctx context.Context,
	request PrincipalDirectoryRequest,
) (PrincipalDirectoryPage, error) {
	request.Search = strings.TrimSpace(request.Search)
	if service == nil || !canonicalUUID(request.NamespaceID) ||
		!validDirectoryPage(request.AfterID, request.Limit) || !validDirectorySearch(request.Search) {
		return PrincipalDirectoryPage{}, ErrInvalidLifecycleRequest
	}
	if request.Limit == 0 {
		request.Limit = defaultDirectoryPageSize
	}
	return service.repository.ListPrincipalDirectory(ctx, request)
}

func (service *Service) ListPrincipalUserLinks(
	ctx context.Context,
	request PrincipalUserLinkListRequest,
) (PrincipalUserLinkPage, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) ||
		(request.PrincipalID != "" && !canonicalUUID(request.PrincipalID)) ||
		(request.UserID != "" && !canonicalUUID(request.UserID)) ||
		!validDirectoryPage(request.AfterID, request.Limit) {
		return PrincipalUserLinkPage{}, ErrInvalidLifecycleRequest
	}
	if request.Limit == 0 {
		request.Limit = defaultDirectoryPageSize
	}
	return service.repository.ListPrincipalUserLinks(ctx, request)
}

func (service *Service) ListPrincipalLinks(
	ctx context.Context,
	principalID string,
	request ListRequest,
) (PrincipalUserLinkPage, error) {
	if service == nil || !canonicalUUID(principalID) || !validDirectoryPage(request.AfterID, request.Limit) {
		return PrincipalUserLinkPage{}, ErrInvalidLifecycleRequest
	}
	if request.Limit == 0 {
		request.Limit = defaultDirectoryPageSize
	}
	return service.repository.ListPrincipalLinks(ctx, principalID, request)
}

func validDirectoryPage(afterID string, limit int) bool {
	return (afterID == "" || canonicalUUID(afterID)) && limit >= 0 && limit <= maximumDirectoryPageSize
}

func validDirectorySearch(value string) bool {
	if len(value) > maximumDirectorySearch {
		return false
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return false
		}
	}
	return true
}
