package managementserver

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

var (
	ErrNamespaceRequired = errors.New("a Management namespace is required")
	ErrNamespaceConflict = errors.New("management namespace selectors conflict")
)

// ExplicitNamespaceResolver implements the one public namespace-selection
// contract for unscoped Management resource URLs. A path-scoped namespace is
// authoritative; when the header is also present it must match exactly.
type ExplicitNamespaceResolver struct{}

func (ExplicitNamespaceResolver) ResolveNamespace(
	_ context.Context,
	request *http.Request,
) (string, error) {
	if request == nil || request.URL == nil {
		return "", ErrNamespaceRequired
	}
	pathNamespace := namespaceFromPath(request.URL.Path)
	headerNamespace, headerPresent, err := namespaceFromHeader(request.Header)
	if err != nil {
		return "", err
	}
	if pathNamespace != "" {
		if !canonicalUUID(pathNamespace) {
			return "", ErrNamespaceRequired
		}
		if headerPresent && headerNamespace != pathNamespace {
			return "", ErrNamespaceConflict
		}
		return pathNamespace, nil
	}
	if !headerPresent || !canonicalUUID(headerNamespace) {
		return "", ErrNamespaceRequired
	}
	return headerNamespace, nil
}

func namespaceFromHeader(header http.Header) (string, bool, error) {
	values := header.Values(managementapi.HeaderNamespaceID)
	if len(values) == 0 {
		return "", false, nil
	}
	if len(values) != 1 || values[0] == "" || strings.TrimSpace(values[0]) != values[0] || strings.Contains(values[0], ",") {
		return "", false, ErrNamespaceRequired
	}
	return values[0], true, nil
}

func namespaceFromPath(path string) string {
	prefix := managementapi.BasePath + "/namespaces/"
	if !strings.HasPrefix(path, prefix) {
		return ""
	}
	value := strings.TrimPrefix(path, prefix)
	segment, _, _ := strings.Cut(value, "/")
	return segment
}
