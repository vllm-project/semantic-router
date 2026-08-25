// Package providerdiscovery executes model discovery in the Management control
// plane. Provider products remain declarative catalog data; adapters implement
// only stable discovery wire protocols.
package providerdiscovery

import (
	"context"
	"errors"
	"io"
	"net/url"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

var (
	ErrInvalidRequest     = errors.New("provider discovery request is invalid")
	ErrAdapterUnavailable = errors.New("provider discovery adapter is unavailable")
	ErrCredentialMismatch = errors.New("provider credential does not match discovery binding")
	ErrUpstream           = errors.New("provider discovery upstream failed")
	ErrInvalidResponse    = errors.New("provider discovery response is invalid")
	ErrInvalidClaim       = errors.New("provider discovery claim is invalid")
	ErrExpiredClaim       = errors.New("provider discovery claim has expired")
)

// Adapter is one stable control-plane discovery wire protocol. It never sees
// credential material and cannot choose an origin or path.
type Adapter interface {
	providercatalog.DiscoveryRequestValidator
	Query(providercatalog.DiscoveryPlan) (url.Values, error)
	Decode(providercatalog.DiscoveryPlan, io.Reader) (AdapterPage, error)
}

type AdapterModel struct {
	ProviderModelID string
	DisplayName     string
	// Capabilities contains model-specific evidence decoded from the discovery
	// response. Transport support declared by the Provider is not a fallback.
	Capabilities []string
}

type AdapterPage struct {
	Models     []AdapterModel
	NextCursor string
	HasMore    bool
}

type Model struct {
	CatalogItemID   string
	ProviderModelID string
	DisplayName     string
	Capabilities    []string
}

type Result struct {
	Models            []Model
	NextCursor        string
	HasMore           bool
	CatalogRevision   string
	DiscoveryRevision string
	ExpiresAt         time.Time
}

// ExecuteRequest contains only trusted authority context and the validated
// catalog plan. AuthorityDigest is a stable digest of the permissions used to
// authorize discovery and later bulk import.
type ExecuteRequest struct {
	Plan            providercatalog.DiscoveryPlan
	AuthorityDigest string
}

// CredentialResolver pins and resolves a ProviderCredential version. It is
// intentionally the same narrow contract used by physical backend dispatch.
type CredentialResolver interface {
	Pin(context.Context, string, string, string) (string, error)
	ResolvePinned(context.Context, string, string, string, string) (backendinvoker.Credential, error)
}
