package pluginruntime

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"golang.org/x/net/http/httpguts"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// RequestPlugin is a decision plugin that acts on the provider-bound request
// once its decision has selected a model. A registered plugin type whose
// payload implements it runs there, after the built-in plugins, with no
// change to the request pipeline.
type RequestPlugin interface {
	OnRequest(ctx context.Context, request *PluginRequest) error
}

// ResponsePlugin is a decision plugin that acts on the response headers the
// client receives.
type ResponsePlugin interface {
	OnResponse(ctx context.Context, response *PluginResponse) error
}

// PluginRequest is the provider-bound request a RequestPlugin sees and
// changes.
type PluginRequest struct {
	// Decision and Model name the decision that matched and the model it
	// selected.
	Decision string
	Model    string

	header    func(name string) string
	mutations []HeaderMutation
}

// NewPluginRequest returns the request a plugin sees; header reads the
// client's request headers.
func NewPluginRequest(decision, model string, header func(name string) string) *PluginRequest {
	return &PluginRequest{Decision: decision, Model: model, header: header}
}

// Header returns a client request header.
func (r *PluginRequest) Header(name string) string {
	if r.header == nil {
		return ""
	}
	return r.header(strings.ToLower(name))
}

// SetHeader sets a provider-bound header.
func (r *PluginRequest) SetHeader(name, value string) error {
	if err := writableHeader(name); err != nil {
		return err
	}
	r.mutations = append(r.mutations, HeaderMutation{Name: strings.ToLower(name), Value: value, Operation: "set"})
	return nil
}

// RemoveHeader removes a provider-bound header.
func (r *PluginRequest) RemoveHeader(name string) error {
	if err := writableHeader(name); err != nil {
		return err
	}
	r.mutations = append(r.mutations, HeaderMutation{Name: strings.ToLower(name), Operation: "remove"})
	return nil
}

// Mutations returns the header changes in the order the plugin made them.
func (r *PluginRequest) Mutations() []HeaderMutation { return slices.Clone(r.mutations) }

// PluginResponse is the client response a ResponsePlugin sees and changes.
type PluginResponse struct {
	Decision string
	Model    string
	// Status is the upstream response status.
	Status int

	mutations []HeaderMutation
}

// NewPluginResponse returns the response a plugin sees.
func NewPluginResponse(decision, model string, status int) *PluginResponse {
	return &PluginResponse{Decision: decision, Model: model, Status: status}
}

// SetHeader sets a client response header.
func (r *PluginResponse) SetHeader(name, value string) error {
	if err := writableHeader(name); err != nil {
		return err
	}
	r.mutations = append(r.mutations, HeaderMutation{Name: strings.ToLower(name), Value: value, Operation: "set"})
	return nil
}

// Mutations returns the header changes in the order the plugin made them.
func (r *PluginResponse) Mutations() []HeaderMutation { return slices.Clone(r.mutations) }

// routerHeaders are the headers a plugin may not write: they belong to one
// connection, frame the body, or are the route key Envoy routes on, which
// would send a request elsewhere behind Envoy and nowhere in standalone mode.
var routerHeaders = []string{
	"connection", "keep-alive", "proxy-connection", "proxy-authenticate", "proxy-authorization",
	"te", "trailer", "transfer-encoding", "upgrade", "expect", "host", "content-length",
	headers.SelectedModel,
}

// routerNamespaces hold the Router's own facts and controls and Envoy's,
// such as the reliability headers.
var routerNamespaces = []string{"x-vsr-", "x-envoy-"}

// writableHeader refuses names that are not valid header names, which
// include pseudo-headers, and the headers the Router owns.
func writableHeader(name string) error {
	normalized := strings.ToLower(strings.TrimSpace(name))
	if !httpguts.ValidHeaderFieldName(normalized) {
		return fmt.Errorf("header %q is not a valid header name", name)
	}
	if slices.Contains(routerHeaders, normalized) || slices.ContainsFunc(routerNamespaces, func(prefix string) bool {
		return strings.HasPrefix(normalized, prefix)
	}) {
		return fmt.Errorf("header %q is managed by the router", name)
	}
	return nil
}
