package upstream

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/textproto"
	"net/url"
	"strings"

	"golang.org/x/net/http/httpguts"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Request is one call to send upstream: the request as the router core left
// it after its own mutations, plus the route key that selects the cluster.
type Request struct {
	Method string
	// Path is the request target, path and query, sent byte for byte.
	Path   string
	Header http.Header
	// Body is the complete request body. Every attempt sends it again.
	Body []byte
	// RouteKey selects the cluster (the x-selected-model value). An empty key,
	// or one no cluster serves, takes the default route.
	RouteKey string
	// Listener names the frontend listener whose route defaults apply.
	Listener string
	// Reliability are the call's overrides, lowest first (the matched
	// decision's, then a request-graph step's), each merged over the
	// cluster's policy as Envoy merges its per-request headers.
	Reliability []*routing.Reliability
	// Policy, when set, overrides the cluster's policy for this call, above
	// Reliability.
	Policy *Policy
}

// internalRequestHeaders never reach a model backend: the Envoy template
// removes them from every upstream request.
var internalRequestHeaders = []string{
	headers.VSRLooperRequest,
	"x-vsr-looper-secret",
	headers.VSRLooperDecision,
	headers.VSRLooperIteration,
	headers.AuthzUserID,
	headers.AuthzUserGroups,
}

// hopByHopHeaders are scoped to one connection and are never forwarded.
var hopByHopHeaders = []string{
	"Connection", "Keep-Alive", "Proxy-Connection", "Transfer-Encoding", "Upgrade", "Te", "Trailer", "Expect",
}

func (r *Request) validate() *Error {
	if r == nil {
		return invalidRequest("request is nil")
	}
	if !httpguts.ValidHeaderFieldName(r.Method) {
		return invalidRequest("invalid method %q", r.Method)
	}
	if !strings.HasPrefix(r.Path, "/") || strings.ContainsFunc(r.Path, invalidTargetRune) {
		return invalidRequest("invalid request path %q", r.Path)
	}
	for name, values := range r.Header {
		if strings.HasPrefix(name, ":") {
			continue
		}
		if !httpguts.ValidHeaderFieldName(name) {
			return invalidRequest("invalid header name %q", name)
		}
		for _, value := range values {
			if !httpguts.ValidHeaderFieldValue(value) {
				return invalidRequest("invalid value for header %q", name)
			}
		}
	}
	return nil
}

func invalidTargetRune(r rune) bool { return r <= ' ' || r == 0x7f }

// outbound builds the request for one attempt against one endpoint, applying
// the route behavior of the Envoy template in its order: route headers are
// set (replacing existing values), internal headers are removed, the host is
// rewritten to the endpoint's authority, and the default route rewrites a
// /v1 path onto the provider's base path.
func outbound(ctx context.Context, req *Request, spec *ClusterSpec, ep *EndpointSpec, defaultRoute bool) (*http.Request, error) {
	target := req.Path
	if defaultRoute {
		target = rewriteDefaultPath(target, spec.PathPrefix)
	}
	u, err := requestURL(ep.Scheme, ep.Address(), target)
	if err != nil {
		return nil, invalidRequest("%v", err)
	}
	header := outboundHeader(req.Header)
	for _, h := range spec.RouteHeaders {
		header.Set(h.Name, h.Value)
	}
	for _, name := range internalRequestHeaders {
		header.Del(name)
	}
	body := req.Body
	out := (&http.Request{
		Method:        req.Method,
		URL:           u,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        header,
		Host:          ep.Authority(),
		ContentLength: int64(len(body)),
	}).WithContext(ctx)
	if len(body) > 0 {
		out.Body = io.NopCloser(bytes.NewReader(body))
		out.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	}
	return out, nil
}

// outboundHeader copies the caller's header with canonical names, without
// pseudo-headers, hop-by-hop headers, or the fields the transport writes
// itself. A User-Agent stays absent rather than taking Go's default.
func outboundHeader(in http.Header) http.Header {
	out := make(http.Header, len(in)+1)
	for name, values := range in {
		if strings.HasPrefix(name, ":") {
			continue
		}
		key := textproto.CanonicalMIMEHeaderKey(name)
		out[key] = append(out[key], values...)
	}
	removeHopByHop(out)
	out.Del("Host")
	out.Del("Content-Length")
	if _, ok := out["User-Agent"]; !ok {
		out["User-Agent"] = nil
	}
	return out
}

// rewriteDefaultPath applies the default route's regex_rewrite
// "^/v1([/?].*)?$" -> "<prefix>\1": only the complete /v1 segment is
// replaced, so /v1beta/... and other paths pass through.
func rewriteDefaultPath(target, prefix string) string {
	if prefix == "" {
		return target
	}
	if target == "/v1" || strings.HasPrefix(target, "/v1/") || strings.HasPrefix(target, "/v1?") {
		return prefix + target[len("/v1"):]
	}
	return target
}

// requestURL addresses an endpoint while keeping the target's path bytes as
// given: the opaque form writes them to the request line unchanged.
func requestURL(scheme, address, target string) (*url.URL, error) {
	path, query, _ := strings.Cut(target, "?")
	u := &url.URL{Scheme: scheme, Host: address, RawQuery: query, ForceQuery: strings.HasSuffix(target, "?")}
	if !strings.HasPrefix(path, "//") {
		u.Opaque = path
		return u, nil
	}
	// An opaque value starting with "//" would read as an authority.
	unescaped, err := url.PathUnescape(path)
	if err != nil {
		return nil, err
	}
	u.Path, u.RawPath = unescaped, path
	return u, nil
}

// removeHopByHop drops connection-scoped fields, and every field the
// Connection header nominates, from a request or response header.
func removeHopByHop(h http.Header) {
	for _, nominated := range h.Values("Connection") {
		for _, name := range strings.Split(nominated, ",") {
			if name = strings.TrimSpace(name); name != "" {
				h.Del(name)
			}
		}
	}
	for _, name := range hopByHopHeaders {
		h.Del(name)
	}
}
