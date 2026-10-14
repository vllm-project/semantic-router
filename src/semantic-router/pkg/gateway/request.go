package gateway

import (
	"crypto/rand"
	"fmt"
	"net/http"
	"net/textproto"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// hopByHopHeaders belong to one connection; Envoy's connection manager never
// passes them to filters or upstream.
var hopByHopHeaders = map[string]bool{
	"connection":        true,
	"keep-alive":        true,
	"proxy-connection":  true,
	"transfer-encoding": true,
	"upgrade":           true,
	"te":                true,
}

// routingRequest builds the request the way Envoy's HTTP connection manager
// hands it to ext_proc: pseudo-headers, the client's headers in lowercase
// (sorted, since Go does not keep their order) without the proxy-control
// headers only a trusted proxy may set, the forwarded protocol, and a request
// ID. strip names identity headers the gateway does not trust.
func routingRequest(r *http.Request, body []byte, strip map[string]bool) *routing.Request {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	header := routing.Header{
		{Name: ":method", Value: r.Method},
		{Name: ":path", Value: requestTarget(r)},
		{Name: ":authority", Value: r.Host},
		{Name: ":scheme", Value: scheme},
	}
	dropped := connectionHeaders(r.Header)
	for _, name := range sortedNames(r.Header) {
		lower := strings.ToLower(name)
		if hopByHopHeaders[lower] || dropped[lower] || strip[lower] || routing.IsProxyControlHeader(lower) {
			continue
		}
		for _, value := range r.Header[name] {
			header = append(header, routing.HeaderField{Name: lower, Value: value})
		}
	}
	if !header.Has("x-forwarded-proto") {
		header = append(header, routing.HeaderField{Name: "x-forwarded-proto", Value: scheme})
	}
	if !header.Has("x-request-id") {
		header = append(header, routing.HeaderField{Name: "x-request-id", Value: newRequestID()})
	}
	return &routing.Request{Header: header, Body: body}
}

// requestTarget is the path and query exactly as the client sent them.
func requestTarget(r *http.Request) string {
	if strings.HasPrefix(r.RequestURI, "/") {
		return r.RequestURI
	}
	return r.URL.RequestURI()
}

// connectionHeaders lists the headers the Connection header names, which are
// hop-by-hop too.
func connectionHeaders(h http.Header) map[string]bool {
	names := map[string]bool{}
	for _, value := range h.Values("Connection") {
		for _, name := range strings.Split(value, ",") {
			if name = strings.TrimSpace(strings.ToLower(name)); name != "" {
				names[name] = true
			}
		}
	}
	return names
}

func sortedNames(h http.Header) []string {
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// newRequestID returns a random UUID, as Envoy generates for a request that
// arrives without one.
func newRequestID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// identityHeaders are set by an authenticator in front of the Router in
// Envoy deployments. The native gateway has no such authenticator, so a
// client-sent value is dropped before routing sees it.
var identityHeaders = []string{"x-authz-user-id", "x-authz-user-groups", "x-authz-team-id", "x-authz-tenant-id"}

func headerSet(names []string) map[string]bool {
	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[strings.ToLower(textproto.TrimString(name))] = true
	}
	return set
}
