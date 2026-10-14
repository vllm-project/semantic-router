package routing

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// The rules below are Envoy's for applying ext_proc mutations (Envoy v1.35,
// the version the CLI ships, with the local template's default mutation
// rules). A gateway that applies effects with them forwards the same request
// and returns the same response as Envoy does for the same effects.

// Limits bounds a header map after mutation, like Envoy's HTTP connection
// manager limits.
type Limits struct {
	MaxHeaderCount int
	MaxHeaderBytes int
}

// DefaultLimits are Envoy's connection manager defaults: 100 headers, 60 KiB.
var DefaultLimits = Limits{MaxHeaderCount: 100, MaxHeaderBytes: 60 * 1024}

// ErrMutation reports a mutation Envoy rejects; the request fails.
var ErrMutation = errors.New("invalid mutation")

// ApplyHeaderMutation applies m to h: every removal first, then every set in
// order. Mutations of routing headers (host, :authority, :method, :scheme),
// removals of pseudo-headers, appends to pseudo-headers and changes to
// x-envoy-* headers are ignored, as Envoy ignores them. With
// skipContentLength, content-length sets are skipped, as Envoy does when the
// body is streamed.
func ApplyHeaderMutation(h *Header, m *HeaderMutation, skipContentLength bool, limits Limits) error {
	if m == nil {
		return nil
	}
	if len(m.Remove) > limits.MaxHeaderCount || len(m.Set) > limits.MaxHeaderCount {
		return fmt.Errorf("%w: %d removals or %d sets exceed %d headers", ErrMutation, len(m.Remove), len(m.Set), limits.MaxHeaderCount)
	}
	for _, name := range m.Remove {
		if !validHeaderName(name) {
			return fmt.Errorf("%w: invalid header name to remove %q", ErrMutation, name)
		}
		name = strings.ToLower(name)
		if mutationAllowed(name, false, true) {
			h.Del(name)
		}
	}
	for _, option := range m.Set {
		if skipContentLength && strings.EqualFold(option.Name, "content-length") {
			continue
		}
		if !validHeaderName(option.Name) || !validHeaderValue(option.Value) {
			return fmt.Errorf("%w: invalid header %q", ErrMutation, option.Name)
		}
		name := strings.ToLower(option.Name)
		appending := option.Append && h.Has(name)
		if !mutationAllowed(name, appending, false) || !validSetValue(name, option.Value) {
			continue
		}
		if option.Append {
			h.Add(name, option.Value)
		} else {
			h.Set(name, option.Value)
		}
	}
	return checkHeaderLimits(*h, limits)
}

// ApplyBodyMutation returns body after m.
func ApplyBodyMutation(body []byte, m *BodyMutation) []byte {
	switch {
	case m == nil:
		return body
	case m.Clear:
		return nil
	default:
		return m.Body
	}
}

// CheckContentLength enforces Envoy's rule for buffered body mutations: when
// the held headers declare a content-length, the new body must have exactly
// that length.
func CheckContentLength(h Header, m *BodyMutation) error {
	if m == nil {
		return nil
	}
	declared := h.Get("content-length")
	if declared == "" {
		return nil
	}
	length, err := strconv.Atoi(declared)
	if err != nil {
		return nil
	}
	size := 0
	if !m.Clear {
		size = len(m.Body)
	}
	if length != size {
		return fmt.Errorf("%w: content-length %d does not match the mutated body (%d bytes)", ErrMutation, length, size)
	}
	return nil
}

// RenderImmediate builds the client response Envoy sends for an immediate
// response (a local reply): the status, the header mutation applied to a
// fresh header map, and, for a non-empty body, its length and a text/plain
// content type unless one was set.
func RenderImmediate(ir *ImmediateResponse, limits Limits) *Response {
	status := ir.Status
	if status < 200 {
		status = 200
	}
	header := Header{{Name: ":status", Value: strconv.Itoa(status)}}
	// Envoy logs a failed local-reply mutation and sends the reply anyway.
	_ = ApplyHeaderMutation(&header, ir.Header, false, limits)
	if len(ir.Body) > 0 {
		header.Set("content-length", strconv.Itoa(len(ir.Body)))
		if !header.Has("content-type") {
			header.Set("content-type", "text/plain")
		}
	} else {
		header.Del("content-length")
		header.Del("content-type")
	}
	if code, err := strconv.Atoi(header.Get(":status")); err == nil {
		status = code
	}
	return &Response{Status: status, Header: header.WithoutPseudo(), Body: ir.Body}
}

// mutationAllowed mirrors Envoy's default HeaderMutationRules.
func mutationAllowed(name string, appending, removing bool) bool {
	if removing && (name == "host" || isPseudoHeader(name)) {
		return false
	}
	switch name {
	case "host", ":authority", ":method", ":scheme":
		return false
	}
	if isPseudoHeader(name) && appending {
		return false
	}
	return !strings.HasPrefix(name, "x-envoy-")
}

func validSetValue(name, value string) bool {
	if name != ":status" {
		return true
	}
	code, err := strconv.Atoi(value)
	return err == nil && code >= 200
}

func validHeaderName(name string) bool {
	if name == "" {
		return false
	}
	start := 0
	if name[0] == ':' {
		start = 1
		if len(name) == 1 {
			return false
		}
	}
	for i := start; i < len(name); i++ {
		if !isTokenChar(name[i]) || (start == 1 && name[i] >= 'A' && name[i] <= 'Z') {
			return false
		}
	}
	return true
}

func isTokenChar(c byte) bool {
	switch {
	case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		return true
	}
	return strings.IndexByte("!#$%&'*+-.^_`|~", c) >= 0
}

func validHeaderValue(value string) bool {
	for i := 0; i < len(value); i++ {
		c := value[i]
		if (c < 0x20 && c != '\t') || c == 0x7f {
			return false
		}
	}
	return true
}

func checkHeaderLimits(h Header, limits Limits) error {
	size := 0
	for _, field := range h {
		size += len(field.Name) + len(field.Value)
	}
	if len(h) > limits.MaxHeaderCount || size > limits.MaxHeaderBytes {
		return fmt.Errorf("%w: %d headers (%d bytes) exceed the limits", ErrMutation, len(h), size)
	}
	return nil
}
