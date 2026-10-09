package parity

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Placeholder replaces a volatile value.
const Placeholder = "<volatile>"

// volatileHeaders change on every run without changing behavior: trace
// context, generated identifiers and clocks. Measured latencies are matched
// by suffix in VolatileHeader.
var volatileHeaders = map[string]bool{
	"date":                          true,
	"traceparent":                   true,
	"tracestate":                    true,
	"x-envoy-upstream-service-time": true,
	"x-vsr-replay-id":               true,
}

// VolatileHeader reports whether a header's value is normalized away.
func VolatileHeader(name string) bool {
	name = strings.ToLower(name)
	return volatileHeaders[name] || strings.HasSuffix(name, "-latency-ms")
}

// volatileJSONFields are body fields that carry the clock.
var volatileJSONFields = map[string]bool{"created": true, "created_at": true}

// NormalizeBody sets clock fields to 0 in a JSON body, or in each JSON event
// of an SSE body, and keeps every other byte as it was.
func NormalizeBody(body []byte) []byte {
	if out, ok := zeroClockFields(body); ok {
		return out
	}
	if !bytes.Contains(body, []byte("data:")) {
		return body
	}
	lines := bytes.Split(body, []byte("\n"))
	for i, line := range lines {
		payload, found := bytes.CutPrefix(line, []byte("data: "))
		if !found {
			continue
		}
		if out, ok := zeroClockFields(payload); ok {
			lines[i] = append([]byte("data: "), out...)
		}
	}
	return bytes.Join(lines, []byte("\n"))
}

// zeroClockFields rewrites the number tokens of clock fields in one JSON
// document. It reports false when body is not exactly one JSON document.
func zeroClockFields(body []byte) ([]byte, bool) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || (trimmed[0] != '{' && trimmed[0] != '[') {
		return nil, false
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	type frame struct{ object, expectKey bool }
	var stack []frame
	var spans [][2]int
	clockValue := false
	for {
		start := int(decoder.InputOffset())
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, false
		}
		end := int(decoder.InputOffset())
		top := len(stack) - 1
		if delim, isDelim := token.(json.Delim); isDelim {
			switch delim {
			case '{', '[':
				stack = append(stack, frame{object: delim == '{', expectKey: delim == '{'})
			default:
				stack = stack[:top]
				if top > 0 && stack[top-1].object {
					stack[top-1].expectKey = true
				}
			}
			clockValue = false
			if len(stack) == 0 {
				break
			}
			continue
		}
		if top >= 0 && stack[top].object && stack[top].expectKey {
			key, _ := token.(string)
			clockValue = volatileJSONFields[key]
			stack[top].expectKey = false
			continue
		}
		if _, isNumber := token.(json.Number); isNumber && clockValue {
			spans = append(spans, [2]int{start + bytes.IndexFunc(body[start:end], isNumberStart), end})
		}
		clockValue = false
		if top >= 0 && stack[top].object {
			stack[top].expectKey = true
		}
	}
	if len(stack) != 0 || len(bytes.TrimSpace(body[decoder.InputOffset():])) != 0 {
		return nil, false
	}
	if len(spans) == 0 {
		return body, true
	}
	out := make([]byte, 0, len(body))
	last := 0
	for _, span := range spans {
		out = append(out, body[last:span[0]]...)
		out = append(out, '0')
		last = span[1]
	}
	return append(out, body[last:]...), true
}

func isNumberStart(r rune) bool {
	return r == '-' || (r >= '0' && r <= '9')
}

// NormalizeHeader replaces volatile header values.
func NormalizeHeader(h routing.Header) routing.Header {
	out := h.Clone()
	for i := range out {
		if VolatileHeader(out[i].Name) {
			out[i].Value = Placeholder
		}
	}
	return out
}

// NormalizeMutation replaces volatile values and sorts removals: Envoy applies
// every removal before any set, so their order carries no meaning, and some
// removal lists come from map iteration.
func NormalizeMutation(m *routing.HeaderMutation) *routing.HeaderMutation {
	if m == nil {
		return nil
	}
	out := &routing.HeaderMutation{
		Set:    append([]routing.HeaderOption(nil), m.Set...),
		Remove: append([]string(nil), m.Remove...),
	}
	for i := range out.Set {
		if VolatileHeader(out.Set[i].Name) {
			out.Set[i].Value = Placeholder
		}
	}
	sort.Strings(out.Remove)
	return out
}
