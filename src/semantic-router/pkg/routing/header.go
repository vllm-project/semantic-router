package routing

import "strings"

// HeaderField is one HTTP header field. Names are lowercase.
type HeaderField struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// Header is an ordered list of header fields in the shape Envoy hands to
// ext_proc: lowercase names, pseudo-headers (":method", ":path",
// ":authority", ":scheme", ":status") before regular headers, and a repeated
// name kept as separate fields.
type Header []HeaderField

// Get returns the first value of name, or "" when it is absent.
func (h Header) Get(name string) string {
	name = strings.ToLower(name)
	for _, field := range h {
		if field.Name == name {
			return field.Value
		}
	}
	return ""
}

// Values returns every value of name in order.
func (h Header) Values(name string) []string {
	name = strings.ToLower(name)
	var values []string
	for _, field := range h {
		if field.Name == name {
			values = append(values, field.Value)
		}
	}
	return values
}

// Has reports whether name is present.
func (h Header) Has(name string) bool {
	name = strings.ToLower(name)
	for _, field := range h {
		if field.Name == name {
			return true
		}
	}
	return false
}

// Set replaces every value of name with value. Like Envoy's setCopy, the new
// field takes the position a new header would.
func (h *Header) Set(name, value string) {
	h.Del(name)
	h.Add(name, value)
}

// Add appends a value for name. Pseudo-headers stay ahead of regular headers.
func (h *Header) Add(name, value string) {
	field := HeaderField{Name: strings.ToLower(name), Value: value}
	if !isPseudoHeader(field.Name) {
		*h = append(*h, field)
		return
	}
	end := 0
	for end < len(*h) && isPseudoHeader((*h)[end].Name) {
		end++
	}
	*h = append(*h, HeaderField{})
	copy((*h)[end+1:], (*h)[end:])
	(*h)[end] = field
}

// Del removes every value of name.
func (h *Header) Del(name string) {
	name = strings.ToLower(name)
	kept := (*h)[:0]
	for _, field := range *h {
		if field.Name != name {
			kept = append(kept, field)
		}
	}
	*h = kept
}

// Clone returns an independent copy.
func (h Header) Clone() Header {
	if h == nil {
		return nil
	}
	return append(Header(nil), h...)
}

// WithoutPseudo returns the regular headers only.
func (h Header) WithoutPseudo() Header {
	out := make(Header, 0, len(h))
	for _, field := range h {
		if !isPseudoHeader(field.Name) {
			out = append(out, field)
		}
	}
	return out
}

func isPseudoHeader(name string) bool {
	return strings.HasPrefix(name, ":")
}
