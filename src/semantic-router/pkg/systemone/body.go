package systemone

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
)

// maxNesting is encoding/json's nesting limit: a deeper document is invalid there too.
const maxNesting = 10000

// scannedRequest is what the public handler reads of a native request body:
// its model and where its options are.
type scannedRequest struct {
	// model is the last top-level "model" member decoded as a string, and empty
	// when that member is absent, null or not a string.
	model string
	// options is the value of the top-level "options" member (nil when absent)
	// and body[optionsStart:optionsEnd] its place; repeatedOptions reports a
	// body with more than one such member.
	options                  []byte
	optionsStart, optionsEnd int
	repeatedOptions          bool
}

// scanRequest reads a native request body as json.Unmarshal reads it into a
// map[string]json.RawMessage, without decoding the media it carries. ok is
// false unless the body is one JSON object with the syntax json.Valid accepts.
// Strings are scanned a word at a time, so megabytes of base64 images or
// videos cost a memory scan instead of encoding/json's byte-at-a-time state
// machine.
func scanRequest(body []byte) (request scannedRequest, ok bool) {
	s := bodyScanner{data: body}
	start := s.space(0)
	if start >= len(body) || body[start] != '{' {
		return scannedRequest{}, false
	}
	end := s.object(start)
	if end < 0 || s.space(end) != len(body) {
		return scannedRequest{}, false
	}
	if s.model != nil && json.Unmarshal(s.model, &request.model) != nil {
		request.model = ""
	}
	request.options, request.optionsStart, request.optionsEnd = s.options, s.optionsStart, s.optionsEnd
	request.repeatedOptions = s.optionsCount > 1
	return request, true
}

// withOptions returns body with the value of its options member replaced by
// options, or body itself when it has no options or they are already options.
func (r scannedRequest) withOptions(body, options []byte) []byte {
	if r.options == nil || bytes.Equal(r.options, options) {
		return body
	}
	replaced := make([]byte, 0, len(body)-len(r.options)+len(options))
	replaced = append(replaced, body[:r.optionsStart]...)
	replaced = append(replaced, options...)
	return append(replaced, body[r.optionsEnd:]...)
}

type bodyScanner struct {
	data  []byte
	depth int
	// model is the value of the last top-level "model" member.
	model []byte
	// options is the value of the last top-level "options" member, at
	// data[optionsStart:optionsEnd]; optionsCount counts those members.
	options                  []byte
	optionsStart, optionsEnd int
	optionsCount             int
}

func (s *bodyScanner) space(i int) int {
	for i < len(s.data) {
		switch s.data[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// value returns the index after the JSON value that starts at i, or -1.
func (s *bodyScanner) value(i int) int {
	if i >= len(s.data) {
		return -1
	}
	switch c := s.data[i]; {
	case c == '"':
		return s.text(i)
	case c == '{':
		return s.object(i)
	case c == '[':
		return s.array(i)
	case c == 't':
		return s.literal(i, "true")
	case c == 'f':
		return s.literal(i, "false")
	case c == 'n':
		return s.literal(i, "null")
	case c == '-' || c >= '0' && c <= '9':
		return s.number(i)
	}
	return -1
}

func (s *bodyScanner) object(i int) int {
	s.depth++
	if s.depth > maxNesting {
		return -1
	}
	top := s.depth == 1
	i = s.space(i + 1)
	if i < len(s.data) && s.data[i] == '}' {
		s.depth--
		return i + 1
	}
	for {
		if i >= len(s.data) || s.data[i] != '"' {
			return -1
		}
		keyEnd := s.text(i)
		if keyEnd < 0 {
			return -1
		}
		key := s.data[i:keyEnd]
		i = s.space(keyEnd)
		if i >= len(s.data) || s.data[i] != ':' {
			return -1
		}
		valueStart := s.space(i + 1)
		valueEnd := s.value(valueStart)
		if valueEnd < 0 {
			return -1
		}
		if top && isKey(key, "model") {
			s.model = s.data[valueStart:valueEnd]
		}
		if top && isKey(key, "options") {
			s.options, s.optionsStart, s.optionsEnd = s.data[valueStart:valueEnd], valueStart, valueEnd
			s.optionsCount++
		}
		i = s.space(valueEnd)
		if i < len(s.data) && s.data[i] == ',' {
			i = s.space(i + 1)
			continue
		}
		if i < len(s.data) && s.data[i] == '}' {
			s.depth--
			return i + 1
		}
		return -1
	}
}

func (s *bodyScanner) array(i int) int {
	s.depth++
	if s.depth > maxNesting {
		return -1
	}
	i = s.space(i + 1)
	if i < len(s.data) && s.data[i] == ']' {
		s.depth--
		return i + 1
	}
	for {
		end := s.value(i)
		if end < 0 {
			return -1
		}
		i = s.space(end)
		if i < len(s.data) && s.data[i] == ',' {
			i = s.space(i + 1)
			continue
		}
		if i < len(s.data) && s.data[i] == ']' {
			s.depth--
			return i + 1
		}
		return -1
	}
}

// text returns the index after the string that starts at i, or -1. Plain runs
// between escapes are checked eight bytes at a time.
func (s *bodyScanner) text(i int) int {
	data := s.data
	j := i + 1
	quote := -1
	for {
		if quote < j {
			k := bytes.IndexByte(data[j:], '"')
			if k < 0 {
				return -1
			}
			quote = j + k
		}
		run := data[j:quote]
		slash := bytes.IndexByte(run, '\\')
		if slash < 0 {
			if hasControl(run) {
				return -1
			}
			return quote + 1
		}
		if hasControl(run[:slash]) {
			return -1
		}
		j += slash
		if j+1 >= len(data) {
			return -1
		}
		switch data[j+1] {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			j += 2
		case 'u':
			if j+6 > len(data) || !isHex(data[j+2]) || !isHex(data[j+3]) || !isHex(data[j+4]) || !isHex(data[j+5]) {
				return -1
			}
			j += 6
		default:
			return -1
		}
	}
}

func (s *bodyScanner) literal(i int, word string) int {
	if len(s.data)-i < len(word) || string(s.data[i:i+len(word)]) != word {
		return -1
	}
	return i + len(word)
}

func (s *bodyScanner) number(i int) int {
	data := s.data
	if data[i] == '-' {
		i++
	}
	switch {
	case i < len(data) && data[i] == '0':
		i++
	case i < len(data) && data[i] >= '1' && data[i] <= '9':
		for i < len(data) && data[i] >= '0' && data[i] <= '9' {
			i++
		}
	default:
		return -1
	}
	if i < len(data) && data[i] == '.' {
		i++
		if i >= len(data) || data[i] < '0' || data[i] > '9' {
			return -1
		}
		for i < len(data) && data[i] >= '0' && data[i] <= '9' {
			i++
		}
	}
	if i < len(data) && (data[i] == 'e' || data[i] == 'E') {
		i++
		if i < len(data) && (data[i] == '+' || data[i] == '-') {
			i++
		}
		if i >= len(data) || data[i] < '0' || data[i] > '9' {
			return -1
		}
		for i < len(data) && data[i] >= '0' && data[i] <= '9' {
			i++
		}
	}
	return i
}

// isKey reports whether a quoted, already validated key decodes to name.
func isKey(quoted []byte, name string) bool {
	if bytes.IndexByte(quoted, '\\') < 0 {
		return len(quoted) == len(name)+2 && string(quoted[1:len(quoted)-1]) == name
	}
	var key string
	return json.Unmarshal(quoted, &key) == nil && key == name
}

// hasControl reports whether b holds a byte below 0x20, which a JSON string must escape.
func hasControl(b []byte) bool {
	const ones, highs = 0x0101010101010101, 0x8080808080808080
	for len(b) >= 8 {
		x := binary.LittleEndian.Uint64(b)
		if (x-0x20*ones)&^x&highs != 0 {
			return true
		}
		b = b[8:]
	}
	for _, c := range b {
		if c < 0x20 {
			return true
		}
	}
	return false
}

func isHex(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
}
