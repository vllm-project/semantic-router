package systemone

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
)

// maxNesting is encoding/json's nesting limit: a deeper document is invalid there too.
const maxNesting = 10000

// requestModel reads a native request body as json.Unmarshal reads it into a
// map[string]json.RawMessage, without decoding the media it carries. ok is
// false unless the body is one JSON object with the syntax json.Valid accepts;
// model is the last top-level "model" member decoded as a string, and empty when
// that member is absent, null or not a string. Strings are scanned a word at a
// time, so megabytes of base64 images or videos cost a memory scan instead of
// encoding/json's byte-at-a-time state machine.
func requestModel(body []byte) (model string, ok bool) {
	s := bodyScanner{data: body}
	start := s.space(0)
	if start >= len(body) || body[start] != '{' {
		return "", false
	}
	end := s.object(start)
	if end < 0 || s.space(end) != len(body) {
		return "", false
	}
	if s.model != nil && json.Unmarshal(s.model, &model) != nil {
		model = ""
	}
	return model, true
}

type bodyScanner struct {
	data  []byte
	depth int
	// model is the value of the last top-level "model" member.
	model []byte
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
		if top && isModelKey(key) {
			s.model = s.data[valueStart:valueEnd]
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

// isModelKey reports whether a quoted, already validated key decodes to "model".
func isModelKey(quoted []byte) bool {
	if bytes.IndexByte(quoted, '\\') < 0 {
		return string(quoted) == `"model"`
	}
	var key string
	return json.Unmarshal(quoted, &key) == nil && key == "model"
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
