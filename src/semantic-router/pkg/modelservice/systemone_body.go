package modelservice

import (
	"bytes"
	"encoding/json"
)

// withServedModel returns body with its top-level "model" member set to model,
// every other byte unchanged, so a request that carries large images or videos
// is not decoded and encoded again. ok is false when body is not a JSON object
// this scan reads plainly (an escaped "model" key, a repeated one, or a
// malformed body); the caller then rewrites it with encoding/json.
func withServedModel(body []byte, model string) ([]byte, bool) {
	quoted, err := json.Marshal(model)
	if err != nil {
		return nil, false
	}
	open := skipSpace(body, 0)
	if open >= len(body) || body[open] != '{' {
		return nil, false
	}
	start, end, found := -1, -1, false
	i := skipSpace(body, open+1)
	if i < len(body) && body[i] == '}' {
		return joinBytes(body[:open+1], []byte(`"model":`), quoted, body[i:]), true
	}
	for {
		if i >= len(body) || body[i] != '"' {
			return nil, false
		}
		keyEnd, ok := skipString(body, i)
		if !ok {
			return nil, false
		}
		key := body[i:keyEnd]
		i = skipSpace(body, keyEnd)
		if i >= len(body) || body[i] != ':' {
			return nil, false
		}
		valueStart := skipSpace(body, i+1)
		valueEnd, ok := skipValue(body, valueStart)
		if !ok {
			return nil, false
		}
		if bytes.IndexByte(key, '\\') >= 0 {
			return nil, false
		}
		if string(key) == `"model"` {
			if found {
				return nil, false
			}
			start, end, found = valueStart, valueEnd, true
		}
		i = skipSpace(body, valueEnd)
		if i < len(body) && body[i] == ',' {
			i = skipSpace(body, i+1)
			continue
		}
		if i < len(body) && body[i] == '}' && skipSpace(body, i+1) == len(body) {
			break
		}
		return nil, false
	}
	if found {
		return joinBytes(body[:start], quoted, body[end:]), true
	}
	return joinBytes(body[:open+1], []byte(`"model":`), quoted, []byte(","), body[open+1:]), true
}

func joinBytes(parts ...[]byte) []byte {
	size := 0
	for _, part := range parts {
		size += len(part)
	}
	out := make([]byte, 0, size)
	for _, part := range parts {
		out = append(out, part...)
	}
	return out
}

func skipSpace(body []byte, i int) int {
	for i < len(body) {
		switch body[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// skipString returns the index after the string that starts at body[i] (a quote).
func skipString(body []byte, i int) (int, bool) {
	for j := i + 1; j < len(body); {
		k := bytes.IndexByte(body[j:], '"')
		if k < 0 {
			return 0, false
		}
		k += j
		backslashes := 0
		for p := k - 1; p > i && body[p] == '\\'; p-- {
			backslashes++
		}
		if backslashes%2 == 0 {
			return k + 1, true
		}
		j = k + 1
	}
	return 0, false
}

// skipValue returns the index after the JSON value that starts at body[i].
func skipValue(body []byte, i int) (int, bool) {
	if i >= len(body) {
		return 0, false
	}
	switch body[i] {
	case '"':
		return skipString(body, i)
	case '{', '[':
		depth := 0
		for i < len(body) {
			switch body[i] {
			case '"':
				end, ok := skipString(body, i)
				if !ok {
					return 0, false
				}
				i = end
				continue
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					return i + 1, true
				}
			}
			i++
		}
		return 0, false
	default:
		end := i
		for end < len(body) {
			switch body[end] {
			case ',', '}', ']', ' ', '\t', '\n', '\r':
				return end, end > i
			}
			end++
		}
		return end, end > i
	}
}
