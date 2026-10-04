package topiccontinuity

import (
	"strings"
)

// segmentEntities extracts entities from original-case text in bounded linear
// work: one pass per delimiter kind, then one pass over word runs. It finds
// the v1 entity classes:
//
//  1. backtick spans `...` (no newline inside, any length);
//  2. double-quoted spans "..." (no newline inside, at least 3 bytes);
//  3. paths: a word run containing '/' and a letter;
//  4. dotted names: identifier(.identifier)+ inside a word run;
//  5. snake_case: an underscore part containing a letter;
//  6. camelCase / PascalCase: a part with a lower-to-upper transition;
//  7. numbers: digit runs of at least 3 digits.
//
// A word run is a maximal run of [A-Za-z0-9_.~/-]. Delimited spans have no
// upper length limit: a long span is captured whole and then cut to a
// prefix, so it is never skipped. Values are lowercased, trailing ".,;:)" is
// trimmed, and each value is cut to a rune-safe maxEntityBytes prefix, which
// can only create matches, never hide one.
//
// The scan stops as soon as maxEntitiesPerSegment distinct values are found;
// reaching the cap forces partial coverage because a dropped entity could be
// the only overlap.
func segmentEntities(segment textSegment) (map[string]struct{}, bool) {
	collector := entityCollector{out: make(map[string]struct{})}
	text := string(segment)
	collector.delimited(text, '`', 1)
	collector.delimited(text, '"', 3)
	for start := 0; start < len(text) && !collector.capped; {
		if !isWordByte(text[start]) {
			start++
			continue
		}
		end := start
		for end < len(text) && isWordByte(text[end]) {
			end++
		}
		collector.word(text[start:end])
		start = end
	}
	return collector.out, collector.capped
}

type entityCollector struct {
	out    map[string]struct{}
	capped bool
}

func (c *entityCollector) add(value string) {
	if c.capped {
		return
	}
	value = strings.TrimRight(strings.ToLower(value), ".,;:)")
	value = runePrefix(value, maxEntityBytes)
	if value == "" {
		return
	}
	c.out[value] = struct{}{}
	if len(c.out) >= maxEntitiesPerSegment {
		c.capped = true
	}
}

func (c *entityCollector) delimited(text string, delimiter byte, minimum int) {
	for offset := 0; offset < len(text) && !c.capped; {
		open := strings.IndexByte(text[offset:], delimiter)
		if open < 0 {
			return
		}
		open += offset
		closeAt := -1
		for i := open + 1; i < len(text); i++ {
			if text[i] == '\n' {
				break
			}
			if text[i] == delimiter {
				closeAt = i
				break
			}
		}
		if closeAt < 0 {
			offset = open + 1
			continue
		}
		if closeAt-open-1 >= minimum {
			c.add(text[open+1 : closeAt])
		}
		offset = closeAt + 1
	}
}

func isWordByte(b byte) bool {
	return isAlnum(b) || b == '_' || b == '.' || b == '~' || b == '/' || b == '-'
}

func isAlnum(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}

func isLetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func hasLetter(text string) bool {
	for i := 0; i < len(text); i++ {
		if isLetter(text[i]) {
			return true
		}
	}
	return false
}

// word classifies one word run and its parts. Runs are ASCII by
// construction, so parts are found by byte index without allocation.
func (c *entityCollector) word(run string) {
	if strings.IndexByte(run, '/') >= 0 && hasLetter(run) {
		c.add(run)
	}
	forEachPart(run, isChunkSeparator, func(chunk string) {
		if dotted(chunk) {
			c.add(chunk)
		}
		forEachPart(chunk, func(b byte) bool { return b == '.' }, func(part string) {
			if strings.IndexByte(part, '_') >= 0 && hasLetter(part) {
				c.add(part)
			}
			if camel(part) {
				c.add(part)
			}
			c.numbers(part)
		})
	})
}

func isChunkSeparator(b byte) bool {
	return b == '/' || b == '~' || b == '-'
}

func forEachPart(text string, separator func(byte) bool, fn func(string)) {
	start := 0
	for i := 0; i <= len(text); i++ {
		if i == len(text) || separator(text[i]) {
			if i > start {
				fn(text[start:i])
			}
			start = i + 1
		}
	}
}

// dotted reports identifier(.identifier)+ where each identifier starts with a
// letter or underscore.
func dotted(chunk string) bool {
	chunk = strings.TrimRight(chunk, ".")
	parts, start := 0, 0
	for i := 0; i <= len(chunk); i++ {
		if i < len(chunk) && chunk[i] != '.' {
			continue
		}
		if i == start || (!isLetter(chunk[start]) && chunk[start] != '_') {
			return false
		}
		parts++
		start = i + 1
	}
	return parts >= 2
}

// camel reports a part with a lowercase-to-uppercase transition (camelCase,
// or PascalCase with an inner capital).
func camel(part string) bool {
	for i := 1; i < len(part); i++ {
		if part[i] >= 'A' && part[i] <= 'Z' && part[i-1] >= 'a' && part[i-1] <= 'z' {
			return true
		}
	}
	return false
}

func (c *entityCollector) numbers(part string) {
	for i := 0; i < len(part); {
		if part[i] < '0' || part[i] > '9' {
			i++
			continue
		}
		j := i
		for j < len(part) && part[j] >= '0' && part[j] <= '9' {
			j++
		}
		if j-i >= 3 {
			c.add(part[i:j])
		}
		i = j
	}
}
