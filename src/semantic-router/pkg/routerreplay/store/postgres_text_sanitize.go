package store

import (
	"bytes"
	"strings"
	"unicode/utf8"
)

// sanitizePostgresText makes a string safe for a Postgres text column. Postgres
// cannot store NUL bytes and rejects invalid UTF-8 byte sequences, so a replay
// body carrying either (for example PDF-derived text) fails at insert time and
// the record is lost. NUL bytes are dropped and invalid UTF-8 is replaced with
// the Unicode replacement character; the remaining content is preserved.
func sanitizePostgresText(s string) string {
	if s == "" {
		return s
	}
	if !strings.ContainsRune(s, 0) && utf8.ValidString(s) {
		return s
	}
	return strings.ToValidUTF8(strings.ReplaceAll(s, "\x00", ""), "\uFFFD")
}

// sanitizePostgresJSON removes NUL escape sequences from marshaled JSON before
// it is bound to a Postgres jsonb column, which rejects \u0000 with
// "unsupported Unicode escape sequence". json.Marshal already replaces invalid
// UTF-8 with the replacement character, so \u0000 is the only jsonb-hostile
// artifact left.
//
// Only a \u0000 that is itself a JSON escape (an odd number of backslashes
// precedes it) encodes an actual NUL byte and is removed. A literal string
// value containing the text "\u0000" marshals with an escaped backslash
// (\\u0000, an even run), so the match falls on an escaped backslash and must
// be preserved — stripping it there would leave a dangling backslash and
// corrupt the JSON.
func sanitizePostgresJSON(b []byte) []byte {
	if len(b) == 0 {
		return b
	}
	nul := []byte(`\u0000`)
	if !bytes.Contains(b, nul) {
		return b
	}
	out := make([]byte, 0, len(b))
	for i := 0; i < len(b); {
		if bytes.HasPrefix(b[i:], nul) {
			// Count the backslashes already emitted directly before this one.
			backslashes := 0
			for j := len(out) - 1; j >= 0 && out[j] == '\\'; j-- {
				backslashes++
			}
			if backslashes%2 == 0 {
				// Real escaped NUL: drop the whole \u0000 sequence.
				i += len(nul)
				continue
			}
		}
		out = append(out, b[i])
		i++
	}
	return out
}
