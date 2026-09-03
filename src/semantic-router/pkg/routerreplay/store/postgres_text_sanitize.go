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
func sanitizePostgresJSON(b []byte) []byte {
	if len(b) == 0 {
		return b
	}
	nul := []byte(`\u0000`)
	if !bytes.Contains(b, nul) {
		return b
	}
	return bytes.ReplaceAll(b, nul, nil)
}
