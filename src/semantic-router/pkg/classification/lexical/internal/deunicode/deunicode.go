// Package deunicode transliterates Unicode text to ASCII with the tables and
// rules of the deunicode crate 1.6.2 (BSD-3-Clause, see LICENSE), which the
// BM25 tokenizer applied before matching. The tables load on first use of
// non-ASCII input.
package deunicode

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"io"
	"strings"
	"sync"
	"unicode/utf8"
)

// Tofu replaces characters the tables cannot transliterate.
const Tofu = "[?]"

var (
	//go:embed pointers.bin.gz
	pointersGz []byte
	//go:embed mapping.txt.gz
	mappingGz []byte

	loadOnce sync.Once
	pointers []byte // 3 bytes per code point: two inline chars or a mapping offset, then a length
	mapping  string
)

func load() {
	pointers = gunzip(pointersGz)
	mapping = string(gunzip(mappingGz))
}

func gunzip(data []byte) []byte {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		panic("deunicode: corrupt embedded table: " + err.Error())
	}
	out, err := io.ReadAll(reader)
	if err != nil {
		panic("deunicode: corrupt embedded table: " + err.Error())
	}
	return out
}

// lookup returns the ASCII transliteration of r, or false when it has none.
func lookup(r rune) (string, bool) {
	index := int(r) * 3
	if r < 0 || index+3 > len(pointers) {
		return "", false
	}
	entry := pointers[index : index+3]
	length := int(entry[2])
	if length <= 2 {
		return string(entry[:length]), true
	}
	offset := int(entry[0]) | int(entry[1])<<8
	if offset+length > len(mapping) {
		return "", false
	}
	return mapping[offset : offset+length], true
}

// Transliterate returns s in ASCII. Like the crate, it drops a trailing space
// of a multi-character transliteration at the end of the text or before a
// transliteration that starts with a space.
func Transliterate(s string) string {
	ascii := 0
	for ascii < len(s) && s[ascii] < 0x7f {
		ascii++
	}
	if ascii == len(s) {
		return s
	}
	loadOnce.Do(load)
	var out strings.Builder
	out.Grow(len(s) | 15)
	out.WriteString(s[:ascii])
	rest := s[ascii:]
	current, currentOK := decode(&rest)
	for {
		hasNext := rest != ""
		var next string
		var nextOK bool
		if hasNext {
			next, nextOK = decode(&rest)
		}
		if !currentOK {
			out.WriteString(Tofu)
		} else {
			trim := len(current) > 1 && current[len(current)-1] == ' ' &&
				(!hasNext || (nextOK && next != "" && next[0] == ' '))
			if trim {
				out.WriteString(current[:len(current)-1])
			} else {
				out.WriteString(current)
			}
		}
		if !hasNext {
			break
		}
		current, currentOK = next, nextOK
	}
	return out.String()
}

// decode takes the first character of *rest and returns its transliteration.
// Invalid UTF-8 decodes to the replacement character, as Rust strings cannot
// hold it.
func decode(rest *string) (string, bool) {
	r, size := utf8.DecodeRuneInString(*rest)
	*rest = (*rest)[size:]
	return lookup(r)
}
