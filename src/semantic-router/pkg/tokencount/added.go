package tokencount

import "unicode/utf8"

// WordJoiner is the character EscapeAddedTokens puts after the first
// character of an added token's text: U+2060 WORD JOINER has no width and no
// meaning, so the text still reads the same, and it is in no added token, so
// a tokenizer can no longer match the token there.
const WordJoiner = "\u2060"

// addedTokens indexes the text of a tokenizer's added tokens, special or
// not, by their bytes: a tokenizer that matches added tokens in text, as a
// model server does when it tokenizes a rendered chat template, reads that
// text as the token wherever it occurs. A token of one character is left
// out: no character can go inside it, and such a token is ordinary text.
type addedTokens struct {
	first   [256]*addedTokenNode
	longest int
}

type addedTokenNode struct {
	next  map[byte]*addedTokenNode
	token bool
}

func newAddedTokens(tokens []string) *addedTokens {
	index := &addedTokens{}
	for _, token := range tokens {
		if utf8.RuneCountInString(token) < 2 {
			continue
		}
		node := index.first[token[0]]
		if node == nil {
			node = &addedTokenNode{}
			index.first[token[0]] = node
		}
		for i := 1; i < len(token); i++ {
			child := node.next[token[i]]
			if child == nil {
				if node.next == nil {
					node.next = map[byte]*addedTokenNode{}
				}
				child = &addedTokenNode{}
				node.next[token[i]] = child
			}
			node = child
		}
		node.token = true
		index.longest = max(index.longest, len(token))
	}
	if index.longest == 0 {
		return nil
	}
	return index
}

// walk follows text[start:] through the index as far as it goes, and
// reports whether an added token's text ends on the way and whether the
// whole rest of text was followed.
func (a *addedTokens) walk(text string, start int) (token, whole bool) {
	node := a.first[text[start]]
	for i := start + 1; node != nil; i++ {
		if node.token {
			return true, false
		}
		if i == len(text) {
			return false, len(node.next) > 0
		}
		node = node.next[text[i]]
	}
	return false, false
}

// AddedTokens reports whether the encoding knows added tokens to escape:
// those of a Hugging Face tokenizer.json, never a built-in vocabulary's.
func (e *Encoding) AddedTokens() bool {
	return e != nil && e.added != nil
}

// EscapeAddedTokens returns text with WordJoiner after the first character
// of every added token's text in it, and the byte offsets in text where it
// put one; none when text holds no added token. A token that starts inside
// another is escaped too, so the result holds no added token's text.
func (e *Encoding) EscapeAddedTokens(text string) (string, []int) {
	if !e.AddedTokens() {
		return text, nil
	}
	var at []int
	for i := 0; i < len(text); {
		_, width := utf8.DecodeRuneInString(text[i:])
		if token, _ := e.added.walk(text, i); token {
			at = append(at, i+width)
		}
		i += width
	}
	if len(at) == 0 {
		return text, nil
	}
	escaped := make([]byte, 0, len(text)+len(at)*len(WordJoiner))
	from := 0
	for _, offset := range at {
		escaped = append(escaped, text[from:offset]...)
		escaped = append(escaped, WordJoiner...)
		from = offset
	}
	return string(append(escaped, text[from:]...)), at
}

// OpensAddedToken reports whether text ends with the beginning of an added
// token's text, which text written right after it could complete.
func (e *Encoding) OpensAddedToken(text string) bool {
	if !e.AddedTokens() {
		return false
	}
	for start := max(0, len(text)-e.added.longest+1); start < len(text); start++ {
		if !utf8.RuneStart(text[start]) {
			continue
		}
		if _, whole := e.added.walk(text, start); whole {
			return true
		}
	}
	return false
}
