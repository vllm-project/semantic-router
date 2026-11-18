package tokencount

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// gpt2Pattern is the split ByteLevel applies when its use_regex is on.
const gpt2Pattern = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

type hfTokenizer struct {
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Decoder      json.RawMessage `json:"decoder"`
	Model        struct {
		Type         string          `json:"type"`
		Vocab        map[string]int  `json:"vocab"`
		Merges       json.RawMessage `json:"merges"`
		IgnoreMerges bool            `json:"ignore_merges"`
	} `json:"model"`
}

type hfComponent struct {
	Type    string `json:"type"`
	Pattern struct {
		Regex  string `json:"Regex"`
		String string `json:"String"`
	} `json:"pattern"`
	Behavior         string            `json:"behavior"`
	Invert           bool              `json:"invert"`
	UseRegex         *bool             `json:"use_regex"`
	IndividualDigits bool              `json:"individual_digits"`
	PreTokenizers    []json.RawMessage `json:"pretokenizers"`
	Normalizers      []json.RawMessage `json:"normalizers"`
}

// ParseHFTokenizer reads a Hugging Face tokenizer.json whose model is
// byte-level BPE: GPT-2's byte alphabet, merges, and Split, Digits or
// ByteLevel pre-tokenizers. Other tokenizers (SentencePiece's Metaspace,
// WordPiece, Unigram) are refused.
func ParseHFTokenizer(name string, data []byte) (*Encoding, error) {
	var tok hfTokenizer
	if err := json.Unmarshal(data, &tok); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if tok.Model.Type != "BPE" {
		return nil, fmt.Errorf("%s: the model is %q, and only byte-level BPE is counted", name, tok.Model.Type)
	}
	enc := &Encoding{name: name}
	byteLevel, err := enc.addPreTokenizer(tok.PreTokenizer)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if !byteLevel && !isComponent(tok.Decoder, "ByteLevel") {
		return nil, fmt.Errorf("%s: the tokenizer is not byte-level", name)
	}
	enc.nfc = normalizesNFC(tok.Normalizer)
	merges, err := parseMerges(tok.Model.Merges)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	enc.ranks = make(map[string]int, len(merges)+len(tok.Model.Vocab)/8)
	for i, merge := range merges {
		merged, ok := byteLevelBytes(merge[0] + merge[1])
		if !ok {
			continue
		}
		if _, seen := enc.ranks[merged]; !seen {
			enc.ranks[merged] = i
		}
	}
	if tok.Model.IgnoreMerges {
		for token, id := range tok.Model.Vocab {
			if raw, ok := byteLevelBytes(token); ok {
				if _, seen := enc.ranks[raw]; !seen {
					enc.ranks[raw] = len(merges) + id
				}
			}
		}
	}
	return enc, nil
}

// addPreTokenizer appends the splits of a pre-tokenizer and reports whether
// it maps text to GPT-2's byte alphabet.
func (e *Encoding) addPreTokenizer(raw json.RawMessage) (bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return false, nil
	}
	var c hfComponent
	if err := json.Unmarshal(raw, &c); err != nil {
		return false, err
	}
	switch c.Type {
	case "Sequence":
		byteLevel := false
		for _, child := range c.PreTokenizers {
			isByteLevel, err := e.addPreTokenizer(child)
			if err != nil {
				return false, err
			}
			byteLevel = byteLevel || isByteLevel
		}
		return byteLevel, nil
	case "Split":
		// Isolated keeps matches and the text between them as pieces;
		// Removed with invert keeps only the matches.
		keepMatches := c.Behavior == "Removed" && c.Invert
		if !keepMatches && (c.Behavior != "Isolated" || c.Invert) {
			return false, fmt.Errorf("a Split with behavior %q (invert %t) is not supported", c.Behavior, c.Invert)
		}
		pattern := c.Pattern.Regex
		if pattern == "" {
			pattern = regexp.QuoteMeta(c.Pattern.String)
		}
		return false, e.addSplit(pattern, keepMatches)
	case "Digits":
		if c.IndividualDigits {
			return false, e.addSplit(`\p{N}`, false)
		}
		return false, e.addSplit(`\p{N}+`, false)
	case "ByteLevel":
		if c.UseRegex == nil || *c.UseRegex {
			return true, e.addSplit(gpt2Pattern, false)
		}
		return true, nil
	default:
		return false, fmt.Errorf("pre-tokenizer %q is not supported", c.Type)
	}
}

func (e *Encoding) addSplit(pattern string, dropGaps bool) error {
	split, err := newSplitter(pattern)
	if err != nil {
		return fmt.Errorf("split pattern %q: %w", pattern, err)
	}
	split.dropGaps = dropGaps
	e.splits = append(e.splits, split)
	return nil
}

func isComponent(raw json.RawMessage, kind string) bool {
	var c hfComponent
	return len(raw) > 0 && json.Unmarshal(raw, &c) == nil && c.Type == kind
}

// normalizesNFC reports a normalizer that composes text, NFKC included: the
// compatibility mappings it adds change few prompts.
func normalizesNFC(raw json.RawMessage) bool {
	var c hfComponent
	if len(raw) == 0 || json.Unmarshal(raw, &c) != nil {
		return false
	}
	if c.Type == "NFC" || c.Type == "NFKC" {
		return true
	}
	for _, child := range c.Normalizers {
		if normalizesNFC(child) {
			return true
		}
	}
	return false
}

// parseMerges reads merges written either as "a b" or as ["a", "b"].
func parseMerges(raw json.RawMessage) ([][2]string, error) {
	var pairs [][2]string
	if err := json.Unmarshal(raw, &pairs); err == nil {
		return pairs, nil
	}
	var lines []string
	if err := json.Unmarshal(raw, &lines); err != nil {
		return nil, fmt.Errorf("merges: %w", err)
	}
	pairs = make([][2]string, 0, len(lines))
	for _, line := range lines {
		a, b, ok := strings.Cut(line, " ")
		if !ok {
			return nil, fmt.Errorf("merge %q is not a pair", line)
		}
		pairs = append(pairs, [2]string{a, b})
	}
	return pairs, nil
}

// runeToByte inverts GPT-2's map of bytes to printable characters, the
// alphabet byte-level vocabularies are written in.
var runeToByte = byteLevelAlphabet()

func byteLevelAlphabet() map[rune]byte {
	inverse := make(map[rune]byte, 256)
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	next := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !printable(b) {
			r = rune(256 + next)
			next++
		}
		inverse[r] = byte(b)
	}
	return inverse
}

// byteLevelBytes decodes a vocabulary entry to the bytes it stands for.
func byteLevelBytes(token string) (string, bool) {
	out := make([]byte, 0, len(token))
	for _, r := range token {
		b, ok := runeToByte[r]
		if !ok {
			return "", false
		}
		out = append(out, b)
	}
	return string(out), true
}
