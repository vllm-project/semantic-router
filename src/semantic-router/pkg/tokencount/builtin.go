package tokencount

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"strconv"
	"sync"

	tiktokenassets "github.com/pkoukk/tiktoken-go-loader/assets"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount/tokenizerspec"
)

// Built-in encodings: OpenAI's tiktoken vocabularies, embedded in the binary
// so they work offline.
const (
	O200kBase  = tokenizerspec.O200kBase
	Cl100kBase = tokenizerspec.Cl100kBase
)

// The split patterns of the built-in encodings, as tiktoken writes them.
var builtinPatterns = map[string]string{O200kBase: o200kPattern, Cl100kBase: cl100kPattern}

type builtinEntry struct {
	once sync.Once
	enc  *Encoding
	err  error
}

var builtins = map[string]*builtinEntry{O200kBase: {}, Cl100kBase: {}}

// Builtin returns a built-in encoding, read from the binary on first use.
func Builtin(name string) (*Encoding, error) {
	entry, ok := builtins[name]
	if !ok {
		return nil, fmt.Errorf("%q is not a built-in encoding (%s, %s)", name, O200kBase, Cl100kBase)
	}
	entry.once.Do(func() { entry.enc, entry.err = loadBuiltin(name) })
	return entry.enc, entry.err
}

func loadBuiltin(name string) (*Encoding, error) {
	data, err := tiktokenassets.Assets.ReadFile(name + ".tiktoken")
	if err != nil {
		return nil, err
	}
	ranks := make(map[string]int, bytes.Count(data, []byte{'\n'})+1)
	for len(data) > 0 {
		line := data
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			line, data = data[:i], data[i+1:]
		} else {
			data = nil
		}
		if token, rank, ok, lineErr := parseRankLine(line); lineErr != nil {
			return nil, fmt.Errorf("%s: %w", name, lineErr)
		} else if ok {
			ranks[token] = rank
		}
	}
	split, err := newSplitter(builtinPatterns[name])
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return &Encoding{name: name, ranks: ranks, splits: []*splitter{split}}, nil
}

// parseRankLine reads a "base64-token rank" line of a .tiktoken file; ok is
// false for a line without one.
func parseRankLine(line []byte) (string, int, bool, error) {
	space := bytes.IndexByte(line, ' ')
	if space < 0 {
		return "", 0, false, nil
	}
	token, err := base64.StdEncoding.DecodeString(string(line[:space]))
	if err != nil {
		return "", 0, false, err
	}
	rank, err := strconv.Atoi(string(line[space+1:]))
	if err != nil {
		return "", 0, false, err
	}
	return string(token), rank, true, nil
}
