package tokencount

import (
	"context"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var referenceTexts = []string{
	"Hello, world!",
	"The quick brown fox jumps over the lazy dog.",
	"  indented:\n\tcode();\n\n\n    return x  \n",
	"I'm sure they're here, we've seen it; you'll know. HE'S GONE",
	"HTTPServer URLs and camelCaseWords, ALLCAPS, MixedCASEtext",
	"Numbers: 1234567890, 3.14159, -42, 1e10, ٣٤٥ and ²",
	"中文的文本，包括标点符号。日本語のテキスト、カタカナ。",
	"Emoji 😀 👩‍💻 🇨🇳 and accents: café naïve Ærøskøbing straße",
	"path/to/file.go:42 // comment\r\nnext line\u00a0nbsp\u3000ideographic",
	`{"key": [1, 2, 3], "nested": {"a": null}}`,
}

// TestBuiltinCountsMatchTiktoken holds the counts of Python's tiktoken 0.12.0.
func TestBuiltinCountsMatchTiktoken(t *testing.T) {
	want := map[string][]int{
		O200kBase:  {4, 10, 10, 16, 15, 29, 21, 25, 16, 20},
		Cl100kBase: {4, 10, 10, 20, 15, 32, 28, 30, 15, 20},
	}
	for name, counts := range want {
		enc, err := Builtin(name)
		if err != nil {
			t.Fatal(err)
		}
		for i, text := range referenceTexts {
			if got := enc.Count(text); got != counts[i] {
				t.Errorf("%s: Count(%q) = %d, tiktoken counts %d", name, text, got, counts[i])
			}
		}
	}
}

// mixedText draws from ASCII, the Unicode classes the patterns tell apart
// and the contractions, so that matchers meet every boundary.
func mixedText(rng *rand.Rand, n int) string {
	pieces := []string{
		"a", "z", "Q", "e", "s", "S", "l", "L", "r", "v", "d", "m", "t",
		"0", "7", "123", " ", "  ", "\t", "\n", "\r\n", "\v", "\f",
		"'", "'s", "'S", "'re", "'LL", "'ve", "'d", "'M", "'t", "'ſ",
		".", ",", "/", "(", ")", "-", "_", "\"", "{", "}", "$", "~", "\x00", "\x1f", "\x7f",
		"é", "É", "ß", "ſ", "ǅ", "ʰ", "中", "文", "ひ", "カ", "한", "\u0301", "²", "٣",
		"\u00a0", "\u3000", "\u2028", "\u0085", "“", "…", "€", "😀", "\u200d",
	}
	var b strings.Builder
	for b.Len() < n {
		b.WriteString(pieces[rng.Intn(len(pieces))])
	}
	return b.String()
}

func TestFastMatchersAgreeWithTheirPatterns(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for pattern := range fastMatchers {
		fast, err := newSplitter(pattern)
		if err != nil {
			t.Fatal(err)
		}
		slow := *fast
		slow.fast = nil
		for range 400 {
			text := mixedText(rng, 200)
			for pos := 0; pos < len(text); pos++ {
				if got, want := fast.matchAt(text, pos), slow.matchAt(text, pos); got != want {
					t.Fatalf("pattern %q at %d of %q: fast match ends at %d, the pattern's at %d", pattern, pos, text, got, want)
				}
			}
		}
	}
}

func TestClassSplitsAgreeWithTheirPatterns(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, pattern := range []string{`\p{N}{1,3}`, `\p{N}`, `\p{N}+`, "[一-龥\u3040-ゟ゠-ヿ]+"} {
		split, err := newSplitter(pattern)
		if err != nil {
			t.Fatal(err)
		}
		if split.class == nil {
			t.Fatalf("pattern %q is not matched as a character class", pattern)
		}
		slow := *split
		slow.class = nil
		for range 400 {
			text := mixedText(rng, 200)
			for pos := 0; pos < len(text); pos++ {
				if got, want := split.matchAt(text, pos), slow.matchAt(text, pos); got != want {
					t.Fatalf("pattern %q at %d of %q: class match ends at %d, the pattern's at %d", pattern, pos, text, got, want)
				}
			}
		}
	}
}

func TestTranslatePattern(t *testing.T) {
	for pattern, want := range map[string]string{
		`\p{L}+`:          `\p{L}+`,
		`a++|b*+|c?+`:     `a+|b*|c?`,
		`\p{N}{1,3}+`:     `\p{N}{1,3}`,
		`(?i:'s)|(?:x)+`:  `(?i:'s)|(?:x)+`,
		`[{|}+]+`:         `[{|}+]+`,
		`\s+`:             `[` + whiteSpace + `]+`,
		`[^\s\p{L}]\S`:    `[^` + whiteSpace + `\p{L}][^` + whiteSpace + `]`,
		`[]a]+[^]b]`:      `[]a]+[^]b]`,
		`\\s`:             `\\s`,
		`\x{2028}+\p{Lu}`: `\x{2028}+\p{Lu}`,
	} {
		if got := translatePattern(pattern); got != want {
			t.Errorf("translatePattern(%q) = %q, want %q", pattern, got, want)
		}
	}
}

func TestCountInChunksMatchesOnePass(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	var b strings.Builder
	for b.Len() < 5*chunkBytes {
		b.WriteString(referenceTexts[rng.Intn(len(referenceTexts))])
		b.WriteString(" ")
		b.WriteString(mixedText(rng, 64))
		b.WriteString(" ")
	}
	text := b.String()
	if len(splitChunks(text, chunkBytes)) < 4 {
		t.Fatal("the text should span several chunks")
	}
	for _, name := range []string{O200kBase, Cl100kBase} {
		enc, err := Builtin(name)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := enc.Count(text), enc.countChunk(text); got != want {
			t.Errorf("%s: %d tokens in chunks, %d in one pass", name, got, want)
		}
	}
}

// A byte-level BPE tokenizer.json with the merges that spell "hello".
const tinyTokenizer = `{
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false},
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}
  ]},
  "decoder": {"type": "ByteLevel"},
  "model": {"type": "BPE", "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hell": 6, "hello": 7, "Ġ": 8},
    "merges": ["h e", "l l", "he ll", "hell o"]}
}`

func TestParseHFTokenizer(t *testing.T) {
	pairs := strings.NewReplacer(`"h e"`, `["h", "e"]`, `"l l"`, `["l", "l"]`, `"he ll"`, `["he", "ll"]`, `"hell o"`, `["hell", "o"]`)
	for _, source := range []string{tinyTokenizer, pairs.Replace(tinyTokenizer)} {
		enc, err := ParseHFTokenizer("tiny", []byte(source))
		if err != nil {
			t.Fatal(err)
		}
		for text, want := range map[string]int{
			"hello":       1,
			"hell":        1,
			"hello hello": 3, // "hello" and " hello", whose space has no merge
			"helo":        3, // "he", "l", "o"
			"12345":       5, // "123" and "45", one token per byte
		} {
			if got := enc.Count(text); got != want {
				t.Errorf("Count(%q) = %d, want %d", text, got, want)
			}
		}
	}

	for _, unsupported := range []string{
		`{"pre_tokenizer": {"type": "Metaspace"}, "model": {"type": "BPE", "vocab": {}, "merges": []}}`,
		`{"pre_tokenizer": null, "decoder": {"type": "ByteFallback"}, "model": {"type": "BPE", "vocab": {}, "merges": []}}`,
		`{"model": {"type": "WordPiece", "vocab": {}}}`,
		`{"pre_tokenizer": {"type": "Split", "pattern": {"Regex": "a"}, "behavior": "MergedWithNext"}, "model": {"type": "BPE", "vocab": {}, "merges": []}}`,
	} {
		if _, err := ParseHFTokenizer("unsupported", []byte(unsupported)); err == nil {
			t.Errorf("ParseHFTokenizer(%s) should fail", unsupported)
		}
	}
}

func TestRegistryLoadsEachKindOfSpec(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(file, []byte(tinyTokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
	commit := strings.Repeat("ab", 20)
	cached := filepath.Join(dir, "hub", "models--org--cached", "snapshots", commit, hubFile)
	if err := writeFileAtomic(cached, []byte(tinyTokenizer)); err != nil {
		t.Fatal(err)
	}
	if err := writeFileAtomic(filepath.Join(dir, "hub", "models--org--cached", "refs", "main"), []byte(commit)); err != nil {
		t.Fatal(err)
	}
	registry := NewRegistry(Hub{CacheDir: filepath.Join(dir, "hub"), Offline: true})

	for _, spec := range []string{O200kBase, file, dir, "org/cached", "org/cached@" + commit} {
		enc, ok := registry.Lookup(spec)
		if !ok {
			t.Fatalf("Lookup(%q) found nothing", spec)
		}
		if enc.Count("hello") < 1 {
			t.Fatalf("Lookup(%q) counts nothing", spec)
		}
	}
	for _, spec := range []string{"org/missing", filepath.Join(dir, "missing.json"), "not a spec"} {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, err := registry.Load(ctx, spec)
		cancel()
		if err == nil {
			t.Errorf("Load(%q) should fail offline", spec)
		}
		if _, ok := registry.Lookup(spec); ok {
			t.Errorf("Lookup(%q) should find nothing", spec)
		}
	}
}

func TestRegistryDownloadsIntoTheHubCache(t *testing.T) {
	commit := strings.Repeat("cd", 20)
	auths := make(chan string, 1)
	release := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/org/model/resolve/main/tokenizer.json", func(w http.ResponseWriter, r *http.Request) {
		<-release
		auths <- r.Header.Get("Authorization")
		w.Header().Set("X-Repo-Commit", commit)
		http.Redirect(w, r, "/blob/tokenizer.json", http.StatusFound)
	})
	mux.HandleFunc("/blob/tokenizer.json", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(tinyTokenizer))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	cache := t.TempDir()
	registry := NewRegistry(Hub{Endpoint: server.URL, Token: "test-token", CacheDir: cache})
	if _, ok := registry.Lookup("org/model"); ok {
		t.Fatal("a repo outside the cache should load in the background")
	}
	close(release)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := registry.Load(ctx, "org/model"); err != nil {
		t.Fatal(err)
	}
	if auth := <-auths; auth != "Bearer test-token" {
		t.Errorf("the download sent Authorization %q", auth)
	}
	if _, err := os.Stat(filepath.Join(cache, "models--org--model", "snapshots", commit, hubFile)); err != nil {
		t.Fatalf("the tokenizer is not in the cache's snapshot: %v", err)
	}
	ref, err := os.ReadFile(filepath.Join(cache, "models--org--model", "refs", "main"))
	if err != nil || string(ref) != commit {
		t.Fatalf("refs/main = %q, %v; want %s", ref, err, commit)
	}
	offline := NewRegistry(Hub{CacheDir: cache, Offline: true})
	if _, ok := offline.Lookup("org/model"); !ok {
		t.Fatal("the downloaded tokenizer should load offline")
	}
}
