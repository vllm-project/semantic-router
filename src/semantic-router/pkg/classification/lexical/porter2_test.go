package lexical

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// The sample is every seventh word of the Snowball English test vocabulary
// (plus every word of up to three letters) with its reference stem.
func TestStemEnglishMatchesSnowballVocabulary(t *testing.T) {
	path := "testdata/porter2_sample.txt"
	if full := os.Getenv("SNOWBALL_ENGLISH_VOCABULARY"); full != "" {
		path = full
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	words := 0
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 {
			continue
		}
		words++
		if got := stemEnglish(fields[0]); got != fields[1] {
			t.Errorf("stemEnglish(%q) = %q, want %q", fields[0], got, fields[1])
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if words < 1000 {
		t.Fatalf("only %d vocabulary words read", words)
	}
}

func TestStemEnglishExamples(t *testing.T) {
	for word, want := range map[string]string{
		"connection": "connect", "connections": "connect", "connective": "connect",
		"connected": "connect", "connecting": "connect", "running": "run",
		"generously": "generous", "skies": "sky", "yes": "yes", "sayyid": "sayyid",
		"ab": "ab", "can't": "can't", "3.14": "3.14", "debugging": "debug",
	} {
		if got := stemEnglish(word); got != want {
			t.Errorf("stemEnglish(%q) = %q, want %q", word, got, want)
		}
	}
}
