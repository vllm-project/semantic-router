package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Thai, Lao, Khmer and Myanmar are written without spaces between words, like
// CJK. A literal keyword in these scripts must match inside running text, and a
// Latin keyword must match next to them, the same way "C++" matches in
// "解释C++语言". A match must not split a syllable: it must not end before a
// combining mark ("ไม" inside "ไม่"), start after a Thai/Lao preposed vowel
// ("ขา" inside "เขา"), or start after a Khmer coeng or Myanmar virama, which
// attach to the next consonant.
var unspacedScriptKeywordCases = []struct {
	word, text string
	want       bool
}{
	// Fail on main: running text without spaces.
	{"ราคา", "อยากทราบราคาสินค้า", true},
	{"ราคา", "ราคาเท่าไหร่", true},
	{"ราคา", "อยากทราบราคา", true},
	{"คืนเงิน", "ขอคืนเงินได้ไหม", true},
	{"ລາຄາ", "ຢາກຮູ້ລາຄາສິນຄ້າ", true},
	{"តម្លៃ", "ខ្ញុំចង់ដឹងតម្លៃទំនិញ", true},
	{"ဈေး", "ဒီပစ္စည်းဈေးဘယ်လောက်လဲ", true},
	// Fail on main: a Latin keyword next to Thai letters.
	{"iphone", "ราคาiphoneเท่าไหร่", true},
	// Fail on main: the keyword ends before a tone mark of the same cluster.
	{"ไม", "ไม่", false},
	// Controls: pass on main and must keep passing. "ขา" ("leg") must not
	// match inside "เขา" ("he/she"): the preposed vowel เ belongs to ข.
	{"ขา", "เขาไปไหน", false},
	// "စည်း" must not match inside the stacked cluster of "ပစ္စည်း" (one word).
	{"စည်း", "ဒီပစ္စည်းဈေး", false},
	{"ราคา", "ราคา สินค้า", true},
	{"ราคา", "อยากรู้ราคา", true},
	{"价格", "我想知道价格多少", true},
	{"cat", "concatenate", false},
	{"python", "python3", false},
}

func TestKeywordClassifierUnspacedScripts(t *testing.T) {
	for _, tc := range unspacedScriptKeywordCases {
		t.Run(tc.word+"/"+tc.text, func(t *testing.T) {
			c, err := NewKeywordClassifier([]config.KeywordRule{{Name: "selected", Operator: "OR", Keywords: []string{tc.word}}})
			if err != nil {
				t.Fatal(err)
			}
			name, confidence, err := c.Classify(tc.text)
			if err != nil || (name == "selected") != tc.want {
				t.Fatalf("word=%q input=%q route=%q confidence=%v error=%v want=%t", tc.word, tc.text, name, confidence, err, tc.want)
			}
		})
	}
}

// Explicit regex stays a plain substring match (keyword_classifier_regex.go:57),
// so it is today's workaround. This passes on main and must keep passing.
func TestKeywordClassifierUnspacedScriptsRegexWorkaround(t *testing.T) {
	c, err := NewKeywordClassifier([]config.KeywordRule{{Name: "selected", Operator: "OR", Method: "regex", Keywords: []string{"ราคา"}}})
	if err != nil {
		t.Fatal(err)
	}
	if name, _, err := c.Classify("อยากทราบราคาสินค้า"); err != nil || name != "selected" {
		t.Fatalf("route=%q error=%v, want selected", name, err)
	}
}

// The script exemption covers letters, not digits: a digit or identifier
// keyword keeps its word boundaries, so it does not match inside a longer
// number or identifier. Thai, Lao, Khmer and Myanmar digits behave the same.
func TestKeywordClassifierUnspacedScriptDigitBoundaries(t *testing.T) {
	for _, tc := range []struct {
		word, text string
		want       bool
	}{
		{"๓", "๑๓๕", false},
		{"๓", "๓", true},
		{"code๓", "prefixcode๓suffix", false},
		{"code๓", "code๓", true},
		{"໓", "໑໓໕", false}, // Lao
		{"៣", "១៣៥", false}, // Khmer
		{"၃", "၁၃၅", false}, // Myanmar
	} {
		t.Run(tc.word+"/"+tc.text, func(t *testing.T) {
			c, err := NewKeywordClassifier([]config.KeywordRule{{Name: "selected", Operator: "OR", Keywords: []string{tc.word}}})
			if err != nil {
				t.Fatal(err)
			}
			name, confidence, err := c.Classify(tc.text)
			if err != nil || (name == "selected") != tc.want {
				t.Fatalf("word=%q input=%q route=%q confidence=%v error=%v want=%t", tc.word, tc.text, name, confidence, err, tc.want)
			}
		})
	}
}

// Counting shares the matcher, so a digit keyword counts only where it stands
// as a word of its own, not inside a longer number.
func TestKeywordCountUnspacedScriptDigitBoundaries(t *testing.T) {
	classifier := countingKeywordClassifier(t, config.KeywordRule{
		Name: "digits", Operator: "OR", Keywords: []string{"๓"}, Count: &config.NumericPredicate{GTE: bound(0)},
	})
	for _, tc := range []struct {
		text string
		want float64
	}{
		{"๑๓๕", 0},
		{"๓", 1},
		{"๓ ๓", 2},
	} {
		t.Run(tc.text, func(t *testing.T) {
			_, values, err := classifier.MatchAllWithHistory(tc.text, "")
			if err != nil || values["digits"] != tc.want {
				t.Fatalf("count(%q) = %v (%v), want %v", tc.text, values["digits"], err, tc.want)
			}
		})
	}
}
