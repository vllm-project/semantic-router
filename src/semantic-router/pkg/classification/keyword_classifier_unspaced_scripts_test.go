package classification

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Thai, Lao, Khmer and Myanmar are written without spaces between words, like
// CJK. A literal keyword in these scripts must match inside running text, and a
// Latin keyword must match next to them, the same way "C++" matches in
// "解释C++语言". A match must not end before a combining mark ("ไม" inside
// "ไม่"), start after a Thai/Lao preposed vowel ("ขา" inside "เขา"), or start
// after a Khmer coeng or Myanmar virama, which attach to the next consonant.
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

// Each edge of a match is checked by the keyword's rune at that edge. A Thai
// or Lao edge must not end before a following vowel that is a letter, not a
// combining mark ("เก" inside "เกาหลี"). A Latin or digit edge of a mixed
// keyword stays a word edge, as it is for a keyword without these scripts.
var unspacedScriptKeywordEdgeCases = []struct {
	word, text string
	want       bool
}{
	// The match ends before a Thai or Lao following vowel.
	{"เก", "เกาหลี", false},
	{"จ", "จะไป", false},
	{"ກ", "ກາ", false},
	{"ไม", "ไม้กวาด", false},
	// A Khmer keyword ending in a vowel sign ends before a visarga (Mc).
	{"ទេ", "ទេះ", false},
	// A keyword ending in a tone mark matches inside running text.
	{"ไม่", "ไม่ได้", true},
	// The match starts after a Khmer coeng.
	{"ញុំ", "ខ្ញុំ", false},
	// A Latin or digit edge of a mixed keyword is a word edge.
	{"รุ่น 15", "ไอโฟนรุ่น 15 ราคา", true},
	{"รุ่น 15", "รุ่น 150", false},
	{"ราคา GPU", "อยากรู้ราคา GPU", true},
	{"ราคา GPU", "ราคา GPUเท่าไหร่", true},
	{"ราคา GPU", "ราคา GPUs", false},
	{"ราคา GPU", "ราคา GPU_x", false},
	{"ราคาiphone", "ราคาiphones", false},
	{"AI ไทย", "โมเดลAI ไทยดีไหม", true},
	{"AI ไทย", "OpenAI ไทย", false},
	{"AI ไทย", "ใช้ AI ไทยดีไหม", true},
	// A Thai or Lao edge does not join an underscore, as a CJK edge does not.
	{"ราคา", "price_ราคา", true},
	// CJK edges keep their substring behavior.
	{"价格", "iPhone价格多少", true},
}

func TestKeywordClassifierUnspacedScriptEdges(t *testing.T) {
	for _, tc := range unspacedScriptKeywordEdgeCases {
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

// count and density rules find the same occurrences as matching.
func TestKeywordCountUnspacedScriptEdges(t *testing.T) {
	for _, tc := range []struct {
		word, text string
		count      float64
	}{
		{"ราคา", "ราคาเท่าไหร่ อยากทราบราคา ราคาถูก", 3},
		{"เก", "เกาหลี เก", 1},
		{"รุ่น 15", "รุ่น 150 รุ่น 15", 1},
		{"must", "ต้องmustเลย must๓", 1},
	} {
		t.Run(tc.word+"/"+tc.text, func(t *testing.T) {
			classifier := countingKeywordClassifier(t,
				config.KeywordRule{Name: "count", Operator: "OR", Keywords: []string{tc.word}, Count: &config.NumericPredicate{GTE: bound(0)}},
				config.KeywordRule{Name: "density", Operator: "OR", Keywords: []string{tc.word}, Density: &config.NumericPredicate{GTE: bound(0)}},
			)
			_, values, err := classifier.MatchAllWithHistory(tc.text, "")
			density := 1000 * tc.count / float64(len([]rune(tc.text)))
			if err != nil || values["count"] != tc.count || values["density"] != density {
				t.Fatalf("count = %v, density = %v (%v), want %v and %v", values["count"], values["density"], err, tc.count, density)
			}
		})
	}
}

// Masking finds the same places as matching.
func TestKeywordRuleSpansUnspacedScriptEdges(t *testing.T) {
	for _, tc := range []struct {
		word, text string
		want       []string
	}{
		{"ราคา", "อยากทราบราคาสินค้า", []string{"ราคา"}},
		{"ไม", "ไม่ไม", []string{"ไม"}},
		{"รุ่น 15", "รุ่น 150 รุ่น 15", []string{"รุ่น 15"}},
	} {
		t.Run(tc.word+"/"+tc.text, func(t *testing.T) {
			c, err := NewKeywordClassifier([]config.KeywordRule{{Name: "found", Operator: "OR", Keywords: []string{tc.word}}})
			if err != nil {
				t.Fatal(err)
			}
			spans, err := c.RuleSpans("found", tc.text)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			runes := []rune(tc.text)
			for _, span := range spans {
				if string(runes[span.Start:span.End]) != span.Text {
					t.Fatalf("span %+v does not cover its text in code points of %q", span, tc.text)
				}
				got = append(got, span.Text)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("spans = %q, want %q", got, tc.want)
			}
		})
	}
}
