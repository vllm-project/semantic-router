package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestKeywordClassifierLiteralBoundaries(t *testing.T) {
	for _, tc := range []struct {
		word, text string
		noMatch    bool
	}{
		{"python", "I like python code", false},
		{"数学", "我喜欢数学", false},
		{"C++", "Explain C++ move semantics", false},
		{"C#", "Convert this C# code", false},
		{".NET", "Is .NET 8 faster?", false},
		{"Answer:", "Answer: 42", false},
		{"привет", "скажи привет всем", false},
		{"café", "a café nearby", false},
		{"cat", "concatenate", true},
		{"café", "caféteria", true},
		{"привет", "приветствие", true},
		{"python", "python3", true},
	} {
		t.Run(tc.word+"/"+tc.text, func(t *testing.T) {
			c, err := NewKeywordClassifier([]config.KeywordRule{{Name: "selected", Operator: "OR", Keywords: []string{tc.word}}})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Free()
			name, confidence, err := c.Classify(tc.text)
			t.Logf("word=%q input=%q route=%q confidence=%v error=%v", tc.word, tc.text, name, confidence, err)
			want := "selected"
			if tc.noMatch {
				want = ""
			}
			if err != nil || name != want {
				t.Errorf("supported literal keyword must select route, got %q err=%v", name, err)
			}
		})
	}
}

// Exercise the public classifier rather than only its compiled pattern. Literal
// matching must inspect both neighboring runes, including at a symbol edge.
func TestKeywordClassifierLiteralNeighborContract(t *testing.T) {
	for _, tc := range []struct {
		word, text string
		want       bool
	}{
		{"C++", "C++Builder", false},
		{"C#", "C#script", false},
		{".NET", "ASP.NET", false},
		{".NET", ".NETCore", false},
		{"Answer:", "Answer:42", false},
		{"A)", "A)text", false},
		{"A)", "A) answer", true},
		{"привет", "скажи привет!", true},
		{"안녕하세요", "안녕하세요 여러분", true},
		{"こんにちは", "こんにちは 世界", true},
		{"café", "a café nearby", true},
		{"café", "caféteria", false},
		{"C++", "解释C++语言", true},
		{"数学", "我喜欢数学课", true},
		{"C++", "C++Builder then C++", true},
		{".NET", "ASP.NET then .NET", true},
		{"C++", "c++", true},
		{",,", "A,,, ", true},
	} {
		t.Run(tc.word+"/"+tc.text, func(t *testing.T) {
			c, err := NewKeywordClassifier([]config.KeywordRule{{Name: "selected", Operator: "OR", Keywords: []string{tc.word}}})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Free()
			name, confidence, err := c.Classify(tc.text)
			if err != nil || (name == "selected") != tc.want {
				t.Fatalf("word=%q input=%q route=%q confidence=%v error=%v want=%t", tc.word, tc.text, name, confidence, err, tc.want)
			}
			if tc.want && confidence != 1 {
				t.Fatalf("confidence=%v, want1", confidence)
			}
		})
	}
}

func TestKeywordClassifierLiteralOperatorsAndRegexControl(t *testing.T) {
	for _, tc := range []struct {
		name, operator, method, text string
		keywords                     []string
		caseSensitive                bool
		want                         bool
		confidence                   float64
	}{
		{"AND all", "AND", "", "C++ and .NET", []string{"C++", ".NET"}, false, true, 1},
		{"AND adjacent", "AND", "", "C++Builder and .NET", []string{"C++", ".NET"}, false, false, 0},
		{"OR confidence", "OR", "", "C++Builder and .NET", []string{"C++", ".NET"}, false, true, .75},
		{"NOR rejects whole", "NOR", "", "C++ and .NET", []string{"C++", ".NET"}, false, false, 0},
		{"NOR ignores adjacent", "NOR", "", "C++Builder and ASP.NET", []string{"C++", ".NET"}, false, true, .5},
		{"explicit regex remains substring", "OR", "regex", "C++Builder", []string{`C\+\+`}, false, true, 1},
		{"case sensitive control", "OR", "", "c++", []string{"C++"}, true, false, 0},
		{"overlapping keyword rules", "OR", "", "C++", []string{"C", "C++"}, false, true, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewKeywordClassifier([]config.KeywordRule{{Name: "selected", Operator: tc.operator, Method: tc.method, Keywords: tc.keywords, CaseSensitive: tc.caseSensitive}})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Free()
			name, confidence, err := c.Classify(tc.text)
			if err != nil || (name == "selected") != tc.want || confidence != tc.confidence {
				t.Fatalf("route=%q confidence=%v error=%v, want selected=%t confidence=%v", name, confidence, err, tc.want, tc.confidence)
			}
		})
	}
}
