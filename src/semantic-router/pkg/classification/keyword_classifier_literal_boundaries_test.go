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
		{"python", "I like python code", false}, {"数学", "我喜欢数学", false},
		{"C++", "Explain C++ move semantics", false}, {"C#", "Convert this C# code", false}, {".NET", "Is .NET 8 faster?", false},
		{"Answer:", "Answer: 42", false}, {"привет", "скажи привет всем", false}, {"café", "a café nearby", false},
		{"cat", "concatenate", true}, {"café", "caféteria", true}, {"привет", "приветствие", true}, {"python", "python3", true},
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
