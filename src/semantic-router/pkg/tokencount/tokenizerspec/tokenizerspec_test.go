package tokenizerspec

import "testing"

func TestParse(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want Spec
	}{
		{"o200k_base", Spec{Kind: Builtin, Value: "o200k_base"}},
		{" cl100k_base ", Spec{Kind: Builtin, Value: "cl100k_base"}},
		{"/models/tokenizer.json", Spec{Kind: File, Value: "/models/tokenizer.json"}},
		{"./tokenizer.json", Spec{Kind: File, Value: "./tokenizer.json"}},
		{"vendor/qwen.json", Spec{Kind: File, Value: "vendor/qwen.json"}},
		{"Qwen/Qwen3-8B", Spec{Kind: Hub, Value: "Qwen/Qwen3-8B", Revision: "main"}},
		{"deepseek-ai/DeepSeek-V3", Spec{Kind: Hub, Value: "deepseek-ai/DeepSeek-V3", Revision: "main"}},
		{"gpt2", Spec{Kind: Hub, Value: "gpt2", Revision: "main"}},
		{"zai-org/GLM-4.6@v1.0", Spec{Kind: Hub, Value: "zai-org/GLM-4.6", Revision: "v1.0"}},
		{"org/model@refs/pr/1", Spec{Kind: Hub, Value: "org/model", Revision: "refs/pr/1"}},
		{"org/model@0123456789abcd", Spec{Kind: Hub, Value: "org/model", Revision: "0123456789abcd"}},
	} {
		got, err := Parse(tc.raw)
		if err != nil || got != tc.want {
			t.Errorf("Parse(%q) = %+v, %v; want %+v", tc.raw, got, err, tc.want)
		}
	}
	for _, raw := range []string{"", "  ", "org/model/extra", "has space", "org/model@has space", "-org/model"} {
		if _, err := Parse(raw); err == nil {
			t.Errorf("Parse(%q) should fail", raw)
		}
	}
}
