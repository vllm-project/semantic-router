package extproc

import "testing"

// TestGetModelFamilyAndTemplateParam verifies model-family detection and template parameter mapping
func TestGetModelFamilyAndTemplateParam(t *testing.T) {
	testCases := []struct {
		name           string
		model          string
		expectedFamily string
		expectedParam  string
	}{
		{
			name:           "Qwen3 family",
			model:          "Qwen3-7B",
			expectedFamily: "qwen3",
			expectedParam:  "enable_thinking",
		},
		{
			name:           "DeepSeek family",
			model:          "deepseek-v31",
			expectedFamily: "deepseek",
			expectedParam:  "thinking",
		},
		{
			name:           "DeepSeek alias ds (prefix)",
			model:          "DS-1.5B",
			expectedFamily: "deepseek",
			expectedParam:  "thinking",
		},
		{
			name:           "Non-leading ds should not match DeepSeek",
			model:          "mistral-ds-1b",
			expectedFamily: "unknown",
			expectedParam:  "",
		},
		{
			name:           "GPT-OSS family",
			model:          "gpt-oss-20b",
			expectedFamily: "gpt-oss",
			expectedParam:  "reasoning_effort",
		},
		{
			name:           "GPT generic family",
			model:          "gpt-4o-mini",
			expectedFamily: "gpt",
			expectedParam:  "reasoning_effort",
		},
		{
			name:           "GPT underscore variant",
			model:          "  GPT_OSS-foo  ",
			expectedFamily: "gpt-oss",
			expectedParam:  "reasoning_effort",
		},
		{
			name:           "Unknown family",
			model:          "phi4",
			expectedFamily: "unknown",
			expectedParam:  "",
		},
		{
			name:           "Empty model name",
			model:          "",
			expectedFamily: "unknown",
			expectedParam:  "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			family, param := getModelFamilyAndTemplateParam(tc.model)
			if family != tc.expectedFamily || param != tc.expectedParam {
				t.Fatalf("for model %q got (family=%q, param=%q), want (family=%q, param=%q)", tc.model, family, param, tc.expectedFamily, tc.expectedParam)
			}
		})
	}
}
