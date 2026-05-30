package classification

import (
	"reflect"
	"testing"
)

func TestVLLMJailbreakParserDetermineParserType(t *testing.T) {
	tests := []struct {
		name       string
		modelName  string
		parserType string
		want       string
	}{
		{name: "explicit json", parserType: "json", want: "json"},
		{name: "explicit simple", parserType: "simple", want: "simple"},
		{name: "qwen3guard model", modelName: "Qwen3Guard-8B", want: "qwen3guard"},
		{name: "json model", modelName: "policy-json-classifier", want: "json"},
		{name: "default auto", modelName: "custom-safety-model", want: "auto"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &VLLMJailbreakInference{
				modelName:  tt.modelName,
				parserType: tt.parserType,
			}
			if got := parser.determineParserType(); got != tt.want {
				t.Fatalf("determineParserType() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestVLLMJailbreakParserQwen3GuardFormats(t *testing.T) {
	tests := []struct {
		name           string
		output         string
		wantJailbreak  bool
		wantConfidence float32
		wantCategories []string
	}{
		{
			name:           "safety unsafe with categories",
			output:         "Safety: Unsafe\nCategories: Jailbreak, Illegal",
			wantJailbreak:  true,
			wantConfidence: 0.9,
			wantCategories: []string{"Jailbreak", "Illegal"},
		},
		{
			name:           "severity controversial",
			output:         "Severity level: controversial\nCategories: None",
			wantJailbreak:  false,
			wantConfidence: 0.6,
		},
		{
			name:           "category-only unsafe",
			output:         "Categories: harmful behavior",
			wantJailbreak:  true,
			wantConfidence: 0.9,
			wantCategories: []string{"harmful behavior"},
		},
		{
			name:           "unparsable",
			output:         "No structured safety result",
			wantJailbreak:  false,
			wantConfidence: 0.0,
		},
	}

	parser := &VLLMJailbreakInference{parserType: "qwen3guard"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotJailbreak, gotConfidence, gotCategories := parser.parseQwen3GuardFormat(tt.output)
			if gotJailbreak != tt.wantJailbreak {
				t.Fatalf("isJailbreak = %v, want %v", gotJailbreak, tt.wantJailbreak)
			}
			if gotConfidence != tt.wantConfidence {
				t.Fatalf("confidence = %.3f, want %.3f", gotConfidence, tt.wantConfidence)
			}
			if !reflect.DeepEqual(gotCategories, tt.wantCategories) {
				t.Fatalf("categories = %#v, want %#v", gotCategories, tt.wantCategories)
			}
		})
	}
}

func TestVLLMJailbreakParserJSONAndSimpleFormats(t *testing.T) {
	parser := &VLLMJailbreakInference{}

	isJailbreak, confidence := parser.parseJSONFormat(`{"safety":"unsafe"}`)
	if !isJailbreak || confidence != 0.9 {
		t.Fatalf("unsafe JSON parse = (%v, %.3f), want (true, 0.900)", isJailbreak, confidence)
	}

	isJailbreak, confidence = parser.parseJSONFormat(`{"is_unsafe": false}`)
	if isJailbreak || confidence != 0.1 {
		t.Fatalf("safe JSON bool parse = (%v, %.3f), want (false, 0.100)", isJailbreak, confidence)
	}

	isJailbreak, confidence = parser.parseSimpleFormat("The request is controversial but not blocked")
	if isJailbreak || confidence != 0.6 {
		t.Fatalf("simple parse = (%v, %.3f), want (false, 0.600)", isJailbreak, confidence)
	}
}

func TestVLLMJailbreakParserAutoFallback(t *testing.T) {
	parser := &VLLMJailbreakInference{parserType: "auto"}

	isJailbreak, confidence, categories := parser.parseSafetyOutput(`{"is_jailbreak": true}`)
	if !isJailbreak || confidence != 0.9 || len(categories) != 0 {
		t.Fatalf("auto JSON fallback = (%v, %.3f, %#v), want (true, 0.900, nil)", isJailbreak, confidence, categories)
	}
}
