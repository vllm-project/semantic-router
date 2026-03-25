package classification

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMultilingualTextUnitCount(t *testing.T) {
	tests := []struct {
		name string
		text string
		want int
	}{
		{
			name: "english words use whitespace independent runs",
			text: "please output markdown table",
			want: 4,
		},
		{
			name: "chinese counts continuous cjk characters",
			text: "请用表格输出",
			want: 6,
		},
		{
			name: "mixed script counts latin run and cjk characters",
			text: "请输出JSON表格",
			want: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := multilingualTextUnitCount(tt.text); got != tt.want {
				t.Fatalf("multilingualTextUnitCount(%q) = %d, want %d", tt.text, got, tt.want)
			}
		})
	}
}

func TestKeywordOccurrenceCountMatchesContinuousCJKAndMixedScript(t *testing.T) {
	t.Run("continuous chinese keyword matches", func(t *testing.T) {
		got := keywordOccurrenceCount("请用表格输出，最多三点。", []string{"表格", "最多"}, false)
		if got != 2 {
			t.Fatalf("keywordOccurrenceCount returned %d, want 2", got)
		}
	})

	t.Run("latin keyword next to cjk still matches", func(t *testing.T) {
		got := keywordOccurrenceCount("请输出JSON格式。", []string{"json"}, false)
		if got != 1 {
			t.Fatalf("keywordOccurrenceCount returned %d, want 1", got)
		}
	})

	t.Run("latin keyword inside latin word does not match", func(t *testing.T) {
		got := keywordOccurrenceCount("jsonschema output", []string{"json"}, false)
		if got != 0 {
			t.Fatalf("keywordOccurrenceCount returned %d, want 0", got)
		}
	})
}

func TestStructureDensityUsesMultilingualTextUnits(t *testing.T) {
	rule := config.StructureRule{
		Name: "format_directive_dense",
		Feature: config.StructureFeature{
			Type: "density",
			Source: config.StructureSource{
				Type:     "keyword_set",
				Keywords: []string{"表格", "json"},
			},
		},
	}

	classifier, err := NewStructureClassifier([]config.StructureRule{rule})
	if err != nil {
		t.Fatalf("NewStructureClassifier error: %v", err)
	}

	matches, err := classifier.Classify("请用JSON表格输出")
	if err != nil {
		t.Fatalf("Classify error: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}

	want := 2.0 / 7.0
	if math.Abs(matches[0].Value-want) > 1e-9 {
		t.Fatalf("density = %v, want %v", matches[0].Value, want)
	}
}
