//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"strings"
	"testing"
)

func TestWindowDocumentBudgetDoesNotRaiseSingleForwardLimit(t *testing.T) {
	options := specialTokenFixture(t, "sequence")
	configureSequenceFixture(t, options, map[string]any{"max_position_embeddings": 32768})
	options.MaxInputTokens, options.DocumentMaxInputTokens = 32768, 65536
	options.ExecutionMaxInputTokens = 32768
	model, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	shared, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer shared.Close()
	text := strings.Repeat("hello ", 65534)
	for _, handle := range []*SequenceClassifier{model, shared} {
		info, err := handle.Info()
		if err != nil {
			t.Fatal(err)
		}
		if info.EffectiveLimit != 32768 || info.DocumentMaxInputTokens != 65536 {
			t.Fatalf("conflated physical and document budgets: %+v", info)
		}
		if _, classifyErr := handle.Classify(text); classifyErr == nil {
			t.Fatal("single forward accepted a document above its physical limit")
		}
		result, err := handle.ClassifyWindows(text, SequenceWindowOptions{Size: 32768, Overlap: 128})
		if err != nil {
			t.Fatal(err)
		}
		if result.Input.OriginalTokens != 65536 || result.Input.ProcessedTokens != 65536 || result.Input.Truncated || result.ContentTokens != 65534 {
			t.Fatalf("incomplete document usage: %+v", result.Input)
		}
		if len(result.Windows) != 3 || result.Windows[0].Start != 0 || result.Windows[0].End != 32766 || result.Windows[1].Start != 32638 || result.Windows[2].End != 65534 {
			t.Fatalf("incomplete original-token coverage: %+v", result.Windows)
		}
		if _, err := handle.ClassifyWindows(text, SequenceWindowOptions{Size: 32769}); err == nil {
			t.Fatal("document budget increased physical window")
		}
		if _, err := handle.ClassifyWindows(text+"hello", SequenceWindowOptions{Size: 32768}); err == nil {
			t.Fatal("document budget overflow accepted")
		}
	}
	options.MaxInputTokens = 32769
	if invalid, err := LoadSequenceClassifier(options); err == nil {
		invalid.Close()
		t.Fatal("document budget raised checkpoint capacity")
	}
}

func TestTokenWindowDocumentBudgetPreservesOriginalTail(t *testing.T) {
	options := specialTokenFixture(t, "token")
	options.MaxInputTokens, options.DocumentMaxInputTokens = 8, 24
	configureSequenceFixture(t, options, map[string]any{"max_position_embeddings": 8})
	model, err := LoadTokenClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	text := strings.Repeat("hello ", 21) + "秘密"
	if _, detectErr := model.Detect(text); detectErr == nil {
		t.Fatal("single token forward accepted oversized document")
	}
	result, err := model.DetectWindows(text, SequenceWindowOptions{Size: 8, Overlap: 2})
	if err != nil {
		t.Fatal(err)
	}
	if result.Input.OriginalTokens != 24 || result.Input.ProcessedTokens != 24 || result.Input.Truncated || result.ContentTokens != 22 || len(result.Windows) != 5 || result.Windows[4] != [2]int{16, 22} {
		t.Fatalf("incomplete token coverage: %+v", result)
	}
	if len(result.Spans) != 22 {
		t.Fatalf("fixture token entities were lost or duplicated: %+v", result.Spans)
	}
	tail := result.Spans[len(result.Spans)-1]
	if tail.Text != "秘密" || tail.Start != len(text)-len("秘密") || tail.End != len(text) {
		t.Fatalf("lost document-tail entity or UTF-8 offsets: %+v", result.Spans)
	}
	if _, err := model.DetectWindows(text, SequenceWindowOptions{Size: 9}); err == nil {
		t.Fatal("oversized physical token window accepted")
	}
	if _, err := model.DetectWindows(text+" hello", SequenceWindowOptions{Size: 8}); err == nil {
		t.Fatal("document overflow accepted")
	}
}

func TestLabelWindowsNeverTruncateTheDocument(t *testing.T) {
	options := specialTokenFixture(t, "sequence")
	configureSequenceFixture(t, options, map[string]any{"max_position_embeddings": 8, "problem_type": "multi_label_classification"})
	options.MaxInputTokens, options.DocumentMaxInputTokens, options.Overflow = 8, 24, "truncate_right"
	options.ExecutionMaxInputTokens = 8
	model, err := LoadLabelScorer(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	text := strings.Repeat("hello ", 22)
	single, err := model.Score(text)
	if err != nil || single.Input.ProcessedTokens != 8 || !single.Input.Truncated {
		t.Fatalf("single-forward truncation changed: %+v %v", single, err)
	}
	result, err := model.ScoreWindows(text, SequenceWindowOptions{Size: 8, Overlap: 2})
	if err != nil {
		t.Fatal(err)
	}
	if result.Input.ProcessedTokens != 24 || result.Input.Truncated || len(result.Windows) != 5 {
		t.Fatalf("window task silently truncated document: %+v", result)
	}
	if _, err := model.ScoreWindows(text+"hello", SequenceWindowOptions{Size: 8}); err == nil {
		t.Fatal("document overflow was silently truncated")
	}
	options.DocumentMaxInputTokens = 7
	if invalid, err := LoadLabelScorer(options); err == nil {
		invalid.Close()
		t.Fatal("document budget below physical budget accepted")
	}
}
