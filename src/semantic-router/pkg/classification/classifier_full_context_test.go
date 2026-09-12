package classification

import (
	"context"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type contextCapturingCategory struct {
	MockCategoryInference
	texts []string
}

func (m *contextCapturingCategory) ClassifyWithProbabilities(ctx context.Context, text string) (candle_binding.ClassResultWithProbs, error) {
	m.texts = append(m.texts, text)
	return m.MockCategoryInference.ClassifyWithProbabilities(ctx, text)
}

func TestNativeLongContextBudgetReachesSignalDispatcher(t *testing.T) {
	full := strings.Repeat("完整上下文 English context. ", 1000)
	for _, tc := range []struct {
		name       string
		limit      int
		compressed bool
	}{
		{"historical default samples", 0, false},
		{"explicit short budget samples", 512, false},
		{"long budget preserves text", 32768, false},
		{"long budget preserves compression exemption", 32768, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mock := &contextCapturingCategory{}
			c := buildDomainClassifier(&mock.MockCategoryInference)
			c.categoryInference = mock
			c.Config.CategoryModel.Variant = config.CategoryVariantMmBERT32K
			c.Config.CategoryModel.MaxSequenceLength = tc.limit
			text, original := full, ""
			if tc.compressed {
				text, original = "compressed summary", full
			}
			c.EvaluateAllSignalsWithContext(text, full, text, nil, nil, false, false,
				original, map[string]bool{config.SignalTypeDomain: true}, ConversationFacts{}, "")
			if len(mock.texts) != 1 {
				t.Fatalf("inference calls = %d, want 1", len(mock.texts))
			}
			if tc.limit > 512 {
				if mock.texts[0] != full {
					t.Fatal("long-context inference received sampled or compressed input")
				}
			} else if !strings.Contains(mock.texts[0], signalWindowOmissionMarker) {
				t.Fatal("short-context policy lost its bounded input")
			}
		})
	}
}

func TestNativeLongContextPIIUsesOnePassWithOriginalByteOffsets(t *testing.T) {
	const email = "alice@example.org"
	text := strings.Repeat("中文填充。", 900) + email + strings.Repeat(" plain context ", 200) + email
	c, model := newLongTextPIIClassifier(email)
	c.Config.PIIModel.UseMmBERT32K = true
	c.Config.PIIModel.MaxSequenceLength = 32768
	model.windowRunes = 32768
	got, err := c.ClassifyPIIWithDetails(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	if len(model.seen) != 1 || model.seen[0] != text {
		t.Fatal("explicit long-context PII did not receive one complete input")
	}
	if len(got) != 2 {
		t.Fatalf("got %d entities, want 2", len(got))
	}
	for _, span := range got {
		if text[span.Start:span.End] != email {
			t.Fatalf("entity offsets do not index the original UTF-8 text: %+v", span)
		}
	}
}

func TestOpenVINORejectsBudgetsItCannotEnforce(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "openvino")
	for _, limit := range []int{1, 128, 256, 513, 32768} {
		initializer := &MmBERT32KCategoryInitializerImpl{maxSequenceLength: limit}
		err := initializer.Init("unused-model-path", true, 2)
		if err == nil || !strings.Contains(err.Error(), "only the default 512-token budget") {
			t.Fatalf("limit=%d did not fail before model loading: %v", limit, err)
		}
	}
}
