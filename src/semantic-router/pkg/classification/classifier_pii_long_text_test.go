package classification

import (
	"fmt"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// truncatingPIIInference stands in for the real classifier: it scores only the
// first windowRunes of whatever it is handed, the way the tokenizer truncates at
// MAX_CLASSIFICATION_SEQ_LEN, and reports byte offsets relative to that text.
type truncatingPIIInference struct {
	windowRunes int
	entityText  string
	seen        []string
}

func (t *truncatingPIIInference) ClassifyTokens(text string) (candle_binding.TokenClassificationResult, error) {
	t.seen = append(t.seen, text)

	runes := []rune(text)
	if len(runes) > t.windowRunes {
		runes = runes[:t.windowRunes]
	}
	scored := string(runes)

	var entities []candle_binding.TokenEntity
	for from := 0; ; {
		index := strings.Index(scored[from:], t.entityText)
		if index < 0 {
			break
		}
		start := from + index
		entities = append(entities, candle_binding.TokenEntity{
			EntityType: "EMAIL",
			Text:       t.entityText,
			Start:      start,
			End:        start + len(t.entityText),
			Confidence: 0.99,
		})
		from = start + len(t.entityText)
	}
	return candle_binding.TokenClassificationResult{Entities: entities}, nil
}

func newLongTextPIIClassifier(entityText string) (*Classifier, *truncatingPIIInference) {
	// ~512 tokens of prose, the sequence limit the classifier is built with.
	model := &truncatingPIIInference{windowRunes: 2048, entityText: entityText}

	cfg := &config.RouterConfig{}
	cfg.PIIModel.ModelID = "test-pii-model"
	cfg.PIIMappingPath = "test-pii-mapping-path"
	cfg.PIIModel.Threshold = 0.7

	classifier, _ := newClassifierWithOptions(cfg,
		withPII(&PIIMapping{
			LabelToIdx: map[string]int{"PERSON": 0, "EMAIL": 1},
			IdxToLabel: map[string]string{"0": "PERSON", "1": "EMAIL"},
		}, &MockPIIInitializer{}, model),
	)
	return classifier, model
}

func longPIIFiller(sentences int) string {
	var builder strings.Builder
	for i := 0; i < sentences; i++ {
		fmt.Fprintf(&builder, "Sailors in year %d used the stars, then the compass, then radio beacons. ", 1500+i)
	}
	return builder.String()
}

// The classification API must not answer on the first chunk alone. The routing
// signal already scans every chunk; both surfaces have to see the same text.
// One entity sits inside the model window and one well past it, so this also
// pins that detections stay ordered by position now that they arrive per chunk.
func TestClassifyPIIWithDetails_DetectsEntityPastTheModelWindow(t *testing.T) {
	const entity = "my contact is alice@corp.example"
	text := entity + " " + longPIIFiller(400) + " " + entity

	classifier, model := newLongTextPIIClassifier(entity)

	detections, err := classifier.ClassifyPIIWithDetails(text)
	if err != nil {
		t.Fatalf("ClassifyPIIWithDetails: %v", err)
	}

	if len(model.seen) < 2 {
		t.Fatalf("a long text must be scanned in chunks; got %d call(s)", len(model.seen))
	}
	if len(detections) != 2 {
		t.Fatalf("expected both the leading and the trailing entity, got %d detections", len(detections))
	}
	if detections[0].Start >= detections[1].Start {
		t.Errorf("detections must stay ordered by position, got starts %d then %d",
			detections[0].Start, detections[1].Start)
	}
	for i, detection := range detections {
		if got := text[detection.Start:detection.End]; got != entity {
			t.Errorf("detection %d: offsets must index the original text: text[%d:%d] = %q, want %q",
				i, detection.Start, detection.End, got, entity)
		}
	}
}

// Offsets are byte offsets while chunking works in runes, so a multi-byte text
// is the case that catches a rune/byte mix-up.
func TestClassifyPIIWithDetails_OffsetsAreCorrectInMultibyteText(t *testing.T) {
	const entity = "連絡先は alice@corp.example です"

	var builder strings.Builder
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&builder, "%d年に船乗りは星を使い、次に羅針盤を使い、そして無線標識を使いました。", 1500+i)
	}
	text := builder.String() + entity

	classifier, _ := newLongTextPIIClassifier(entity)

	detections, err := classifier.ClassifyPIIWithDetails(text)
	if err != nil {
		t.Fatalf("ClassifyPIIWithDetails: %v", err)
	}
	if len(detections) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(detections))
	}
	if got := text[detections[0].Start:detections[0].End]; got != entity {
		t.Errorf("offsets must index the original text: text[%d:%d] = %q, want %q",
			detections[0].Start, detections[0].End, got, entity)
	}
}

// Chunks overlap by piiSignalChunkOverlapRunes, so an entity in that window is
// classified twice and must still be reported once.
func TestClassifyPIIWithDetails_ReportsAnOverlappedEntityOnce(t *testing.T) {
	const entity = "my contact is alice@corp.example"
	filler := []rune(longPIIFiller(400))

	text, spansContaining := "", 0
	for insertAt := 0; insertAt < len(filler); insertAt++ {
		candidate := string(filler[:insertAt]) + entity + string(filler[insertAt:])
		if n := spansFullyContaining(candidate, entity); n >= 2 {
			text, spansContaining = candidate, n
			break
		}
	}
	if text == "" {
		t.Fatal("no insertion point put the entity inside two overlapping chunks")
	}
	t.Logf("entity falls inside %d chunks", spansContaining)

	classifier, _ := newLongTextPIIClassifier(entity)

	detections, err := classifier.ClassifyPIIWithDetails(text)
	if err != nil {
		t.Fatalf("ClassifyPIIWithDetails: %v", err)
	}
	if len(detections) != 1 {
		t.Fatalf("an entity in the overlap window must be reported once, got %d detections", len(detections))
	}
	if got := text[detections[0].Start:detections[0].End]; got != entity {
		t.Errorf("offsets must index the original text: text[%d:%d] = %q, want %q",
			detections[0].Start, detections[0].End, got, entity)
	}
}

func spansFullyContaining(text, entity string) int {
	count := 0
	for _, span := range piiSignalChunkSpans(text) {
		if strings.Contains(span.Text, entity) {
			count++
		}
	}
	return count
}

// A text that fits in one chunk must still take exactly one call with the text
// unchanged: chunking is for long input only.
func TestClassifyPIIWithDetails_ShortTextTakesASingleCall(t *testing.T) {
	const entity = "my contact is alice@corp.example"
	text := "Hello, " + entity + ", thanks."

	classifier, model := newLongTextPIIClassifier(entity)

	detections, err := classifier.ClassifyPIIWithDetails(text)
	if err != nil {
		t.Fatalf("ClassifyPIIWithDetails: %v", err)
	}
	if len(model.seen) != 1 {
		t.Fatalf("a short text must take one call, got %d", len(model.seen))
	}
	if model.seen[0] != text {
		t.Errorf("a short text must be classified unchanged: got %q", model.seen[0])
	}
	if len(detections) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(detections))
	}
	if got := text[detections[0].Start:detections[0].End]; got != entity {
		t.Errorf("text[%d:%d] = %q, want %q", detections[0].Start, detections[0].End, got, entity)
	}
}
