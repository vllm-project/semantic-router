package classification

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// firstCopyPIIInference labels only the first copy of each value, the way
// Vela 2.0 scores a repeated name far lower the second time it appears.
type firstCopyPIIInference struct {
	values map[string]string // value -> entity type
}

func (f *firstCopyPIIInference) ClassifyTokens(_ context.Context, text string) (tasks.TokenClassificationResult, error) {
	var entities []tasks.TokenEntity
	for value, kind := range f.values {
		if index := strings.Index(text, value); index >= 0 {
			entities = append(entities, tasks.TokenEntity{
				EntityType: kind, Text: value, Start: index, End: index + len(value), Confidence: 0.95,
			})
		}
	}
	available := true
	return tasks.TokenClassificationResult{Entities: entities, ScoresAvailable: &available}, nil
}

// fixedSpanPIIInference returns the PERSON spans it is given.
type fixedSpanPIIInference struct {
	spans [][2]int
}

func (f *fixedSpanPIIInference) ClassifyTokens(_ context.Context, text string) (tasks.TokenClassificationResult, error) {
	var entities []tasks.TokenEntity
	for _, span := range f.spans {
		entities = append(entities, tasks.TokenEntity{
			EntityType: "PERSON", Text: text[span[0]:span[1]], Start: span[0], End: span[1], Confidence: 0.95,
		})
	}
	available := true
	return tasks.TokenClassificationResult{Entities: entities, ScoresAvailable: &available}, nil
}

func detectFirstCopies(t *testing.T, text string, values map[string]string) []PIIDetection {
	t.Helper()
	return detectWith(t, text, &firstCopyPIIInference{values: values})
}

func detectWith(t *testing.T, text string, inference PIIInference) []PIIDetection {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.PIIModel.ModelID = "test-pii-model"
	cfg.PIIMappingPath = "test-pii-mapping-path"
	cfg.PIIModel.Threshold = 0.5
	classifier, err := newClassifierWithOptions(cfg, withPII(&PIIMapping{
		LabelToIdx: map[string]int{"O": 0, "PERSON": 1, "PHONE_NUMBER": 2},
		IdxToLabel: map[string]string{"0": "O", "1": "PERSON", "2": "PHONE_NUMBER"},
	}, &MockPIIInitializer{}, inference))
	if err != nil {
		t.Fatalf("newClassifierWithOptions: %v", err)
	}
	detections, err := classifier.ClassifyPIIWithDetails(context.Background(), text)
	if err != nil {
		t.Fatalf("ClassifyPIIWithDetails: %v", err)
	}
	return detections
}

func spanTexts(text string, detections []PIIDetection) []string {
	out := make([]string, 0, len(detections))
	for _, detection := range detections {
		out = append(out, detection.EntityType+":"+text[detection.Start:detection.End])
	}
	return out
}

func TestClassifyPIIWithDetails_CoversEveryCopyOfADetectedValue(t *testing.T) {
	text := "John Smith lives in Boston. Call 555-0142. Yesterday I met John Smith, who said 555-0142 works."
	detections := detectFirstCopies(t, text, map[string]string{"John Smith": "PERSON", "555-0142": "PHONE_NUMBER"})

	got := strings.Join(spanTexts(text, detections), " | ")
	want := "PERSON:John Smith | PHONE_NUMBER:555-0142 | PERSON:John Smith | PHONE_NUMBER:555-0142"
	if got != want {
		t.Fatalf("detections in position order:\n got %s\nwant %s", got, want)
	}
	masked := buildMaskedText(text, detections)
	if strings.Contains(masked, "John Smith") || strings.Contains(masked, "555-0142") {
		t.Fatalf("a repeated value must not survive masking: %q", masked)
	}
}

// Offsets are bytes, so a non-ASCII value at offset 0 checks both the start
// edge and the byte arithmetic of the copies.
func TestClassifyPIIWithDetails_CoversCopiesInNonASCIIText(t *testing.T) {
	text := "김민지 고객님의 주문이 접수되었습니다. 김민지 고객님께 배송 일정을 안내드립니다."
	detections := detectFirstCopies(t, text, map[string]string{"김민지": "PERSON"})

	if len(detections) != 2 || detections[0].Start != 0 {
		t.Fatalf("want both copies with the first at offset 0, got %v", spanTexts(text, detections))
	}
	for _, detection := range detections {
		if text[detection.Start:detection.End] != "김민지" {
			t.Fatalf("offsets must index the original text, got %q", text[detection.Start:detection.End])
		}
	}
}

func TestClassifyPIIWithDetails_CopiesMustBeWholeWordsWithTheSameCase(t *testing.T) {
	text := "Will called. Then Will Turner said he will ask Willa and Will_x. Will left."
	detections := detectFirstCopies(t, text, map[string]string{"Will": "PERSON"})

	var starts []int
	for _, detection := range detections {
		starts = append(starts, detection.Start)
	}
	// "will" differs in case, and "Willa" and "Will_x" are longer words.
	want := []int{0, strings.Index(text, "Will Turner"), strings.LastIndex(text, "Will left")}
	if len(starts) != len(want) {
		t.Fatalf("got starts %v, want %v", starts, want)
	}
	for i := range want {
		if starts[i] != want[i] {
			t.Fatalf("got starts %v, want %v", starts, want)
		}
	}
}

// A model span over part of a copy is not coverage of the copy: masking only
// "John" of the second "John Smith" would send "Smith" in clear text.
func TestClassifyPIIWithDetails_CoversTheRestOfAPartlyDetectedCopy(t *testing.T) {
	text := "John Smith met John Smith"
	detections := detectWith(t, text, &fixedSpanPIIInference{spans: [][2]int{{0, 10}, {15, 19}}})

	if masked := buildMaskedText(text, detections); masked != "[PERSON] met [PERSON]" {
		t.Fatalf("got %q from %v", masked, spanTexts(text, detections))
	}
}

// Copies of a value can overlap each other: after "Anna Anna" at the start of
// "Anna Anna Anna", the next whole-word copy starts inside the first one.
func TestClassifyPIIWithDetails_CoversOverlappingCopies(t *testing.T) {
	text := "Anna Anna Anna"
	detections := detectWith(t, text, &fixedSpanPIIInference{spans: [][2]int{{0, 9}}})

	if masked := buildMaskedText(text, detections); masked != "[PERSON]" {
		t.Fatalf("got %q from %v", masked, spanTexts(text, detections))
	}
}

func TestClassifyPIIWithDetails_SingleCharacterValuesAreNotCopied(t *testing.T) {
	text := "A said hello. Later A left."
	detections := detectFirstCopies(t, text, map[string]string{"A": "PERSON"})
	if len(detections) != 1 {
		t.Fatalf("a one-character value must not mask every copy, got %v", spanTexts(text, detections))
	}
}

// buildMaskedText replaces each detection's bytes the way the PII API's
// masked_text does, merging overlaps, without the placeholder numbering.
func buildMaskedText(text string, detections []PIIDetection) string {
	var builder strings.Builder
	last := 0
	for _, detection := range detections {
		if detection.Start < last {
			last = max(last, detection.End)
			continue
		}
		builder.WriteString(text[last:detection.Start])
		builder.WriteString("[" + detection.EntityType + "]")
		last = detection.End
	}
	builder.WriteString(text[last:])
	return builder.String()
}

// A model that labels every copy itself gives the expansion nothing to add,
// and the work must stay linear: rescanning every copy for each detection
// took 3.3 s for 1,600 copies and grows with the cube of the count.
func TestCoverRepeatedValues_DenseInputStaysLinear(t *testing.T) {
	const copies = 20000
	text := strings.Repeat("John Smith ", copies)
	every := make([]PIIDetection, 0, copies)
	for i := 0; i < copies; i++ {
		start := i * len("John Smith ")
		every = append(every, PIIDetection{EntityType: "PERSON", Start: start, End: start + len("John Smith"), Confidence: 0.9})
	}
	started := time.Now()
	got, err := coverRepeatedValues(context.Background(), text, every)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != copies {
		t.Fatalf("every copy was already labeled, got %d detections, want %d", len(got), copies)
	}
	got, err = coverRepeatedValues(context.Background(), text, every[:1])
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != copies {
		t.Fatalf("one labeled copy must cover the rest, got %d detections, want %d", len(got), copies)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("%d copies took %v", copies, elapsed)
	}
}

func TestCoverRepeatedValues_StopsWhenTheRequestIsCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := coverRepeatedValues(ctx, "John Smith met John Smith", []PIIDetection{{EntityType: "PERSON", Start: 0, End: 10}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}
