package serving

import (
	"fmt"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// byteSpans converts runtime spans, whose offsets count Unicode code points
// into text, to UTF-8 byte offsets. The runtime's span text must equal the
// bytes the offsets select, so an offset disagreement is never published.
func byteSpans(text string, spans []modelservice.Span) ([]tasks.TokenEntity, error) {
	if len(spans) == 0 {
		return nil, nil
	}
	offsets := codePointOffsets(text)
	points := len(text)
	if offsets != nil {
		points = len(offsets) - 1
	}
	entities := make([]tasks.TokenEntity, len(spans))
	for i, span := range spans {
		if span.Start < 0 || span.End <= span.Start || span.End > points {
			return nil, fmt.Errorf("%w: span [%d, %d) is outside the input", binding.ErrInvalidResult, span.Start, span.End)
		}
		start, end := span.Start, span.End
		if offsets != nil {
			start, end = offsets[start], offsets[end]
		}
		if text[start:end] != span.Text {
			return nil, fmt.Errorf("%w: span text differs from its input range", binding.ErrInvalidResult)
		}
		entities[i] = tasks.TokenEntity{EntityType: span.Label, Start: start, End: end, Text: text[start:end], Confidence: float32(span.Probability)}
	}
	return entities, nil
}

// codePointOffsets maps code point index to byte offset (one extra entry for
// the end); nil means the text is ASCII and both units agree.
func codePointOffsets(text string) []int {
	ascii := true
	for i := 0; i < len(text); i++ {
		if text[i] >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	if ascii {
		return nil
	}
	offsets := make([]int, 0, utf8.RuneCountInString(text)+1)
	for offset := range text {
		offsets = append(offsets, offset)
	}
	return append(offsets, len(text))
}
