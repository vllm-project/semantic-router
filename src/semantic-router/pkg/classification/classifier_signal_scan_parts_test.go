package classification

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// TestAScanOfNamedPartsKeepsTheHistoryAndRequestBoundary covers issue #6339: a
// state of named parts (history and the current request) that exceeds one scan
// window must keep those parts in every window. Flattening them into one state
// string asks the model a different question than the non-windowed path does,
// so a history-aware rule changes its meaning at the window threshold.
func TestAScanOfNamedPartsKeepsTheHistoryAndRequestBoundary(t *testing.T) {
	history := strings.Repeat("Earlier the committee reviewed the quarterly report and agreed to meet again. ", 40)
	request := "Now summarize the chapter above in five bullet points."
	state := modelservice.Request{Parts: map[string]string{"history": history, "request": request}}
	scan := newGuardScan("decision", state, scanBudget{size: 256, most: 256, overlap: 32, parallel: 2})
	if scan == nil {
		t.Fatal("a state larger than one window must be scanned")
	}
	calls := scan.windowCalls(context.Background())
	if len(calls) < 3 {
		t.Fatalf("windows = %d, want the input cut into several", len(calls))
	}

	texts := map[string]string{"history": history, "request": request}
	offsets := map[string]int{"history": 0, "request": len(history) + len(scanPartSeparator)}

	type chunk struct{ start, end int }
	covered := map[string][]chunk{}
	for _, call := range calls {
		window := scan.windows[call.window]
		read := call.request
		if read.State != "" {
			t.Fatalf("window %d was flattened into one state: %q", call.window, read.State)
		}
		if len(read.Parts) == 0 {
			t.Fatalf("window %d carries no part", call.window)
		}
		for name, part := range read.Parts {
			text, ok := texts[name]
			if !ok {
				t.Fatalf("window %d invented part %q", call.window, name)
			}
			offset := offsets[name]
			start := max(window.start, offset) - offset
			end := min(window.end, offset+len(text)) - offset
			if start < 0 || end > len(text) || start >= end {
				t.Fatalf("window %d covers part %q out of bounds: [%d,%d) of %d", call.window, name, start, end, len(text))
			}
			if part != text[start:end] {
				t.Fatalf("window %d part %q = %q, want the text it covers %q", call.window, name, part, text[start:end])
			}
			covered[name] = append(covered[name], chunk{start: start, end: end})
		}
	}

	// The first window lies inside the history, so it carries the history
	// alone: the current request is not presented as part of the context.
	for _, call := range calls {
		if call.window != 0 {
			continue
		}
		if _, ok := call.request.Parts["history"]; !ok {
			t.Fatalf("head window parts = %v, want history", call.request.Parts)
		}
		if _, ok := call.request.Parts["request"]; ok {
			t.Fatalf("head window carries the current request inside the history: %v", call.request.Parts)
		}
	}

	// The tail window ends with the input, so the current request reaches it
	// whole and under its own name.
	tail := calls[0]
	for _, call := range calls {
		if scan.windows[call.window].end > scan.windows[tail.window].end {
			tail = call
		}
	}
	if got := tail.request.Parts["request"]; got != request {
		t.Fatalf("tail window request part = %q, want %q", got, request)
	}

	// Every part is covered exactly by the windows that carry it, so no text
	// was dropped while the boundary was kept.
	for name, text := range texts {
		chunks := covered[name]
		if len(chunks) == 0 {
			t.Fatalf("part %q was never read", name)
		}
		for i := 1; i < len(chunks); i++ {
			for j := i; j > 0 && chunks[j].start < chunks[j-1].start; j-- {
				chunks[j], chunks[j-1] = chunks[j-1], chunks[j]
			}
		}
		var built strings.Builder
		read := 0
		for _, part := range chunks {
			if part.end <= read {
				continue
			}
			built.WriteString(text[max(part.start, read):part.end])
			read = part.end
		}
		if built.String() != text || read != len(text) {
			t.Fatalf("part %q read as %d/%d bytes, want the whole part", name, read, len(text))
		}
	}
}

// TestAWindowOfAStateWithoutPartsStaysAState keeps the other branch honest: a
// state that carries one text has no parts to preserve, and its windows still
// read that text.
func TestAWindowOfAStateWithoutPartsStaysAState(t *testing.T) {
	text := strings.Repeat("The clerks counted 412 barges this spring. ", 40)
	scan := newGuardScan("decision", modelservice.Request{State: text}, scanBudget{size: 128, most: 128, overlap: 16, parallel: 1})
	if scan == nil {
		t.Fatal("a state larger than one window must be scanned")
	}
	for _, call := range scan.windowCalls(context.Background()) {
		if len(call.request.Parts) != 0 {
			t.Fatalf("window of a state without parts invented %v", call.request.Parts)
		}
		if !strings.Contains(text, call.request.State) {
			t.Fatalf("window state %q is not part of the input", call.request.State)
		}
	}
}
