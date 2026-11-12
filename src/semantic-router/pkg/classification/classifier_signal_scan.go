package classification

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// A guard rule (jailbreak, safety or pii, at either stage) reads its whole
// input. An input longer than one window of its deployment is read in
// overlapping windows: the tail first, then the head, then the middle in
// order, each window a call of its own, at most the deployment's
// scan_window.max_parallel of them in flight, so a window's answers count as
// soon as it is read. The rule decides on what the windows found: the most
// alarming window's answer, each label's highest probability for a set
// question, and every window's spans, at their places in the input, for a
// span question. When the stage's deadline leaves only some windows read,
// on_unscanned decides (best_effort: on the windows read); when none was, the
// rule is unknown, which on_unknown decides.

// maxScanWindows bounds the windows of one input, and so the calls its scan
// holds: the ones past it in reading order count as unread.
const maxScanWindows = 256

// errWindowLimit marks a window past maxScanWindows.
var errWindowLimit = errors.New("window_limit")

// The coverage a guard rule's read of its input reports.
const (
	scanCoverageFull    = "full"
	scanCoveragePartial = "partial"
	scanCoverageNone    = "none"
)

// scanWindow is one window of a scanned text: its byte range, and where it
// starts in code points, which span offsets count in.
type scanWindow struct {
	start, end int
	runeStart  int
}

// guardScan is the read of one input in windows by the guard rules of one
// deployment that read it, and the windows' answers as they come.
type guardScan struct {
	deployment   string
	rules        []askedRule
	whole        modelservice.Request
	requestAlone bool
	text         string
	windows      []scanWindow
	// windowState is the state a window's call reads.
	windowState func(window string) modelservice.Request
	// parallel bounds the windows in flight.
	parallel int

	mu        sync.Mutex
	responses []modelservice.Response
	errs      []error
	latency   float64
}

// guardScanned reports whether a rule reads its input in windows: the
// jailbreak, safety and pii presets.
func guardScanned(asked askedRule) bool {
	switch asked.preset {
	case presets.Jailbreak, presets.Harm, presets.HarmCategories, presets.PII:
		return true
	}
	return false
}

// scanWindowBudget is the size and overlap, in bytes, of deployment's scan
// windows, at most its card's question cap less the reserve, and how many
// of them are in flight at a time.
func (c *Classifier) scanWindowBudget(deployment string) (int, int, int) {
	window := c.Config.ModelDeployments[deployment]
	tokens, overlap := window.ScanWindowTokens()
	size := c.budgetBytes(deployment, tokens)
	return size, max(0, min(overlap*routingViewBytesPerToken, size/2-1)), window.ScanWindowParallel()
}

// newGuardScan is the scan of state in windows of size bytes overlapping by
// overlap, at most parallel of them in flight, or nil when the state fits
// one window and is read as it is.
func newGuardScan(deployment string, state modelservice.Request, size, overlap, parallel int) *guardScan {
	if stateBytes(state) <= size {
		return nil
	}
	text, windowSize, windowState := scanSource(state, size)
	windows := scanWindowsOf(text, windowSize, min(overlap, windowSize/2-1))
	return &guardScan{
		deployment: deployment, whole: state, text: text, windows: windows, windowState: windowState, parallel: parallel,
		responses: make([]modelservice.Response, len(windows)), errs: make([]error, len(windows)),
	}
}

// scanSource is the text a state's windows cover, the window size that
// leaves room for what every window carries besides, and how a window
// becomes a state. A response-stage state reads its answer in windows, each
// with a view of the rest; any other reads all of its text, the current
// turn last.
func scanSource(state modelservice.Request, size int) (string, int, func(string) modelservice.Request) {
	if state.Parts == nil {
		return state.State, size, func(window string) modelservice.Request { return modelservice.Request{State: window} }
	}
	if answer, ok := state.Parts["answer"]; ok {
		others := modelservice.Request{Parts: map[string]string{}}
		for name, text := range state.Parts {
			if name != "answer" {
				others.Parts[name] = text
			}
		}
		others = boundedState(others, size/4, config.DefaultRoutingViewHeadShare)
		return answer, size - stateBytes(others), func(window string) modelservice.Request {
			parts := map[string]string{"answer": window}
			for name, text := range others.Parts {
				parts[name] = text
			}
			return modelservice.Request{Parts: parts}
		}
	}
	names := make([]string, 0, len(state.Parts))
	for name := range state.Parts {
		names = append(names, name)
	}
	slices.SortFunc(names, func(a, b string) int {
		if (a == "request") != (b == "request") {
			if a == "request" {
				return 1
			}
			return -1
		}
		return strings.Compare(a, b)
	})
	texts := make([]string, len(names))
	for index, name := range names {
		texts[index] = state.Parts[name]
	}
	return strings.Join(texts, "\n\n"), size, func(window string) modelservice.Request { return modelservice.Request{State: window} }
}

// scanWindowsOf cuts text into windows of at most size bytes, each starting
// overlap bytes before the end of the one before, between characters, and
// between words when a word boundary is near, in text order.
func scanWindowsOf(text string, size, overlap int) []scanWindow {
	size = max(size, utf8.UTFMax)
	var windows []scanWindow
	runes := 0
	for start, counted := 0, 0; start < len(text); {
		end := len(text)
		if start+size < len(text) {
			end = runeStartBefore(text, start+size)
			if snapped := snapBack(text, end); snapped > start {
				end = snapped
			}
			if end <= start {
				_, width := utf8.DecodeRuneInString(text[start:])
				end = start + width
			}
		}
		runes += utf8.RuneCountInString(text[counted:start])
		counted = start
		windows = append(windows, scanWindow{start: start, end: end, runeStart: runes})
		if end >= len(text) {
			break
		}
		next := min(snapForward(text, runeStartAfter(text, end-overlap)), end)
		if next <= start {
			next = end
		}
		start = next
	}
	return windows
}

// scanOrder is the order windows are read in: the tail, the head, then the
// middle in order, at most maxScanWindows of them.
func scanOrder(windows int) []int {
	order := make([]int, 0, windows)
	if windows > 1 {
		order = append(order, windows-1)
	}
	for index := 0; index < windows-1; index++ {
		order = append(order, index)
	}
	return order[:min(len(order), maxScanWindows)]
}

// windowCalls are the calls that read scan's windows, in reading order, and
// mark the windows past maxScanWindows unread.
func (scan *guardScan) windowCalls() []*systemOneCall {
	order := scanOrder(len(scan.windows))
	sent := make([]bool, len(scan.windows))
	calls := make([]*systemOneCall, len(order))
	for rank, index := range order {
		window := scan.windows[index]
		request := scan.windowState(scan.text[window.start:window.end])
		request.Window, request.Parallel = rank+1, scan.parallel
		calls[rank] = &systemOneCall{deployment: scan.deployment, request: request, rules: scan.rules, scan: scan, window: index}
		sent[index] = true
	}
	for index := range scan.windows {
		if !sent[index] {
			scan.errs[index] = errWindowLimit
		}
	}
	return calls
}

// windowReaders are the work that reads scans' windows. In a stage's bundle
// every window parks its call at once and the bundle sends them, at most
// each scan's parallel bound in flight, the lowest rank first; elsewhere each
// scan has that many readers, which take its windows in reading order. A
// window not started before the deadline stays unread.
func (c *Classifier) windowReaders(ctx context.Context, scans []*guardScan) []func() {
	var readers []func()
	for _, scan := range scans {
		calls := scan.windowCalls()
		read := func(call *systemOneCall) {
			if err := ctx.Err(); err != nil {
				scan.record(call.window, modelservice.Response{}, err, 0)
				return
			}
			response, latency, err := c.sendCall(ctx, call)
			scan.record(call.window, response, err, latency)
		}
		if modelservice.InBundle(ctx) {
			for _, call := range calls {
				readers = append(readers, func() { read(call) })
			}
			continue
		}
		var next atomic.Int64
		for range min(max(scan.parallel, 1), len(calls)) {
			readers = append(readers, func() {
				for rank := int(next.Add(1)) - 1; rank < len(calls); rank = int(next.Add(1)) - 1 {
					read(calls[rank])
				}
			})
		}
	}
	return readers
}

func (scan *guardScan) record(window int, response modelservice.Response, err error, latency float64) {
	scan.mu.Lock()
	defer scan.mu.Unlock()
	scan.responses[window], scan.errs[window] = response, err
	scan.latency = max(scan.latency, latency)
}

// windowAnswer is one window's answer to a rule's question.
type windowAnswer struct {
	answer modelservice.Answer
	window int
}

// answers are the windows' answers to a rule's question, and why the first
// window without one has none.
func (scan *guardScan) answers(asked askedRule) ([]windowAnswer, string) {
	var read []windowAnswer
	reason := ""
	for index := range scan.windows {
		if err := scan.errs[index]; err != nil {
			if errors.Is(err, errWindowLimit) {
				reason = cmp.Or(reason, err.Error())
			} else {
				reason = cmp.Or(reason, modelservice.ErrorReason(err))
			}
			continue
		}
		answer, ok := scan.responses[index].Answers[asked.questionID()]
		switch {
		case !ok:
			reason = cmp.Or(reason, "missing_answer")
		case answer.Error != "":
			reason = cmp.Or(reason, answer.Error)
		default:
			read = append(read, windowAnswer{answer: answer, window: index})
		}
	}
	return read, reason
}

// applyScan resolves each rule of a scan from its windows' answers.
func (c *Classifier) applyScan(results *SignalResults, mu *sync.Mutex, scan *guardScan) {
	mu.Lock()
	defer mu.Unlock()
	recordScanPrivacyEvidence(results, scan)
	for _, asked := range scan.rules {
		rule := asked.rule
		key := signalConfidenceKey(asked.signalType, rule.Name)
		c.recordSignalExtraction(asked.signalType, rule.Name, scan.latency)
		read, reason := scan.answers(asked)
		c.recordScanCoverage(results, asked, len(read), len(scan.windows))
		switch {
		case len(read) == 0:
			results.SignalErrors[key] = systemOneSignalErrorPrefix + reason
			modelservice.RecordUnknown(scan.deployment, reason, 1)
			c.applyFailClosed(results, asked, scan.deployment, reason)
			continue
		case len(read) < len(scan.windows):
			switch rule.EffectiveOnUnscanned() {
			case config.SystemOneUnscannedMatch:
				c.applyUnscanned(results, asked, scan.deployment, reason)
				continue
			case config.SystemOneUnscannedNoMatch:
				continue
			case config.SystemOneUnscannedUnknown:
				results.SignalErrors[key] = systemOneSignalErrorPrefix + "partial_scan"
				c.applyFailClosed(results, asked, scan.deployment, "partial_scan")
				continue
			}
		}
		c.applyRuleAnswer(results, asked, scanAnswer(rule, read, scan))
	}
}

// recordScanCoverage counts how much of its input a guard rule read and, for
// one read in windows, records it for the trace and the replay record.
func (c *Classifier) recordScanCoverage(results *SignalResults, asked askedRule, read, windows int) {
	result := scanCoverageFull
	switch {
	case read == 0:
		result = scanCoverageNone
	case read < windows:
		result = scanCoveragePartial
	}
	key := signalConfidenceKey(asked.signalType, asked.rule.Name)
	metrics.RecordSignalScanCoverage(signalConfidenceKey(asked.signalType, c.scopedSignalName(asked.rule.Name)), result)
	if windows < 2 {
		return
	}
	if results.ScanCoverage == nil {
		results.ScanCoverage = make(map[string]string)
	}
	if result == scanCoveragePartial {
		result = fmt.Sprintf("%s %d/%d", scanCoveragePartial, read, windows)
	}
	results.ScanCoverage[key] = result
}

// scanAnswer is a rule's answer from the windows it read: the answer of the
// window most alarming for the rule (the highest P(yes), expected level, or
// probability of a counted option), each label's highest probability and
// every selected label for a set question, and for a span question every
// window's spans, moved to their places in the input and merged where they
// overlap.
func scanAnswer(rule config.SystemOneRule, read []windowAnswer, scan *guardScan) modelservice.Answer {
	switch rule.Question.Type {
	case config.SystemOneQuestionSet:
		return mergedSetAnswer(rule, read)
	case config.SystemOneQuestionSpan:
		return mergedSpanAnswer(read, scan)
	}
	alarm := func(answer modelservice.Answer) float64 {
		switch rule.Question.Type {
		case config.SystemOneQuestionNoul:
			return answer.Noul
		case config.SystemOneQuestionScore:
			return answer.Score
		}
		counted := rule.CountedLabels()
		if len(counted) == 0 {
			return answer.Probabilities[answer.Choice]
		}
		best := 0.0
		for _, label := range counted {
			best = max(best, answer.Probabilities[label])
		}
		return best
	}
	worst := read[0].answer
	for _, entry := range read[1:] {
		if alarm(entry.answer) > alarm(worst) {
			worst = entry.answer
		}
	}
	return worst
}

func mergedSetAnswer(rule config.SystemOneRule, read []windowAnswer) modelservice.Answer {
	merged := read[0].answer
	merged.Probabilities = map[string]float64{}
	selected := map[string]bool{}
	complete := true
	for _, entry := range read {
		for label, probability := range entry.answer.Probabilities {
			merged.Probabilities[label] = max(merged.Probabilities[label], probability)
		}
		for _, label := range entry.answer.Selected {
			selected[label] = true
		}
		complete = complete && entry.answer.InputCoverage == "complete"
	}
	merged.Selected = []string{}
	for _, label := range rule.Question.OptionKeys() {
		if selected[label] {
			merged.Selected = append(merged.Selected, label)
		}
	}
	if !complete {
		merged.InputCoverage = ""
	}
	return merged
}

func mergedSpanAnswer(read []windowAnswer, scan *guardScan) modelservice.Answer {
	merged := read[0].answer
	merged.Spans = nil
	for _, entry := range read {
		merged.Noul = max(merged.Noul, entry.answer.Noul)
		offset := scan.windows[entry.window].runeStart
		for _, span := range entry.answer.Spans {
			span.Start += offset
			span.End += offset
			merged.Spans = append(merged.Spans, span)
		}
	}
	merged.Spans = mergeSpans(merged.Spans, scan.text)
	return merged
}

// mergeSpans joins spans of one label that overlap, as two windows report
// one value both read, keeping the higher probability; text is the input
// their offsets count code points of.
func mergeSpans(spans []modelservice.Span, text string) []modelservice.Span {
	if len(spans) < 2 {
		return spans
	}
	slices.SortStableFunc(spans, func(a, b modelservice.Span) int {
		return cmp.Or(strings.Compare(a.Label, b.Label), a.Start-b.Start, a.End-b.End)
	})
	merged := []modelservice.Span{spans[0]}
	joined := false
	for _, span := range spans[1:] {
		last := &merged[len(merged)-1]
		if span.Label != last.Label || span.Start >= last.End {
			merged = append(merged, span)
			continue
		}
		if span.End > last.End {
			last.End = span.End
			joined = true
		}
		last.Probability = max(last.Probability, span.Probability)
	}
	if joined {
		runes := []rune(text)
		for index := range merged {
			if span := &merged[index]; span.Start >= 0 && span.End <= len(runes) && span.Start < span.End {
				span.Text = string(runes[span.Start:span.End])
			}
		}
	}
	slices.SortStableFunc(merged, func(a, b modelservice.Span) int { return cmp.Or(a.Start-b.Start, strings.Compare(a.Label, b.Label)) })
	return merged
}

// recordScanPrivacyEvidence records what a scan's pii question read of the
// request text alone, as recordPrivacyEvidence does for one call: complete
// only when every window was read whole.
func recordScanPrivacyEvidence(results *SignalResults, scan *guardScan) {
	if !scan.requestAlone || !slices.ContainsFunc(scan.rules, func(asked askedRule) bool { return asked.preset == presets.PII }) {
		return
	}
	complete, clean := true, true
	for index := range scan.windows {
		answer, ok := scan.responses[index].Answers[presetQuestionID(presets.PII)]
		if scan.errs[index] != nil || !ok || answer.Error != "" || answer.InputCoverage != "complete" {
			complete = false
			break
		}
		clean = clean && len(answer.Spans) == 0
	}
	evidence := NewPrivacyEvidence("request", scan.whole.State, complete, complete && clean)
	if !slices.Contains(results.PIIEvidence, evidence) {
		results.PIIEvidence = append(results.PIIEvidence, evidence)
	}
}
