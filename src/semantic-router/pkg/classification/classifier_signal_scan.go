package classification

import (
	"cmp"
	"context"
	cryptorand "crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// A guard rule (jailbreak, safety or pii, at either stage) reads its whole
// input. An input longer than one window of its deployment is read in
// overlapping windows: the tail first, then the head, then the middle by
// bisection, each window a call of its own, at most the deployment's
// scan_window.max_parallel of them in flight, so a window's answers count as
// soon as it is read. The rule decides on what the windows found: the most
// alarming window's answer, each label's highest probability for a set
// question, and every window's spans, at their places in the input, for a
// span question. A window whose answer makes the rule match decides it, and
// when that match alone refuses the request, the stage's scans end there. A
// window larger than the configured size dilutes what it holds, so one whose
// answer puts a rule close below its threshold (scan_window.reread_margin) is
// read again in windows of that size: at once within half the margin, where
// a diluted match lands, and after the windows still unread further below.
// When the stage's deadline leaves only some windows read, on_unscanned
// decides (best_effort: on the windows read, once they reach the rule's
// min_coverage); when none was, the rule is unknown, which on_unknown decides.

// maxScanWindows bounds the windows of one input, and so the calls its scan
// holds. An input that needs more windows of its deployment's size is read in
// larger ones between its head and its tail, up to the largest its model
// takes; only past those do the windows beyond the bound in reading order
// count as unread.
const maxScanWindows = 256

// errScanDecided ends the windows of a request stage's guard scans once one
// window decided a rule whose match refuses the request.
var errScanDecided = errors.New("scan_decided")

// errWindowLimit marks a window past maxScanWindows.
var errWindowLimit = errors.New("window_limit")

// errWindowReplaced marks a window that a scan which fell behind reads in a
// larger window instead (replan).
var errWindowReplaced = errors.New("window_replaced")

// The coverage a guard rule's read of its input reports.
const (
	scanCoverageFull    = "full"
	scanCoveragePartial = "partial"
	scanCoverageNone    = "none"
)

// scanWindow is one window of a scanned text: its byte range, where it
// starts in code points, which span offsets count in, and the windows of its
// scan's cut it reads, first to last: itself for one of the cut, a run of
// them for one read in their place, none (-1) for one of the windows a window
// is read again in.
type scanWindow struct {
	start, end  int
	runeStart   int
	first, last int
}

// guardScan is the read of one input in windows by the guard rules of one
// deployment that read it, and the windows' answers as they come.
type guardScan struct {
	deployment   string
	rules        []askedRule
	whole        modelservice.Request
	requestAlone bool
	text         string
	// windows are the input's cut, its first cut windows, and then the
	// windows that read part of it again.
	windows []scanWindow
	cut     int
	// size is the bytes of a window of the configured size, grown those of
	// the cut's windows between its head and its tail, most the largest a
	// window may grow to, and overlap what two neighbours share.
	size, grown, most, overlap int
	// rereadMargin is how far below its threshold, in log-odds, a window
	// larger than size may put a rule, short of a match, to be read again in
	// windows of size.
	rereadMargin float64
	// order is the cut's reading order (scanOrder), turned by turn.
	order []int
	turn  int
	// windowState is the state a window's call reads.
	windowState func(window string) modelservice.Request
	// parallel bounds the windows in flight.
	parallel int
	// refusing are the rules a window that makes them match ends the
	// stage's scans for, and stop ends them.
	refusing []askedRule
	stop     func()
	pace     scanPace

	mu        sync.Mutex
	responses []modelservice.Response
	errs      []error
	done      []bool
	latency   float64
	// waiting are the re-reads that wait for the windows still unread, and
	// rank the highest rank the scan's windows took.
	waiting []waitingReread
	rank    int
}

// scanBudget is how a deployment reads a long input: windows of size bytes,
// or of up to most bytes when the input needs more than maxScanWindows of
// them or its scan falls behind, each overlapping the one before by overlap
// bytes, at most parallel of them in flight; a window larger than size whose
// answer puts a rule within rereadMargin below its threshold, in log-odds, is
// read again in windows of size.
type scanBudget struct {
	size, most, overlap, parallel int
	rereadMargin                  float64
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

// scanWindowBudget is how deployment reads a long input: its scan windows'
// size and overlap in bytes, at most its card's question cap less the
// reserve; the largest window a long input's windows may grow to, its
// scan_window.max_tokens or else that cap (none larger without either, as
// only they say what the model takes); how many windows are in flight at a
// time; and its scan_window.reread_margin.
func (c *Classifier) scanWindowBudget(deployment string) scanBudget {
	window := c.Config.ModelDeployments[deployment]
	tokens, overlap := window.ScanWindowTokens()
	size := c.budgetBytes(deployment, tokens)
	most := size
	if maxTokens := window.ScanWindowMaxTokens(); maxTokens > 0 {
		most = max(size, c.budgetBytes(deployment, maxTokens))
	} else if card, known := c.deploymentCard(deployment); known && card.QuestionTokens() > 0 {
		most = max(size, c.budgetBytes(deployment, config.MaxModelInputTokens))
	}
	return scanBudget{
		size: size, most: most, overlap: max(0, min(overlap*routingViewBytesPerToken, size/2-1)), parallel: window.ScanWindowParallel(),
		rereadMargin: window.ScanWindowRereadMargin(),
	}
}

// newGuardScan is the scan of state in budget's windows, read in an order
// of its own, or nil when the state fits one window and is read as it is.
func newGuardScan(deployment string, state modelservice.Request, budget scanBudget) *guardScan {
	if stateBytes(state) <= budget.size {
		return nil
	}
	text, size, windowState := scanSource(state, budget.size)
	overlap, most := min(budget.overlap, size/2-1), budget.most-budget.size+size
	windows, grown := cutScanWindows(text, size, most, overlap)
	// The order's turn comes from crypto/rand: a client that could predict
	// it could put an attack where the windows read last are.
	var turn [2]byte
	_, _ = cryptorand.Read(turn[:])
	offset := int(binary.LittleEndian.Uint16(turn[:]))
	return &guardScan{
		deployment: deployment, whole: state, text: text, windows: windows, cut: len(windows),
		size: size, grown: grown, most: most, overlap: overlap, rereadMargin: budget.rereadMargin,
		order: scanOrder(len(windows), offset), turn: offset, windowState: windowState, parallel: budget.parallel, stop: func() {},
		responses: make([]modelservice.Response, len(windows)), errs: make([]error, len(windows)), done: make([]bool, len(windows)),
	}
}

// cutScanWindows cuts text into windows of size bytes or, when those would
// number more than maxScanWindows, into as few larger ones, up to most bytes,
// as cover the text in the bound, so it leaves no text unread unless even the
// largest windows cannot cover it. The head and the tail, the last size bytes
// of the text, stay at size either way: they are read first, and a larger
// window dilutes what it holds. It returns the windows and the size of those
// between the head and the tail.
func cutScanWindows(text string, size, most, overlap int) ([]scanWindow, int) {
	windows, grown := scanWindowsOf(text, size, overlap), size
	if len(windows) > maxScanWindows && most > size {
		head := windows[0]
		start := snapForward(text, runeStartAfter(text, len(text)-size))
		tail := scanWindow{start: start, end: len(text), runeStart: utf8.RuneCountInString(text[:start])}
		from, to := windows[1].start, runeStartAfter(text, tail.start+overlap)
		// A cut moves back to a character and a word boundary, so a window
		// moves on by at least its size less the overlap and this much.
		shortfall := overlap + routingViewSnap + utf8.UTFMax
		middle := maxScanWindows - 2
		grown = min(most, (to-from+(middle-1)*shortfall)/middle+1)
		windows = slices.Concat([]scanWindow{head}, cutWindows(text, from, to, grown, overlap), []scanWindow{tail})
	}
	for index := range windows {
		windows[index].first, windows[index].last = index, index
	}
	return windows, grown
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
	return cutWindows(text, 0, len(text), size, overlap)
}

// cutWindows cuts the part of text from byte from to byte to, both between
// characters, as scanWindowsOf cuts a text, with offsets in text; its windows
// read none of the cut's (first and last -1).
func cutWindows(text string, from, to, size, overlap int) []scanWindow {
	size = max(size, utf8.UTFMax)
	var windows []scanWindow
	runes := utf8.RuneCountInString(text[:from])
	for start, counted := from, from; start < to; {
		end := to
		if start+size < to {
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
		windows = append(windows, scanWindow{start: start, end: end, runeStart: runes, first: -1, last: -1})
		if end >= to {
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
// middle by bisection, at most maxScanWindows of them. The middle's windows
// come at the input's half, then at its quarters, its eighths and so on, so
// the windows read by any moment are spread over the whole input and no part
// of it waits for the rest; offset (at least 0) turns the points the halving
// starts from, so which window of each part is read first differs from scan
// to scan.
func scanOrder(windows, offset int) []int {
	order := make([]int, 0, windows)
	if windows > 1 {
		order = append(order, windows-1)
	}
	if windows > 0 {
		order = append(order, 0)
	}
	if middle := windows - 2; middle > 0 {
		// Points 0 to 2^depth-1 in bit-reversed order halve the input
		// between the head (point 0) and the tail; each falls in a window.
		depth := bits.Len(uint(middle))
		points := 1 << depth
		turn := offset % points
		taken := make([]bool, windows)
		taken[0], taken[windows-1] = true, true
		for step := range points {
			point := (int(bits.Reverse(uint(step))>>(bits.UintSize-depth)) + turn) % points
			if index := point * (middle + 1) / points; !taken[index] {
				taken[index] = true
				order = append(order, index)
			}
		}
	}
	return order[:min(len(order), maxScanWindows)]
}

// windowCalls are the calls that read scan's cut, in reading order, each
// under a context of its own that a re-plan cancels, and mark the windows
// past maxScanWindows unread.
func (scan *guardScan) windowCalls(ctx context.Context) []*systemOneCall {
	scan.pace.reads = make([]context.Context, scan.cut)
	scan.pace.cancels = make([]context.CancelCauseFunc, scan.cut)
	calls := make([]*systemOneCall, len(scan.order))
	for rank, index := range scan.order {
		window := scan.windows[index]
		request := scan.windowState(scan.text[window.start:window.end])
		request.Window, request.Parallel = rank+1, scan.parallel
		calls[rank] = &systemOneCall{deployment: scan.deployment, request: request, rules: scan.rules, scan: scan, window: index}
		scan.pace.reads[index], scan.pace.cancels[index] = context.WithCancelCause(ctx)
	}
	for index := range scan.cut {
		if scan.pace.reads[index] == nil {
			scan.errs[index], scan.done[index] = errWindowLimit, true
		}
	}
	scan.rank = len(scan.order)
	return calls
}

// readContext is the context window is read under: its own for one of the
// cut, ctx for one that reads part of the input again.
func (scan *guardScan) readContext(ctx context.Context, window int) context.Context {
	if window < scan.cut {
		return scan.pace.reads[window]
	}
	return ctx
}

// scanStopKey keys a request stage's scanStop in its context.
type scanStopKey struct{}

// scanStop ends every guard scan of one request stage: once a window decides
// a rule whose match refuses the request, the windows not yet sent are not
// sent and the stage stops waiting for those in flight.
type scanStop struct {
	ctx  context.Context
	stop context.CancelFunc
	// scans are the stage's scans. Those of one input cut the same windows,
	// so in a bundle each window is one call with all their questions; a
	// scan that re-plans re-plans them all, so they keep sharing calls.
	mu    sync.Mutex
	scans []*guardScan
}

// withScanStop returns a request stage's context with the stop of its guard
// scans, and the function that releases it when the stage ends.
func withScanStop(ctx context.Context) (context.Context, context.CancelFunc) {
	stopped, stop := context.WithCancelCause(context.Background())
	end := func() { stop(context.Canceled) }
	state := &scanStop{ctx: stopped, stop: func() { stop(errScanDecided) }}
	return context.WithValue(ctx, scanStopKey{}, state), end
}

// windowReaders are the work that reads scans' windows, and the function
// that releases what they hold once that work is done. In a stage's bundle
// every window parks its call at once and the bundle sends them, at most
// each scan's parallel bound in flight, the lowest rank first; elsewhere each
// scan has that many readers, which take its windows in reading order. A
// window not started before the deadline, or before a window of the stage
// decided a refusing rule, stays unread. A window read may add windows: the
// smaller ones it is read again in, and the larger ones a scan that fell
// behind reads the rest of its cut in; a reader also reads those another
// scan's re-plan left its scan.
func (c *Classifier) windowReaders(ctx context.Context, scans []*guardScan) ([]func(), func()) {
	ctx, cancel := context.WithCancelCause(ctx)
	unhook := func() bool { return false }
	stop := func() {}
	stage, staged := ctx.Value(scanStopKey{}).(*scanStop)
	if staged {
		unhook = context.AfterFunc(stage.ctx, func() { cancel(context.Cause(stage.ctx)) })
		stop = stage.stop
		stage.mu.Lock()
		stage.scans = append(stage.scans, scans...)
		stage.mu.Unlock()
	}
	started := time.Now()
	deadline, _ := ctx.Deadline()
	release := func() {
		unhook()
		cause := context.Cause(ctx)
		for _, scan := range scans {
			c.observePace(scan, cause)
		}
		cancel(context.Canceled)
	}
	bundled := modelservice.InBundle(ctx)
	var readers []func()
	for _, scan := range scans {
		scan.refusing = c.refusingRules(scan.rules)
		scan.stop = stop
		scan.pace.stage, scan.pace.started, scan.pace.deadline = stage, started, deadline
		calls := scan.windowCalls(ctx)
		// Outside a bundle, slots bound the scan's windows in flight.
		var slots chan struct{}
		if !bundled {
			slots = make(chan struct{}, max(scan.parallel, 1))
		}
		var read func(call *systemOneCall)
		read = func(call *systemOneCall) {
			readCtx := scan.readContext(ctx, call.window)
			var response modelservice.Response
			var latency float64
			err := readCtx.Err()
			if err == nil {
				if slots != nil {
					slots <- struct{}{}
				}
				response, latency, err = c.sendCall(readCtx, call)
				if slots != nil {
					<-slots
				}
			}
			if err != nil {
				err = windowError(readCtx, err)
			}
			scan.record(call.window, response, err, latency)
			more := append(scan.rereadCalls(call, response, err), c.replan(scan)...)
			more = append(more, scan.takePending()...)
			more = append(more, scan.waitingCalls()...)
			modelservice.Fan(ctx, len(more), func(i int) { read(more[i]) })
		}
		if bundled {
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
	return readers, release
}

// windowError is why a window read under ctx has no answer, err as its read
// failed: errWindowReplaced for one a re-plan cancelled.
func windowError(ctx context.Context, err error) error {
	if errors.Is(context.Cause(ctx), errWindowReplaced) {
		return errWindowReplaced
	}
	return err
}

// record keeps a window's answers, and ends the stage's scans when they make
// a refusing rule match.
func (scan *guardScan) record(window int, response modelservice.Response, err error, latency float64) {
	scan.mu.Lock()
	scan.responses[window], scan.errs[window], scan.done[window] = response, err, true
	scan.latency = max(scan.latency, latency)
	scan.mu.Unlock()
	if err != nil {
		return
	}
	for _, asked := range scan.refusing {
		if answer, ok := response.Answers[asked.questionID()]; ok && answer.Error == "" && windowMatches(asked, answer) {
			scan.stop()
			return
		}
	}
}

// rereadCalls are the calls that read a window larger than the configured
// size again in windows of that size, when its answer left a rule that one
// window decides short of a match by at most the margin below its threshold
// (thresholdGap): a larger window dilutes what it holds, which a window of the
// configured size reads as the rule expects. Within half the margin, where a
// diluted match lands, they take the window's rank and are read before the
// windows after it, so a match still ends the scan early; further below, they
// wait until the windows still unread are read (waitingCalls).
func (scan *guardScan) rereadCalls(call *systemOneCall, response modelservice.Response, err error) []*systemOneCall {
	scan.mu.Lock()
	defer scan.mu.Unlock()
	window := scan.windows[call.window]
	if err != nil || scan.rereadMargin <= 0 || window.end-window.start <= scan.size {
		return nil
	}
	gap := math.Inf(1)
	for _, asked := range scan.rules {
		answer, ok := response.Answers[asked.questionID()]
		if !ok || answer.Error != "" || !windowDecides(asked.rule) {
			continue
		}
		if below, short := thresholdGap(asked, answer); short {
			gap = min(gap, below)
		}
	}
	if gap > scan.rereadMargin {
		return nil
	}
	parts := scan.rereadPartsLocked(window)
	if gap > scan.rereadMargin/2 {
		scan.waiting = append(scan.waiting, waitingReread{parts: parts, gap: gap})
		return nil
	}
	calls := make([]*systemOneCall, len(parts))
	for index, part := range parts {
		calls[index] = scan.addWindowLocked(part, call.request.Window)
	}
	return calls
}

// waitingReread is the parts of a window to read again once the windows
// still unread are read (rereadCalls), and how far below its threshold the
// window left its rule.
type waitingReread struct {
	parts []scanWindow
	gap   float64
}

// rereadPartsLocked are the windows of the configured size that read window
// again, but for those the scan's cut already read and answered: the parts of
// a run are the cut's windows it was read in place of, and some of them were
// read before the scan fell behind.
func (scan *guardScan) rereadPartsLocked(window scanWindow) []scanWindow {
	parts := cutWindows(scan.text, window.start, window.end, scan.size, scan.overlap)
	if window.first < 0 || window.first == window.last {
		return parts
	}
	return slices.DeleteFunc(parts, func(part scanWindow) bool {
		for index := window.first; index <= window.last; index++ {
			if cut := scan.windows[index]; cut.start == part.start && cut.end == part.end {
				return scan.done[index] && scan.errs[index] == nil
			}
		}
		return false
	})
}

// waitingCalls are the calls that read again, closest to a match first, the
// windows whose re-read waited, once the cut's windows and the runs are all
// read: a window far below its threshold seldom holds a diluted match, and its
// re-read would take the time the windows still unread need. They rank after
// every window of the scan.
func (scan *guardScan) waitingCalls() []*systemOneCall {
	scan.mu.Lock()
	defer scan.mu.Unlock()
	if len(scan.waiting) == 0 {
		return nil
	}
	for index, window := range scan.windows {
		if window.first >= 0 && !scan.done[index] {
			return nil
		}
	}
	slices.SortStableFunc(scan.waiting, func(a, b waitingReread) int { return cmp.Compare(a.gap, b.gap) })
	var calls []*systemOneCall
	for _, waiting := range scan.waiting {
		for _, part := range waiting.parts {
			scan.rank++
			calls = append(calls, scan.addWindowLocked(part, scan.rank))
		}
	}
	scan.waiting = nil
	return calls
}

// addWindowLocked adds a window that reads part of the input again, and
// returns the call that reads it at rank.
func (scan *guardScan) addWindowLocked(window scanWindow, rank int) *systemOneCall {
	index := len(scan.windows)
	scan.windows = append(scan.windows, window)
	scan.responses = append(scan.responses, modelservice.Response{})
	scan.errs = append(scan.errs, nil)
	scan.done = append(scan.done, false)
	request := scan.windowState(scan.text[window.start:window.end])
	request.Window, request.Parallel = rank, scan.parallel
	return &systemOneCall{deployment: scan.deployment, request: request, rules: scan.rules, scan: scan, window: index}
}

// refusingRules are the rules among rules whose match in one window refuses
// the request: request-stage rules a window decides (windowDecides) that a
// decision refuses on alone (config.RouterConfig.RefusedOnMatch).
func (c *Classifier) refusingRules(rules []askedRule) []askedRule {
	var refusing []askedRule
	for _, asked := range rules {
		if asked.rule.EffectiveStage() == config.SystemOneStageRequest && windowDecides(asked.rule) &&
			c.Config.RefusedOnMatch(asked.signalType, asked.rule.Name) {
			refusing = append(refusing, asked)
		}
	}
	return refusing
}

// windowDecides reports whether one window that makes rule match decides
// it, whatever the windows not read: the scan answers with the most
// alarming window, and every label's highest probability and every span,
// so a match on a floor (gte or gt) or a selection only grows with more
// windows.
func windowDecides(rule config.SystemOneRule) bool {
	predicate := rule.EffectivePredicate()
	if predicate != nil && (predicate.LT != nil || predicate.LTE != nil) {
		return false
	}
	switch rule.Question.Type {
	case config.SystemOneQuestionSet, config.SystemOneQuestionSpan:
		return true
	case config.SystemOneQuestionChoice:
		return predicate != nil && len(rule.Question.Choices) == 2 && len(rule.CountedLabels()) > 0
	}
	return false
}

// windowMatches reports whether one window's answer makes asked's rule
// match.
func windowMatches(asked askedRule, answer modelservice.Answer) bool {
	_, matched := windowVerdict(asked, answer)
	return matched
}

// windowVerdict is the value one window's answer gives asked's rule, which
// its predicate weighs, and whether the answer makes the rule match.
func windowVerdict(asked askedRule, answer modelservice.Answer) (float64, bool) {
	results := newResponseSignalResults()
	matched := applySystemOneAnswer(results, asked.signalType, asked.rule, answer)
	return results.SignalValues[signalConfidenceKey(asked.signalType, asked.rule.Name)], slices.Contains(matched, asked.rule.Name)
}

// thresholdGap is how far below its threshold, the floor its predicate sets,
// one window's answer leaves asked's rule, in log-odds, and false when the
// answer makes the rule match or the rule has no floor (a selection without a
// predicate). Dilution lowers a score's log-odds by about the same amount
// wherever the threshold is, so one margin serves a rule at 0.5 and one at
// 0.995 alike.
func thresholdGap(asked askedRule, answer modelservice.Answer) (float64, bool) {
	predicate := asked.rule.EffectivePredicate()
	if predicate == nil {
		return 0, false
	}
	threshold := predicate.GTE
	if threshold == nil {
		threshold = predicate.GT
	}
	if threshold == nil {
		return 0, false
	}
	value, matched := windowVerdict(asked, answer)
	if matched {
		return 0, false
	}
	return logOdds(*threshold) - logOdds(value), true
}

// logOdds is p's log-odds, with p held just inside (0, 1).
func logOdds(p float64) float64 {
	p = min(max(p, 1e-9), 1-1e-9)
	return math.Log(p / (1 - p))
}

// decidedBy reports whether a read window decides asked's rule: its answer
// makes the rule match, which no window left unread can undo.
func decidedBy(asked askedRule, read []windowAnswer) bool {
	if !windowDecides(asked.rule) {
		return false
	}
	return slices.ContainsFunc(read, func(entry windowAnswer) bool { return windowMatches(asked, entry.answer) })
}

// windowAnswer is one window's answer to a rule's question.
type windowAnswer struct {
	answer modelservice.Answer
	window int
}

// answers are the windows' answers to a rule's question, how many of the
// cut's windows they cover, and why the first window of the cut they leave
// uncovered has no answer.
func (scan *guardScan) answers(asked askedRule) ([]windowAnswer, int, string) {
	return scan.answersTo(asked.questionID())
}

func (scan *guardScan) answersTo(question string) ([]windowAnswer, int, string) {
	var read []windowAnswer
	covered := make([]bool, scan.cut)
	why := make([]string, scan.cut)
	for index, window := range scan.windows {
		answer, ok := scan.responses[index].Answers[question]
		answered, reason := false, ""
		switch err := scan.errs[index]; {
		case errors.Is(err, errWindowReplaced):
			// The window read in its place says why.
		case errors.Is(err, errWindowLimit):
			reason = err.Error()
		case err != nil:
			reason = modelservice.ErrorReason(err)
		case !ok:
			reason = "missing_answer"
		case answer.Error != "":
			reason = answer.Error
		default:
			answered = true
			read = append(read, windowAnswer{answer: answer, window: index})
		}
		for of := window.first; of >= 0 && of <= window.last; of++ {
			covered[of] = covered[of] || answered
			why[of] = cmp.Or(why[of], reason)
		}
	}
	count, reason := 0, ""
	for index := range scan.cut {
		if covered[index] {
			count++
		} else {
			reason = cmp.Or(reason, why[index])
		}
	}
	return read, count, reason
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
		read, covered, reason := scan.answers(asked)
		partial := covered < scan.cut
		decided := (partial || asked.toolResultsUnread) && decidedBy(asked, read)
		c.recordScanCoverage(results, asked, covered, scan.cut)
		switch {
		case len(read) == 0:
			results.SignalErrors[key] = systemOneSignalErrorPrefix + reason
			modelservice.RecordUnknown(scan.deployment, reason, 1)
			c.applyFailClosed(results, asked, scan.deployment, reason)
			continue
		case decided:
			if partial {
				metrics.RecordSignalScanDecided(signalConfidenceKey(asked.signalType, c.scopedSignalName(rule.Name)))
			}
		case partial:
			policy := rule.EffectiveOnUnscanned()
			if policy == config.SystemOneUnscannedBestEffort && float64(covered) < asked.minCoverage*float64(scan.cut) {
				policy = config.SystemOneUnscannedUnknown
			}
			switch policy {
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
		if !decided && asked.toolResultsUnread && c.resolveUnread(results, asked, scan.deployment, unreadToolResult) {
			continue
		}
		c.applyRuleAnswer(results, asked, scanAnswer(rule, read, scan))
	}
}

// recordScanCoverage counts how much of its input a guard rule read and, for
// one read in windows, the share of its windows read, which it records for
// the trace and the replay record. A rule that missed part of its tool
// results read its input in part.
func (c *Classifier) recordScanCoverage(results *SignalResults, asked askedRule, read, windows int) {
	result := scanCoverageFull
	switch {
	case read == 0:
		result = scanCoverageNone
	case read < windows || asked.toolResultsUnread:
		result = scanCoveragePartial
	}
	key := signalConfidenceKey(asked.signalType, asked.rule.Name)
	signal := signalConfidenceKey(asked.signalType, c.scopedSignalName(asked.rule.Name))
	metrics.RecordSignalScanCoverage(signal, result)
	if windows < 2 {
		return
	}
	metrics.RecordSignalScanCoverageRatio(signal, float64(read)/float64(windows))
	if results.ScanCoverage == nil {
		results.ScanCoverage = make(map[string]string)
	}
	if result == scanCoveragePartial && read < windows {
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
// only when its windows covered the text and every one read was read whole.
func recordScanPrivacyEvidence(results *SignalResults, scan *guardScan) {
	if !scan.requestAlone || !slices.ContainsFunc(scan.rules, func(asked askedRule) bool { return asked.preset == presets.PII }) {
		return
	}
	read, covered, _ := scan.answersTo(presetQuestionID(presets.PII))
	complete := covered == scan.cut && !slices.ContainsFunc(read, func(entry windowAnswer) bool { return entry.answer.InputCoverage != "complete" })
	clean := !slices.ContainsFunc(read, func(entry windowAnswer) bool { return len(entry.answer.Spans) > 0 })
	evidence := NewPrivacyEvidence("request", scan.whole.State, complete, complete && clean)
	if !slices.Contains(results.PIIEvidence, evidence) {
		results.PIIEvidence = append(results.PIIEvidence, evidence)
	}
}
