package classification

import (
	"context"
	"errors"
	"slices"
	"sync"
	"time"
)

// A guard scan reads its windows at the configured size, so an input whose
// windows are all read by the signals' deadline is read as it would be
// without one. A scan that falls behind, whose windows left would not all be
// read by the deadline at the pace it read the others, reads them in fewer,
// larger windows instead when the deployment's pace says those would be:
// runs of its cut's windows, each read as one, the shortest runs that fit.
// The head and the tail keep the configured size. A deployment's pace, how
// long it takes per window read with its max_parallel windows in flight, is
// learned from the scans it served: an exponentially weighted line through
// their time per window read against their window size. A decision model
// reads each token at about the same cost whatever the window, and the
// windows in flight share a call's own cost, so larger windows save little
// time and dilute what they hold: a scan grows its windows only when that
// makes the rest of them fit.

const (
	// scanReadShare is the share of the time left that a scan that fell
	// behind sizes its larger windows to be read in; the rest absorbs the
	// pace's variance and the stage's other work.
	scanReadShare = 0.8
	// scanPaceDecay is the weight a deployment's earlier scans keep at each
	// new one.
	scanPaceDecay = 0.9
	// scanCallBytes is what one call costs besides its text, as bytes of
	// text read: a decision model reads 1K tokens in about 0.43 s and 30K in
	// about 4.5 s, so a call's own cost is about that of 2,048 tokens of
	// prose. The windows in flight share it; they do not share the cost of
	// their text. It shapes a deployment's line until its scans read windows
	// of two sizes.
	scanCallBytes = 8192
	// scanPaceRounds is how many rounds of its windows in flight a scan reads
	// before it judges its own pace, so the time its first answers took to
	// come weighs little.
	scanPaceRounds = 4
)

// scanPace is what a scan knows of its pace: its stage, when its reads
// started and when its deadline is, the context each window of its cut is
// read under and the function that cancels it, whether it judged that it
// fell behind, and once it re-planned, when, the bytes of its larger windows
// and the calls that read them that its readers have yet to take; and when
// its cut's windows and runs were all read, once re-reads waited for them.
type scanPace struct {
	stage             *scanStop
	started, deadline time.Time
	reads             []context.Context
	cancels           []context.CancelCauseFunc
	judged            bool
	at                time.Time
	size              int
	pending           []*systemOneCall
	cutRead           time.Time
}

// windowPace is one deployment's learned pace: the decayed weight of its
// scans and the decayed sums of their window sizes and times per window.
type windowPace struct {
	weight, size, seconds, sizeSquared, sizeSeconds float64
}

// windowPaces are the deployments' paces.
type windowPaces struct {
	mu    sync.Mutex
	paces map[string]*windowPace
}

// observe records a scan of deployment's windows of size bytes that read
// one window every perWindow.
func (p *windowPaces) observe(deployment string, size int, perWindow time.Duration) {
	if size <= 0 || perWindow <= 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.paces == nil {
		p.paces = map[string]*windowPace{}
	}
	pace := p.paces[deployment]
	if pace == nil {
		pace = &windowPace{}
		p.paces[deployment] = pace
	}
	x, y := float64(size), perWindow.Seconds()
	pace.weight = pace.weight*scanPaceDecay + 1
	pace.size = pace.size*scanPaceDecay + x
	pace.seconds = pace.seconds*scanPaceDecay + y
	pace.sizeSquared = pace.sizeSquared*scanPaceDecay + x*x
	pace.sizeSeconds = pace.sizeSeconds*scanPaceDecay + x*y
}

// line is deployment's time per window read as perCall + perByte*size
// seconds with parallel windows in flight, and false before it served a
// scan. Scans of one window size cannot split the two, so their time shares
// out as scanCallBytes says, a call's own part shared by the windows in
// flight.
func (p *windowPaces) line(deployment string, parallel int) (perCall, perByte float64, ok bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	pace := p.paces[deployment]
	if pace == nil || pace.weight == 0 {
		return 0, 0, false
	}
	size, seconds := pace.size/pace.weight, pace.seconds/pace.weight
	spread := pace.sizeSquared/pace.weight - size*size
	if spread <= 0.0025*size*size {
		callBytes := float64(scanCallBytes) / float64(max(parallel, 1))
		perByte = seconds / (size + callBytes)
		return perByte * callBytes, perByte, true
	}
	perByte = (pace.sizeSeconds/pace.weight - size*seconds) / spread
	perCall = seconds - perByte*size
	switch {
	case perByte < 0:
		return seconds, 0, true
	case perCall < 0:
		return 0, seconds / size, true
	}
	return perCall, perByte, true
}

// cost is how long deployment takes per window of size bytes read relative
// to one of base bytes, with parallel windows in flight: at its line, or as
// scanCallBytes says before it served a scan.
func (p *windowPaces) cost(deployment string, parallel, size, base int) float64 {
	perCall, perByte, ok := p.line(deployment, parallel)
	if !ok {
		perCall, perByte = float64(scanCallBytes)/float64(max(parallel, 1)), 1
	}
	return (perCall + perByte*float64(size)) / (perCall + perByte*float64(base))
}

// replan returns the calls that read the windows a scan has left in larger
// windows, once it fell behind: at the pace it read its cut so far, the
// windows not yet read would not all be read by its deadline, but runs of k
// of them read as one would be, read in scanReadShare of the time left at
// the deployment's line, for the smallest k that fits. A scan judges once,
// and keeps its windows when no k fits. Every scan of the input in the
// request stage re-plans with it: their calls for one window carry each
// other's questions, and they keep doing so (replanLocked).
func (c *Classifier) replan(scan *guardScan) []*systemOneCall {
	k := c.replanRuns(scan)
	if k < 2 {
		return nil
	}
	var own []*systemOneCall
	for _, peer := range scan.peers() {
		peer.mu.Lock()
		if peer.pace.at.IsZero() {
			calls := peer.replanLocked(k)
			if peer == scan {
				own = calls
			} else {
				peer.pace.pending = append(peer.pace.pending, calls...)
			}
		}
		peer.mu.Unlock()
	}
	return own
}

// replanRuns is how many of a scan's windows left it reads as one, or 0
// when it keeps them: before it read scanPaceRounds rounds of its windows
// in flight, when it is not behind, when no k fits, once it judged or
// another scan of its input re-planned it, and once its stage's scans ended.
func (c *Classifier) replanRuns(scan *guardScan) int {
	scan.mu.Lock()
	defer scan.mu.Unlock()
	pace := &scan.pace
	if pace.judged || !pace.at.IsZero() || pace.deadline.IsZero() || scan.most <= scan.grown ||
		pace.stage != nil && pace.stage.ctx.Err() != nil {
		return 0
	}
	read, left := 0, 0
	for index := range scan.cut {
		switch {
		case !scan.done[index]:
			left++
		case scan.errs[index] == nil:
			read++
		}
	}
	if left == 0 || read < scanPaceRounds*max(scan.parallel, 1) {
		return 0
	}
	now := time.Now()
	perWindow, timeLeft := now.Sub(pace.started)/time.Duration(read), pace.deadline.Sub(now)
	if time.Duration(left)*perWindow <= timeLeft {
		return 0
	}
	pace.judged = true
	stride := scan.grown - scan.overlap
	for k := 2; k*stride+scan.overlap <= scan.most; k++ {
		need := float64(len(scan.runsLocked(k))) * float64(perWindow) * c.windowPaces.cost(scan.deployment, scan.parallel, k*stride+scan.overlap, scan.grown)
		if need <= scanReadShare*float64(timeLeft) {
			c.windowPaces.observe(scan.deployment, scan.grown, perWindow)
			return k
		}
	}
	return 0
}

// replanLocked cancels the scan's windows left and returns the calls that
// read them in runs of k (runsLocked), which take the windows' places in the
// reading order, spread over the input as the cut was.
func (scan *guardScan) replanLocked(k int) []*systemOneCall {
	pace := &scan.pace
	pace.judged, pace.at, pace.size = true, time.Now(), k*(scan.grown-scan.overlap)+scan.overlap
	runs := scan.runsLocked(k)
	rank := 0
	for position, index := range scan.order {
		if !scan.done[index] {
			rank = position + 1
			break
		}
	}
	calls := make([]*systemOneCall, 0, len(runs))
	for position, index := range scanOrder(len(runs), scan.turn) {
		run := runs[index]
		for replaced := run.first; replaced <= run.last; replaced++ {
			if !scan.done[replaced] {
				pace.cancels[replaced](errWindowReplaced)
			}
		}
		calls = append(calls, scan.addWindowLocked(run, rank+position))
		scan.rank = max(scan.rank, rank+position)
	}
	return calls
}

// runsLocked are the windows that read the scan's cut in runs of k, one
// for each multiple of k that holds a window not yet read, the head and the
// tail left out: the same runs of one input in every scan, whichever windows
// each read already.
func (scan *guardScan) runsLocked(k int) []scanWindow {
	var runs []scanWindow
	for from := 0; from < scan.cut; from += k {
		first, last := max(from, 1), min(from+k, scan.cut-1)-1
		if first > last || !slices.Contains(scan.done[first:last+1], false) {
			continue
		}
		runs = append(runs, scanWindow{
			start: scan.windows[first].start, end: scan.windows[last].end, runeStart: scan.windows[first].runeStart, first: first, last: last,
		})
	}
	return runs
}

// takePending returns the calls another scan's re-plan left the scan to
// read.
func (scan *guardScan) takePending() []*systemOneCall {
	scan.mu.Lock()
	defer scan.mu.Unlock()
	calls := scan.pace.pending
	scan.pace.pending = nil
	return calls
}

// peers are the scans of the request stage that read the scan's input on
// its deployment, the scan among them.
func (scan *guardScan) peers() []*guardScan {
	stage := scan.pace.stage
	if stage == nil {
		return []*guardScan{scan}
	}
	stage.mu.Lock()
	defer stage.mu.Unlock()
	var peers []*guardScan
	for _, other := range stage.scans {
		if other == scan || other.deployment == scan.deployment && sameState(other.whole, scan.whole) {
			peers = append(peers, other)
		}
	}
	return peers
}

// observePace records how long a scan took per window it read, over its
// cut's windows read or, once it re-planned, over its larger ones read since,
// unless its caller gave up on it or the runtime turned every window away:
// neither says how fast the deployment reads. A scan the deadline cut before
// any window was read took at least the time it had for one.
func (c *Classifier) observePace(scan *guardScan, cause error) {
	if cause != nil && !errors.Is(cause, errScanDecided) && !errors.Is(cause, context.DeadlineExceeded) {
		return
	}
	pace := &scan.pace
	from, size := pace.started, scan.grown
	if !pace.at.IsZero() {
		from, size = pace.at, pace.size
	}
	read, cut := 0, 0
	for index, window := range scan.windows {
		if window.first < 0 || (index < scan.cut) != pace.at.IsZero() {
			continue
		}
		switch err := scan.errs[index]; {
		case err == nil && scan.done[index]:
			read++
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			cut++
		}
	}
	if read == 0 && cut == 0 {
		return
	}
	// The re-reads that waited for the cut read none of its windows.
	end := time.Now()
	if !pace.cutRead.IsZero() {
		end = pace.cutRead
	}
	c.windowPaces.observe(scan.deployment, size, end.Sub(from)/time.Duration(max(read, 1)))
}
