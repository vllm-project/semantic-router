package classification

import (
	"context"
	"errors"
	"sync"
	"time"
	"unicode/utf8"
)

// A guard scan's windows are sized to the time it has. Each deployment's
// pace, how long it takes per window read with its max_parallel windows in
// flight, is learned from the scans it served: an exponentially weighted line
// through their time per window read against their window size. A scan whose
// windows of the configured size would not all be read by the signals'
// deadline at that pace reads larger ones instead, the smallest that fit, up
// to the largest the deployment allows: every window read at a larger size
// is worth more than a fraction of them read at the configured one.

const (
	// scanReadShare is the share of the time left that a scan's windows are
	// sized to be read in; the rest absorbs the pace's variance and the
	// stage's other work.
	scanReadShare = 0.8
	// scanPaceDecay is the weight a deployment's earlier scans keep at each
	// new one.
	scanPaceDecay = 0.9
	// scanCallBytes is what one call costs besides its text, as bytes of
	// text read: a decision model reads 1K tokens in about 0.43 s and 30K in
	// about 4.5 s, so a call's own cost is about that of 2,048 tokens of
	// prose. It shapes a deployment's line until its scans read windows of
	// two sizes.
	scanCallBytes = 8192
)

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
// seconds, and false before it served a scan. Scans of one window size
// cannot split the two, so their time shares out as scanCallBytes says.
func (p *windowPaces) line(deployment string) (perCall, perByte float64, ok bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	pace := p.paces[deployment]
	if pace == nil || pace.weight == 0 {
		return 0, 0, false
	}
	size, seconds := pace.size/pace.weight, pace.seconds/pace.weight
	spread := pace.sizeSquared/pace.weight - size*size
	if spread <= 0.0025*size*size {
		perByte = seconds / (size + scanCallBytes)
		return perByte * scanCallBytes, perByte, true
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

// scanTime is the time a scan started under ctx sizes its windows to be
// read in: a share of what the signals' deadline leaves, or 0 without one.
func scanTime(ctx context.Context) time.Duration {
	deadline, ok := ctx.Deadline()
	if !ok {
		return 0
	}
	return time.Duration(float64(time.Until(deadline)) * scanReadShare)
}

// scanBudgetFor is deployment's scan window budget with the time a scan
// started under ctx has and the deployment's pace; a request stage's scans of
// one deployment all take the budget of its first.
func (c *Classifier) scanBudgetFor(ctx context.Context, deployment string) scanBudget {
	stage, staged := ctx.Value(scanStopKey{}).(*scanStop)
	if staged {
		stage.mu.Lock()
		defer stage.mu.Unlock()
		if budget, ok := stage.budgets[deployment]; ok {
			return budget
		}
	}
	budget := c.scanWindowBudget(deployment)
	if readTime := scanTime(ctx); readTime > 0 {
		if perCall, perByte, ok := c.windowPaces.line(deployment); ok {
			budget.readTime, budget.perCall, budget.perByte = readTime, perCall, perByte
		}
	}
	if staged {
		stage.budgets[deployment] = budget
	}
	return budget
}

// pacedSize is the smallest window size from size to most at which the
// budget's pace reads all windows of a text of length bytes in its time
// (windows of them at size), most when none does, or size without a pace.
func (budget scanBudget) pacedSize(length, size, most, overlap, windows int) int {
	if budget.readTime <= 0 || most <= size {
		return size
	}
	limit := budget.readTime.Seconds()
	fits := func(window, count int) bool {
		return float64(count)*(budget.perCall+budget.perByte*float64(window)) <= limit
	}
	if fits(size, windows) {
		return size
	}
	if !fits(most, windowCount(length, most, overlap)) {
		return most
	}
	low, high := size+1, most
	for low < high {
		middle := low + (high-low)/2
		if fits(middle, windowCount(length, middle, overlap)) {
			high = middle
		} else {
			low = middle + 1
		}
	}
	return low
}

// windowCount is at most how many windows of size bytes, overlapping by
// overlap, cut a text of length bytes: each cut moves back at most to a
// character and a word boundary.
func windowCount(length, size, overlap int) int {
	if length <= size {
		return 1
	}
	stride := size - overlap - routingViewSnap - utf8.UTFMax
	if stride <= 0 {
		return length
	}
	return (length-size+stride-1)/stride + 1
}

// observePace records how long a scan took per window it read, elapsed
// over its windows read, unless its caller gave up on it or the runtime
// turned every window away: neither says how fast the deployment reads. A
// scan the deadline cut before any window was read took at least elapsed
// for one.
func (c *Classifier) observePace(scan *guardScan, elapsed time.Duration, cause error) {
	if cause != nil && !errors.Is(cause, errScanDecided) && !errors.Is(cause, context.DeadlineExceeded) {
		return
	}
	read, cut := 0, 0
	for _, err := range scan.errs {
		switch {
		case err == nil:
			read++
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			cut++
		}
	}
	if read == 0 && cut == 0 {
		return
	}
	c.windowPaces.observe(scan.deployment, scan.size, elapsed/time.Duration(max(read, 1)))
}
