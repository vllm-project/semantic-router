package upstream

import (
	"math"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Envoy's outlier-detection defaults for the fields the template omits.
const (
	defaultOutlierInterval          = 10 * time.Second
	defaultBaseEjectionTime         = 30 * time.Second
	defaultMaxEjectionTime          = 300 * time.Second
	defaultMaxEjectionPercent       = 50
	defaultSuccessRateMinimumHosts  = 5
	defaultSuccessRateRequestVolume = 100
	defaultSuccessRateStdevFactor   = 1.9
)

// Ejection reasons, as metric labels.
const (
	ejectConsecutive5xx = "consecutive_5xx"
	ejectSuccessRate    = "success_rate"
)

func (o OutlierSpec) withDefaults() OutlierSpec {
	o.Interval = pickDuration(o.Interval, defaultOutlierInterval)
	o.BaseEjectionTime = pickDuration(o.BaseEjectionTime, defaultBaseEjectionTime)
	o.MaxEjectionTime = max(pickDuration(o.MaxEjectionTime, defaultMaxEjectionTime), o.BaseEjectionTime)
	o.MaxEjectionPercent = pickInt(o.MaxEjectionPercent, defaultMaxEjectionPercent)
	o.SuccessRateMinimumHosts = pickInt(o.SuccessRateMinimumHosts, defaultSuccessRateMinimumHosts)
	o.SuccessRateRequestVolume = pickInt(o.SuccessRateRequestVolume, defaultSuccessRateRequestVolume)
	if o.SuccessRateStdevFactor <= 0 {
		o.SuccessRateStdevFactor = defaultSuccessRateStdevFactor
	}
	return o
}

// outlierState is one endpoint's passive-health record, kept the way Envoy's
// outlier detector keeps it. The cluster's mutex guards it.
type outlierState struct {
	consecutive5xx int
	// current fills during a sweep interval; the sweep moves it to last,
	// which success-rate ejection reads.
	current, last requestTally
	ejected       bool
	// backoff multiplies the base ejection time. It grows with each ejection
	// and shrinks by one per interval the endpoint stays in.
	backoff      int
	lastEjection time.Time
	lastUneject  time.Time
}

type requestTally struct{ total, success int }

// outlierDetector ejects a cluster's misbehaving endpoints.
type outlierDetector struct {
	spec  OutlierSpec
	timer Timer
}

// observeLocked feeds one result into the detector. code is the response
// status, or the status Envoy records for a local failure: 503 for a failed
// or reset connection and 504 for a timeout.
func (c *cluster) observeLocked(ep *endpoint, code int) {
	o := &ep.outlier
	o.current.total++
	if code < 500 {
		o.current.success++
		o.consecutive5xx = 0
		return
	}
	o.consecutive5xx++
	if o.consecutive5xx != c.outlier.spec.Consecutive5xx || o.ejected {
		return
	}
	c.ejectLocked(ep, ejectConsecutive5xx)
	o.consecutive5xx = 0
}

// ejectLocked takes an endpoint out of rotation unless that would exceed the
// cluster's maximum ejection percentage.
func (c *cluster) ejectLocked(ep *endpoint, reason string) {
	spec := c.outlier.spec
	ejected := 0
	for _, other := range c.endpoints {
		if other.outlier.ejected {
			ejected++
		}
	}
	if 100*float64(ejected+1)/float64(len(c.endpoints)) > float64(spec.MaxEjectionPercent) {
		return
	}
	o := &ep.outlier
	o.ejected = true
	o.lastEjection = c.clock.Now()
	if time.Duration(o.backoff)*spec.BaseEjectionTime < spec.MaxEjectionTime+spec.BaseEjectionTime {
		o.backoff++
	}
	metrics.RecordUpstreamEjection(c.spec.Name, ep.spec.Name, reason)
	c.refreshHostsLocked()
}

func (c *cluster) unejectLocked(ep *endpoint) {
	o := &ep.outlier
	o.ejected = false
	o.consecutive5xx = 0
	o.lastUneject = c.clock.Now()
	c.refreshHostsLocked()
}

// sweep is the detector's periodic pass: it returns endpoints whose ejection
// time has passed, closes the success-rate interval, ejects success-rate
// outliers, and shrinks the back-off of endpoints that stayed in.
func (c *cluster) sweep() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	spec := c.outlier.spec
	now := c.clock.Now()
	for _, ep := range c.endpoints {
		o := &ep.outlier
		ejectionTime := min(time.Duration(o.backoff)*spec.BaseEjectionTime, spec.MaxEjectionTime)
		if o.ejected && now.Sub(o.lastEjection) >= ejectionTime {
			c.unejectLocked(ep)
		}
		o.last, o.current = o.current, requestTally{}
	}
	c.successRateEjectionsLocked()
	for _, ep := range c.endpoints {
		o := &ep.outlier
		if !o.ejected && !o.lastUneject.IsZero() && now.Sub(o.lastUneject) >= spec.Interval && o.backoff > 0 {
			o.backoff--
		}
	}
	c.outlier.timer = c.clock.AfterFunc(spec.Interval, c.sweep)
}

// successRateEjectionsLocked ejects endpoints whose success rate over the
// last interval falls below mean - factor * stdev, once enough endpoints saw
// enough requests to judge.
func (c *cluster) successRateEjectionsLocked() {
	spec := c.outlier.spec
	if len(c.endpoints) < spec.SuccessRateMinimumHosts {
		return
	}
	var judged []*endpoint
	var rates []float64
	var sum float64
	for _, ep := range c.endpoints {
		last := ep.outlier.last
		if ep.outlier.ejected || last.total == 0 || last.total < spec.SuccessRateRequestVolume {
			continue
		}
		rate := 100 * float64(last.success) / float64(last.total)
		judged, rates, sum = append(judged, ep), append(rates, rate), sum+rate
	}
	if len(judged) < spec.SuccessRateMinimumHosts {
		return
	}
	mean := sum / float64(len(rates))
	var variance float64
	for _, rate := range rates {
		variance += (rate - mean) * (rate - mean)
	}
	threshold := mean - spec.SuccessRateStdevFactor*math.Sqrt(variance/float64(len(rates)))
	for i, ep := range judged {
		if rates[i] < threshold {
			c.ejectLocked(ep, ejectSuccessRate)
		}
	}
}
