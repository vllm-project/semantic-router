// Package promscrape reads a Prometheus text exposition for backend telemetry
// adapters and sums the series of one metric family.
package promscrape

import (
	"context"
	"fmt"
	"io"
	"net/http"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
)

// MaxBodyBytes bounds one scrape so a misbehaving endpoint cannot grow the
// Router's memory.
const MaxBodyBytes = 8 << 20

// Families are the metric families of one scrape, keyed by name.
type Families map[string]*dto.MetricFamily

// Fetch scrapes url and parses its text exposition.
func Fetch(ctx context.Context, client *http.Client, url string) (Families, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/plain;version=0.0.4")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return nil, fmt.Errorf("GET %s: status %d", url, resp.StatusCode)
	}
	body := io.LimitReader(resp.Body, MaxBodyBytes+1)
	var parser expfmt.TextParser
	families, err := parser.TextToMetricFamilies(&limitCheck{r: body})
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", url, err)
	}
	return families, nil
}

// limitCheck turns reading past MaxBodyBytes into an error instead of a
// silently truncated exposition.
type limitCheck struct {
	r    io.Reader
	read int64
}

func (l *limitCheck) Read(p []byte) (int, error) {
	n, err := l.r.Read(p)
	l.read += int64(n)
	if l.read > MaxBodyBytes {
		return n, fmt.Errorf("metrics body exceeds %d bytes", MaxBodyBytes)
	}
	return n, err
}

// Gauge sums a gauge or untyped family over its series, and also returns how
// many series it summed. The first name present wins, so callers can list a
// metric's current name before its older ones.
func (f Families) Gauge(names ...string) (sum float64, series int, ok bool) {
	family := f.first(names...)
	if family == nil {
		return 0, 0, false
	}
	for _, m := range family.GetMetric() {
		switch {
		case m.GetGauge() != nil:
			sum += m.GetGauge().GetValue()
		case m.GetUntyped() != nil:
			sum += m.GetUntyped().GetValue()
		default:
			continue
		}
		series++
	}
	return sum, series, series > 0
}

// Counter sums a counter or untyped family over its series.
func (f Families) Counter(names ...string) (float64, bool) {
	family := f.first(names...)
	if family == nil {
		return 0, false
	}
	var sum float64
	var series int
	for _, m := range family.GetMetric() {
		switch {
		case m.GetCounter() != nil:
			sum += m.GetCounter().GetValue()
		case m.GetUntyped() != nil:
			sum += m.GetUntyped().GetValue()
		default:
			continue
		}
		series++
	}
	return sum, series > 0
}

// Histogram sums a histogram family's sample sum and count over its series.
func (f Families) Histogram(names ...string) (sum, count float64, ok bool) {
	family := f.first(names...)
	if family == nil {
		return 0, 0, false
	}
	var series int
	for _, m := range family.GetMetric() {
		h := m.GetHistogram()
		if h == nil {
			continue
		}
		sum += h.GetSampleSum()
		count += float64(h.GetSampleCount())
		series++
	}
	return sum, count, series > 0
}

func (f Families) first(names ...string) *dto.MetricFamily {
	for _, name := range names {
		if family, ok := f[name]; ok {
			return family
		}
	}
	return nil
}
