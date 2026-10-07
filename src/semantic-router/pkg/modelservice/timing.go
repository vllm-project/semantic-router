package modelservice

import (
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// The runtime answers every surface and bundle request with a Server-Timing
// header: its own time for the request by phase, and their total, in
// milliseconds. The Router times each HTTP exchange around the generated
// client, so its encoding and decoding count, and records for every call the
// exchange carried what lay outside the runtime (the transport) and the
// runtime's phases. A bundled call records its bundle's exchange.

// serverPhases are the runtime's phases in header order, then "other": the
// part of its total that no phase covers.
var serverPhases = [...]string{"parse", "tokenize", "queue", "forward", "post", "serialize", "other"}

const otherPhase = len(serverPhases) - 1

// maxServerMillis bounds a credible Server-Timing duration (a day).
const maxServerMillis = float64(24 * time.Hour / time.Millisecond)

// exchangeTiming is one runtime exchange as the Router timed it. reported is
// false when the response carried no Server-Timing total.
type exchangeTiming struct {
	elapsed  time.Duration
	total    time.Duration
	phases   [otherPhase]time.Duration
	reported bool
}

// timedExchange is the timing of a generated client call that started at
// started and returned response.
func timedExchange(started time.Time, response *http.Response) exchangeTiming {
	timing := exchangeTiming{elapsed: time.Since(started)}
	if response != nil {
		for _, value := range response.Header.Values("Server-Timing") {
			timing.read(value)
		}
	}
	return timing
}

// read takes the phases and total from one Server-Timing value, such as
// "parse;dur=0.021, tokenize;dur=0.153, ..., total;dur=5.141". Metrics it
// does not know, and entries without a valid dur, are skipped.
func (t *exchangeTiming) read(value string) {
	for value != "" {
		var entry string
		entry, value, _ = strings.Cut(value, ",")
		name, params, _ := strings.Cut(entry, ";")
		duration, ok := serverTimingDuration(params)
		if !ok {
			continue
		}
		name = strings.TrimSpace(name)
		if name == "total" {
			t.total, t.reported = duration, true
			continue
		}
		for index, phase := range serverPhases[:otherPhase] {
			if name == phase {
				t.phases[index] = duration
			}
		}
	}
}

// serverTimingDuration is the dur parameter of a Server-Timing entry's
// parameters (";"-separated, after the metric name).
func serverTimingDuration(params string) (time.Duration, bool) {
	for params != "" {
		var param string
		param, params, _ = strings.Cut(params, ";")
		key, value, _ := strings.Cut(param, "=")
		if !strings.EqualFold(strings.TrimSpace(key), "dur") {
			continue
		}
		millis, err := strconv.ParseFloat(strings.Trim(strings.TrimSpace(value), `"`), 64)
		if err != nil || math.IsNaN(millis) || millis < 0 || millis > maxServerMillis {
			return 0, false
		}
		return time.Duration(millis * float64(time.Millisecond)), true
	}
	return 0, false
}

// transport is the exchange's time outside the runtime.
func (t exchangeTiming) transport() time.Duration {
	return max(t.elapsed-t.total, 0)
}

// record adds a call's exchange to the transport and server metrics, once the
// runtime has reported its time.
func (t exchangeTiming) record(deployment, surface string) {
	if !t.reported {
		return
	}
	transportDuration.WithLabelValues(deployment, surface).Observe(t.transport().Seconds())
	other := t.total
	for index, duration := range t.phases {
		serverDuration.WithLabelValues(deployment, surface, serverPhases[index]).Observe(duration.Seconds())
		other -= duration
	}
	serverDuration.WithLabelValues(deployment, surface, serverPhases[otherPhase]).Observe(max(other, 0).Seconds())
}
