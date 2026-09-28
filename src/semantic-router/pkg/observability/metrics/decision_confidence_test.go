package metrics

import (
	"math"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
)

func TestDecisionMatchOnlyObservesReportedFiniteConfidence(t *testing.T) {
	for _, test := range []struct {
		name   string
		score  float64
		scored bool
		count  uint64
	}{
		{name: "unscored structural constant", score: 1},
		{name: "unscored zero", score: 0},
		{name: "reported zero", score: 0, scored: true, count: 1},
		{name: "reported score", score: 0.6, scored: true, count: 1},
		{name: "nan", score: math.NaN(), scored: true},
		{name: "positive infinity", score: math.Inf(1), scored: true},
		{name: "negative infinity", score: math.Inf(-1), scored: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			name := t.Name()
			counter := DecisionMatchTotal.WithLabelValues(name)
			before := testutil.ToFloat64(counter)
			RecordDecisionMatch(name, test.score, test.scored)
			if got := testutil.ToFloat64(counter) - before; got != 1 {
				t.Fatalf("match counter delta = %g, want 1", got)
			}
			var metric dto.Metric
			if err := DecisionConfidence.WithLabelValues(name).(prometheus.Metric).Write(&metric); err != nil {
				t.Fatal(err)
			}
			histogram := metric.GetHistogram()
			if histogram.GetSampleCount() != test.count || !isFiniteConfidenceSum(histogram.GetSampleSum()) {
				t.Fatalf("confidence histogram count/sum = %d/%g", histogram.GetSampleCount(), histogram.GetSampleSum())
			}
			if test.count == 1 && histogram.GetSampleSum() != test.score {
				t.Fatalf("reported score = %g, want %g", histogram.GetSampleSum(), test.score)
			}
		})
	}
}

func isFiniteConfidenceSum(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
