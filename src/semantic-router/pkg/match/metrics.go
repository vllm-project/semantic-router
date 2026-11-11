package match

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	recallTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "vsr_match_recall_total",
		Help: "Match recall stages by feature, method (embedding, bm25) and outcome (ok, empty, failed).",
	}, []string{"feature", "method", "outcome"})
	judgeTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "vsr_match_judge_total",
		Help: "Match judge stages by feature, preset (question for a custom one) and outcome (ok, empty, skipped, failed).",
	}, []string{"feature", "preset", "outcome"})
	truncatedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "vsr_match_judge_truncated_total",
		Help: "Match judges that read only their first max_candidates candidates, by feature and preset (question for a custom one).",
	}, []string{"feature", "preset"})
	stageDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "vsr_match_stage_duration_seconds",
		Help:    "Latency of match stages by feature and stage (recall, judge); a judge without candidates is not timed.",
		Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5},
	}, []string{"feature", "stage"})
)

func observe(feature config.MatchFeature, stage Stage) {
	counter := judgeTotal
	if stage.Name == StageRecall {
		counter = recallTotal
	}
	counter.WithLabelValues(string(feature), stage.Method, stage.Outcome).Inc()
	if stage.Name == StageRecall || stage.Outcome != OutcomeEmpty {
		stageDuration.WithLabelValues(string(feature), stage.Name).Observe(stage.Duration.Seconds())
	}
}
