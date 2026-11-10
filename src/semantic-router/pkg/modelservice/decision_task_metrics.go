package modelservice

import (
	"math"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	taskCalls    = promauto.NewCounterVec(prometheus.CounterOpts{Name: "vsr_systemone_task_calls_total", Help: "Semantic judgment task calls by actual implementation and outcome."}, []string{"deployment", "task", "stage", "implementation", "outcome"})
	taskDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{Name: "vsr_systemone_task_duration_seconds", Help: "Semantic task latency including the caller's queue, fusion and inference wait.", Buckets: []float64{.001, .005, .01, .05, .1, .25, .5, 1, 2.5, 5, 10, 30}}, []string{"deployment", "task", "stage", "implementation"})
	taskResults  = promauto.NewCounterVec(prometheus.CounterOpts{Name: "vsr_systemone_task_results_total", Help: "Successful judgment output buckets; these are observations, not quality measurements."}, []string{"deployment", "task", "stage", "implementation", "result"})
)

func recordTaskResult(deployment string, plan TaskPlan, result TaskResult, elapsed time.Duration) {
	labels := []string{deployment, plan.Definition.ID, plan.Definition.Stage, plan.Implementation}
	taskCalls.WithLabelValues(append(labels, result.Status)...).Inc()
	taskDuration.WithLabelValues(labels...).Observe(elapsed.Seconds())
	if result.Status != "ok" {
		return
	}
	value := ""
	switch result.Answer.Type {
	case "choice":
		for _, option := range plan.Question.Choices {
			if result.Answer.Choice == option.Key {
				value = "choice:" + option.Key
				break
			}
		}
	case "noul":
		threshold := .5
		if plan.Question.Threshold != nil {
			threshold = *plan.Question.Threshold
		}
		value = "negative"
		if result.Answer.Noul >= threshold {
			value = "positive"
		}
	case "score":
		if !math.IsNaN(result.Answer.Score) && !math.IsInf(result.Answer.Score, 0) && result.Answer.Score >= 0 && result.Answer.Score <= float64(len(plan.Question.Levels)-1) {
			value = "level:" + strconv.Itoa(int(math.Round(result.Answer.Score)))
		}
	case "set", "span":
		value = "absent"
		if len(result.Answer.Selected) > 0 || len(result.Answer.Spans) > 0 {
			value = "present"
		}
	}
	if value != "" {
		taskResults.WithLabelValues(append(labels, value)...).Inc()
	}
}
