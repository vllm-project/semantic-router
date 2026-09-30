package classification

import (
	"fmt"
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func conversationLatencySum(t *testing.T) float64 {
	t.Helper()
	observer := metrics.SignalExtractionLatency.WithLabelValues(config.SignalTypeConversation)
	histogram, ok := observer.(prometheus.Histogram)
	if !ok {
		t.Fatalf("conversation extraction latency is not a histogram")
	}
	var m dto.Metric
	if err := histogram.Write(&m); err != nil {
		t.Fatalf("histogram Write: %v", err)
	}
	if m.Histogram == nil {
		t.Fatalf("histogram payload missing")
	}
	return m.Histogram.GetSampleSum()
}

// Conversation rules resolve one after another in one goroutine, so the
// reported latencies describe single rules only when they do not carry the
// rules before them. Their sum then stays inside the signal's own execution
// time. A start shared across the loop makes that sum grow with the square of
// the rule count instead.
func TestConversationSignalReportsPerRuleExtractionLatency(t *testing.T) {
	const ruleCount = 12
	cfg := &config.RouterConfig{}
	used := make(map[string]bool, ruleCount)
	for i := 0; i < ruleCount; i++ {
		name := fmt.Sprintf("per_rule_latency_%02d", i)
		cfg.ConversationRules = append(cfg.ConversationRules, config.ConversationRule{
			Name: name,
			Feature: config.ConversationFeature{
				Type:   "exists",
				Source: config.ConversationSource{Type: "tool_definition"},
			},
		})
		used[config.SignalTypeConversation+":"+name] = true
	}
	classifier := &Classifier{Config: cfg}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}

	before := conversationLatencySum(t)
	var mu sync.Mutex
	classifier.evaluateConversationSignal(results, &mu, ConversationFacts{ToolDefinitionCount: 2}, used)
	reported := conversationLatencySum(t) - before

	total := results.Metrics.Conversation.ExecutionTimeMs / 1000
	if reported > total {
		t.Fatalf(
			"reported %v seconds across %d rules against a signal that ran for %v seconds, so a rule carries the rules before it",
			reported, ruleCount, total,
		)
	}
}
