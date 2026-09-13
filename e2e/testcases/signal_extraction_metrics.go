package testcases

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

// signalExtractionMetric counts how often each declared rule was extracted,
// labelled by signal type and rule name. It is the boundary evidence that a
// response-stage rule ran: a streamed response is observed but never enforced,
// so no header and no body change reports it.
const signalExtractionMetric = "llm_signal_extraction_total"

// readSignalExtractionCount scrapes the router's Prometheus endpoint and
// returns how often the named rule has been extracted so far.
func readSignalExtractionCount(
	ctx context.Context,
	metricsSession *fixtures.ServiceSession,
	signalType string,
	signalName string,
) (float64, error) {
	body, err := fetchMetrics(ctx, metricsSession)
	if err != nil {
		return 0, err
	}
	return signalExtractionCount(body, signalType, signalName), nil
}

// signalExtractionCount reads one rule's counter out of the Prometheus text
// exposition. A rule that has never been extracted has no series at all, which
// reads as zero.
func signalExtractionCount(metricsText, signalType, signalName string) float64 {
	typeLabel := fmt.Sprintf("signal_type=%q", signalType)
	nameLabel := fmt.Sprintf("signal_name=%q", signalName)
	for _, line := range strings.Split(metricsText, "\n") {
		if !strings.HasPrefix(line, signalExtractionMetric+"{") {
			continue
		}
		if !strings.Contains(line, typeLabel) || !strings.Contains(line, nameLabel) {
			continue
		}
		fields := strings.Fields(line)
		value, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err != nil {
			continue
		}
		return value
	}
	return 0
}
