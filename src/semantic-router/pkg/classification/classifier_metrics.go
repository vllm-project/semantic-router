package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) recordSignalExtraction(signalType, signalName string, latencySeconds float64) {
	metrics.RecordSignalExtraction(signalType, c.scopedSignalName(signalName), latencySeconds)
}

func (c *Classifier) recordSignalMatch(signalType, signalName string) {
	metrics.RecordSignalMatch(signalType, c.scopedSignalName(signalName))
}

func (c *Classifier) scopedSignalName(signalName string) string {
	if c == nil || c.Config == nil {
		return signalName
	}
	return config.RoutingNamespaceKey(c.Config.RoutingScope, signalName)
}
