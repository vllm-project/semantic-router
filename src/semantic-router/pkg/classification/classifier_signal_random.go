package classification

import (
	crand "crypto/rand"
	"math/big"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateRandomSignal(results *SignalResults, mu *sync.Mutex) {
	start := time.Now()
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	mu.Lock()
	defer mu.Unlock()

	results.Metrics.Random.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Random.Confidence = 1.0

	for _, rule := range c.Config.RandomRules {
		value := randomDigit()
		key := signalConfidenceKey(config.SignalTypeRandom, rule.Name)
		results.MatchedRandomRules = append(results.MatchedRandomRules, rule.Name)
		results.SignalConfidences[key] = 1.0
		results.SignalValues[key] = float64(value)
		metrics.RecordSignalExtraction(config.SignalTypeRandom, rule.Name, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeRandom, rule.Name)
	}

	logging.Debugf("[Signal Computation] Random signal evaluation completed in %v", elapsed)
}

func randomDigit() int64 {
	n, err := crand.Int(crand.Reader, big.NewInt(10))
	if err != nil {
		return time.Now().UnixNano() % 10
	}
	return n.Int64()
}
