package classification

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRunSignalDispatchersTracksOnlyStartedEvaluators(t *testing.T) {
	var calls atomic.Int32
	executed := map[string]bool{}
	dispatchers := []signalDispatch{
		{signalType: config.SignalTypeKeyword, name: "Keyword", evaluate: func() { calls.Add(1) }},
		{signalType: config.SignalTypeDomain, name: "Domain", evaluate: func() { calls.Add(1) }},
		{signalType: config.SignalTypeEvent, name: "Event", evaluate: func() { calls.Add(1) }},
	}
	used := map[string]bool{
		config.SignalTypeKeyword + ":used": true,
		config.SignalTypeDomain + ":used":  true,
	}
	ready := map[string]bool{
		config.SignalTypeKeyword: true,
		config.SignalTypeDomain:  false,
		config.SignalTypeEvent:   true,
	}
	var wg sync.WaitGroup

	runSignalDispatchers(dispatchers, used, ready, executed, &wg)
	wg.Wait()

	require.Equal(t, int32(1), calls.Load())
	require.Equal(t, map[string]bool{config.SignalTypeKeyword: true}, executed)
}
