package looper

import (
	"math"
	"math/rand"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// distributeCallsToModels distributes K calls among models based on strategy
func (l *ReMoMLooper) distributeCallsToModels(cfg *config.ReMoMAlgorithmConfig, numCalls int, modelRefs []config.ModelRef) []ModelCall {
	strategy := cfg.ModelDistribution
	if strategy == "" {
		strategy = remomDistributionWeighted
	}

	switch strategy {
	case remomDistributionWeighted:
		return distributeByWeight(numCalls, modelRefs, cfg.ShuffleSeed)
	case remomDistributionEqual:
		return distributeEqually(numCalls, modelRefs, cfg.ShuffleSeed)
	case remomDistributionRoundRobin:
		return distributeRoundRobin(numCalls, modelRefs)
	case remomDistributionFirstOnly:
		return distributeFirstOnly(numCalls, modelRefs)
	default:
		logging.Warnf("[ReMoM] Unknown distribution strategy %s, using weighted", strategy)
		return distributeByWeight(numCalls, modelRefs, cfg.ShuffleSeed)
	}
}

// distributeByWeight distributes calls proportionally based on model weights
func distributeByWeight(numCalls int, modelRefs []config.ModelRef, seed int) []ModelCall {
	if numCalls <= 0 || len(modelRefs) == 0 {
		return nil
	}

	totalWeight := 0.0
	for _, ref := range modelRefs {
		if ref.Weight > 0 {
			totalWeight += ref.Weight
		}
	}
	if totalWeight == 0 {
		return distributeEqually(numCalls, modelRefs, seed)
	}

	type weightedShare struct {
		index     int
		ref       config.ModelRef
		count     int
		remainder float64
	}

	shares := make([]weightedShare, 0, len(modelRefs))
	assigned := 0
	for i, ref := range modelRefs {
		if ref.Weight <= 0 {
			shares = append(shares, weightedShare{index: i, ref: ref})
			continue
		}
		exact := ref.Weight / totalWeight * float64(numCalls)
		count := int(math.Floor(exact))
		assigned += count
		shares = append(shares, weightedShare{
			index:     i,
			ref:       ref,
			count:     count,
			remainder: exact - float64(count),
		})
	}

	sort.SliceStable(shares, func(i, j int) bool {
		if shares[i].remainder == shares[j].remainder {
			return shares[i].index < shares[j].index
		}
		return shares[i].remainder > shares[j].remainder
	})
	for remaining := numCalls - assigned; remaining > 0; remaining-- {
		shares[(numCalls-assigned-remaining)%len(shares)].count++
	}

	sort.SliceStable(shares, func(i, j int) bool {
		return shares[i].index < shares[j].index
	})

	calls := make([]ModelCall, 0, numCalls)
	for _, share := range shares {
		for range share.count {
			calls = append(calls, modelCallFromRef(share.ref))
		}
	}
	shuffleModelCalls(calls, seed)
	return calls
}

// distributeEqually distributes calls evenly among all models
func distributeEqually(numCalls int, modelRefs []config.ModelRef, seed int) []ModelCall {
	if numCalls <= 0 || len(modelRefs) == 0 {
		return nil
	}

	calls := make([]ModelCall, 0, numCalls)
	callsPerModel := numCalls / len(modelRefs)
	remainder := numCalls % len(modelRefs)

	for i, ref := range modelRefs {
		count := callsPerModel
		if i < remainder {
			count++ // Distribute remainder to first N models
		}

		for j := 0; j < count; j++ {
			calls = append(calls, modelCallFromRef(ref))
		}
	}

	shuffleModelCalls(calls, seed)
	return calls
}

// distributeRoundRobin cycles through models in configured order.
func distributeRoundRobin(numCalls int, modelRefs []config.ModelRef) []ModelCall {
	if numCalls <= 0 || len(modelRefs) == 0 {
		return nil
	}

	calls := make([]ModelCall, 0, numCalls)
	for i := 0; i < numCalls; i++ {
		calls = append(calls, modelCallFromRef(modelRefs[i%len(modelRefs)]))
	}
	return calls
}

// distributeFirstOnly uses only the first model (PaCoRe-compatible)
func distributeFirstOnly(numCalls int, modelRefs []config.ModelRef) []ModelCall {
	if numCalls <= 0 || len(modelRefs) == 0 {
		return nil
	}

	ref := modelRefs[0]
	calls := make([]ModelCall, numCalls)
	for i := 0; i < numCalls; i++ {
		calls[i] = modelCallFromRef(ref)
	}

	return calls
}

func modelCallFromRef(ref config.ModelRef) ModelCall {
	return ModelCall{
		Model:    ref.Model,
		LoRAName: ref.LoRAName,
	}
}

func shuffleModelCalls(calls []ModelCall, seed int) {
	if len(calls) <= 1 {
		return
	}
	r := rand.New(rand.NewSource(int64(seed)))
	r.Shuffle(len(calls), func(i, j int) {
		calls[i], calls[j] = calls[j], calls[i]
	})
}
