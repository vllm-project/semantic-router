/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// AdaOpsSolver implements the AdaOps POMDP solver from AutoMix paper (arXiv:2310.12963)
//
// Key concepts:
// - State S = ⟨current_LM, perf_LM1, ..., perf_LMN⟩
// - Actions: Keep answer OR route to larger LM
// - Observations: Verifier probability v (noisy signal of correctness)
// - Reward: R = Performance - λ × Cost
// - Belief: Probability distribution over states, updated via particle filtering
type AdaOpsSolver struct {
	// Configuration
	numParticles   int     // Number of particles for belief representation
	costLambda     float64 // Cost tradeoff parameter
	discountFactor float64 // Gamma for future rewards

	// Model information
	models     []string           // Ordered by size (smallest first)
	modelCosts map[string]float64 // Cost per 1M tokens
	modelSizes map[string]float64 // Parameter count (for ordering)

	// Belief state: particles representing distribution over model performance
	// Each particle is a map from model -> estimated performance
	particles []map[string]float64
	beliefMu  sync.RWMutex

	// Transition probabilities: P(s'|s,a)
	// Simplified: assume performance estimate updates based on observation

	// Value function approximation
	valueFunc map[string]float64

	rng *rand.Rand
}

// NewAdaOpsSolver creates a new AdaOps POMDP solver
func NewAdaOpsSolver(numParticles int, costLambda, discountFactor float64) *AdaOpsSolver {
	return &AdaOpsSolver{
		numParticles:   numParticles,
		costLambda:     costLambda,
		discountFactor: discountFactor,
		models:         make([]string, 0),
		modelCosts:     make(map[string]float64),
		modelSizes:     make(map[string]float64),
		particles:      make([]map[string]float64, 0),
		valueFunc:      make(map[string]float64),
		rng:            rand.New(rand.NewSource(42)),
	}
}

// RegisterModel adds a model to the solver
func (s *AdaOpsSolver) RegisterModel(name string, cost, paramSize float64) {
	s.modelCosts[name] = cost
	s.modelSizes[name] = paramSize

	// Keep models sorted by size
	s.models = append(s.models, name)
	sort.Slice(s.models, func(i, j int) bool {
		return s.modelSizes[s.models[i]] < s.modelSizes[s.models[j]]
	})

	// Initialize value function
	s.valueFunc[name] = paramSize / 100.0 // Initial estimate based on size

	// Reinitialize particles
	s.initializeParticles()
}

// initializeParticles creates initial particle distribution
func (s *AdaOpsSolver) initializeParticles() {
	s.beliefMu.Lock()
	defer s.beliefMu.Unlock()

	s.particles = make([]map[string]float64, s.numParticles)
	for i := 0; i < s.numParticles; i++ {
		particle := make(map[string]float64)
		for _, model := range s.models {
			// Initialize with prior based on model size (larger = better)
			size := s.modelSizes[model]
			// Prior mean: larger models have higher expected performance
			priorMean := 0.5 + 0.3*math.Min(size/70.0, 1.0) // Scales to ~0.8 for 70B+ models
			// Add noise to create particle diversity
			noise := s.rng.NormFloat64() * 0.1
			particle[model] = math.Max(0.1, math.Min(0.95, priorMean+noise))
		}
		s.particles[i] = particle
	}
}

// GetBeliefMean returns the mean belief about each model's performance
func (s *AdaOpsSolver) GetBeliefMean() map[string]float64 {
	s.beliefMu.RLock()
	defer s.beliefMu.RUnlock()

	if len(s.particles) == 0 {
		return nil
	}

	means := make(map[string]float64)
	for _, model := range s.models {
		sum := 0.0
		for _, particle := range s.particles {
			sum += particle[model]
		}
		means[model] = sum / float64(len(s.particles))
	}
	return means
}

// UpdateBelief updates the belief state given an observation
// This is the core particle filtering step
func (s *AdaOpsSolver) UpdateBelief(model string, observation float64) {
	s.beliefMu.Lock()

	if len(s.particles) == 0 {
		s.beliefMu.Unlock()
		return
	}

	// Compute importance weights based on observation likelihood
	weights := make([]float64, len(s.particles))
	totalWeight := 0.0

	for i, particle := range s.particles {
		// Likelihood: P(observation | particle's belief about model)
		// Using Gaussian likelihood centered at particle's belief
		belief := particle[model]
		// Observation model: v ~ N(true_perf, sigma^2)
		sigma := 0.15 // Observation noise
		diff := observation - belief
		likelihood := math.Exp(-diff * diff / (2 * sigma * sigma))
		weights[i] = likelihood
		totalWeight += likelihood
	}

	// Normalize weights
	if totalWeight > 0 {
		for i := range weights {
			weights[i] /= totalWeight
		}
	} else {
		// Uniform weights if all likelihoods are zero
		for i := range weights {
			weights[i] = 1.0 / float64(len(weights))
		}
	}

	// Resample particles based on weights (importance resampling)
	newParticles := make([]map[string]float64, s.numParticles)
	for i := 0; i < s.numParticles; i++ {
		// Select a particle based on weights
		idx := s.selectParticle(weights)

		// Create new particle with perturbation (for diversity)
		newParticle := make(map[string]float64)
		for m, v := range s.particles[idx] {
			noise := s.rng.NormFloat64() * 0.02 // Small transition noise
			newParticle[m] = math.Max(0.1, math.Min(0.95, v+noise))
		}

		// Update the observed model's estimate towards observation
		alpha := 0.2 // Learning rate
		newParticle[model] = (1-alpha)*newParticle[model] + alpha*observation

		newParticles[i] = newParticle
	}

	s.particles = newParticles
	s.beliefMu.Unlock() // Release lock before calling updateValueFunction

	// Update value function (acquires its own lock)
	s.updateValueFunction()

	// Log debug info
	beliefs := s.GetBeliefMean()
	logging.Debugf("[AdaOps] Updated belief for %s with observation %.3f, new mean: %.3f",
		model, observation, beliefs[model])
}

// selectParticle selects a particle index based on weights (roulette wheel selection)
func (s *AdaOpsSolver) selectParticle(weights []float64) int {
	r := s.rng.Float64()
	cumSum := 0.0
	for i, w := range weights {
		cumSum += w
		if r <= cumSum {
			return i
		}
	}
	return len(weights) - 1
}

// updateValueFunction updates the value function approximation
// NOTE: Must be called WITHOUT holding beliefMu lock to avoid deadlock
func (s *AdaOpsSolver) updateValueFunction() {
	// Value = E[Performance] - λ * Cost
	// Compute beliefs inline (lock is already released)
	beliefs := make(map[string]float64)
	s.beliefMu.RLock()
	if len(s.particles) > 0 {
		for _, model := range s.models {
			sum := 0.0
			for _, p := range s.particles {
				sum += p[model]
			}
			beliefs[model] = sum / float64(len(s.particles))
		}
	}
	s.beliefMu.RUnlock()

	for model, belief := range beliefs {
		cost := s.modelCosts[model]
		normalizedCost := cost / 10.0 // Normalize assuming max cost ~$10
		s.valueFunc[model] = belief - s.costLambda*normalizedCost
	}
}

// SelectAction decides the optimal action: which model to use next
// Returns (model_to_use, expected_value, should_keep_or_escalate)
func (s *AdaOpsSolver) SelectAction(currentModel string, currentConfidence float64, candidateModels []string) (string, float64, string) {
	s.beliefMu.RLock()
	defer s.beliefMu.RUnlock()

	beliefs := make(map[string]float64)
	if len(s.particles) > 0 {
		for _, model := range candidateModels {
			sum := 0.0
			for _, particle := range s.particles {
				if v, ok := particle[model]; ok {
					sum += v
				} else {
					sum += 0.5 // Default belief
				}
			}
			beliefs[model] = sum / float64(len(s.particles))
		}
	} else {
		// No particles, use model size as proxy
		for _, model := range candidateModels {
			beliefs[model] = s.modelSizes[model] / 70.0 // Normalize to ~1.0 for large models
		}
	}

	// Compute Q-values for each action
	qValues := make(map[string]float64)

	for _, model := range candidateModels {
		belief := beliefs[model]
		cost := s.modelCosts[model]
		if cost == 0 {
			cost = 0.1 // Default cost
		}

		// Q(s, a) = R(s, a) + γ * V(s')
		// R(s, a) = expected_performance - λ * cost
		normalizedCost := cost / 10.0
		immediateReward := belief - s.costLambda*normalizedCost

		// Future value: if we use this model, what's the expected future value?
		// Simplified: assume we'll get value proportional to model capability
		futureValue := s.valueFunc[model]
		if futureValue == 0 {
			futureValue = belief
		}

		qValues[model] = immediateReward + s.discountFactor*futureValue
	}

	// Find best action
	bestModel := currentModel
	bestValue := -math.MaxFloat64
	for model, value := range qValues {
		if value > bestValue {
			bestValue = value
			bestModel = model
		}
	}

	// Determine if we should keep current answer or escalate
	action := "keep"
	if currentModel != "" {
		currentValue := qValues[currentModel]
		if bestValue > currentValue+0.1 { // Threshold for escalation
			action = "escalate"
		}
	}

	return bestModel, bestValue, action
}

// ComputeIBC computes the Incremental Benefit per Cost (IBC) metric from the paper
// IBC = (Performance_new - Performance_old) / (Cost_new - Cost_old)
func (s *AdaOpsSolver) ComputeIBC(fromModel, toModel string, perfImprovement float64) float64 {
	fromCost := s.modelCosts[fromModel]
	toCost := s.modelCosts[toModel]

	costDiff := toCost - fromCost
	if costDiff <= 0 {
		return math.Inf(1) // Free improvement
	}

	return perfImprovement / costDiff
}

// GetEscalationPath returns the ordered path of models to try
func (s *AdaOpsSolver) GetEscalationPath(startModel string, maxEscalations int) []string {
	path := []string{}
	started := false
	count := 0

	for _, model := range s.models {
		if model == startModel {
			started = true
			continue
		}
		if started && count < maxEscalations {
			path = append(path, model)
			count++
		}
	}

	return path
}
