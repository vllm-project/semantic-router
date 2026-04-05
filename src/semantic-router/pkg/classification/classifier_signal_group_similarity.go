package classification

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// SignalGroupCentroidWarning describes a softmax_exclusive group whose embedding
// member centroids are too similar to provide a clear winner.
type SignalGroupCentroidWarning struct {
	GroupName   string
	LeftMember  string
	RightMember string
	Similarity  float64
}

func (c *Classifier) AnalyzeSoftmaxSignalGroupCentroids(threshold float64) ([]SignalGroupCentroidWarning, error) {
	if c == nil || c.Config == nil {
		return nil, nil
	}
	if threshold <= 0 {
		threshold = 0.7
	}

	rulesByName := embeddingRulesByName(c.Config.EmbeddingRules)
	groups := softmaxEmbeddingSignalGroups(c.Config.Projections.Partitions, rulesByName)
	if len(groups) == 0 {
		return nil, nil
	}
	if c.keywordEmbeddingClassifier == nil {
		return nil, fmt.Errorf("embedding classifier is not initialized")
	}

	centroids := make(map[string][]float32)
	warnings := make([]SignalGroupCentroidWarning, 0)
	for _, group := range groups {
		groupWarnings, err := c.analyzeSoftmaxEmbeddingGroup(group, rulesByName, centroids, threshold)
		if err != nil {
			return nil, err
		}
		warnings = append(warnings, groupWarnings...)
	}

	sort.Slice(warnings, func(i, j int) bool {
		if warnings[i].GroupName != warnings[j].GroupName {
			return warnings[i].GroupName < warnings[j].GroupName
		}
		if warnings[i].LeftMember != warnings[j].LeftMember {
			return warnings[i].LeftMember < warnings[j].LeftMember
		}
		return warnings[i].RightMember < warnings[j].RightMember
	})
	return warnings, nil
}

func (c *Classifier) analyzeSoftmaxEmbeddingGroup(
	group config.ProjectionPartition,
	rulesByName map[string]config.EmbeddingRule,
	centroids map[string][]float32,
	threshold float64,
) ([]SignalGroupCentroidWarning, error) {
	warnings := make([]SignalGroupCentroidWarning, 0)
	for i := 0; i < len(group.Members); i++ {
		left := group.Members[i]
		leftCentroid, err := c.embeddingRuleCentroid(left, rulesByName[left], centroids)
		if err != nil {
			return nil, err
		}
		pairWarnings, err := c.signalGroupCentroidWarningsForMember(
			group,
			left,
			leftCentroid,
			group.Members[i+1:],
			rulesByName,
			centroids,
			threshold,
		)
		if err != nil {
			return nil, err
		}
		warnings = append(warnings, pairWarnings...)
	}
	return warnings, nil
}

func (c *Classifier) signalGroupCentroidWarningsForMember(
	group config.ProjectionPartition,
	left string,
	leftCentroid []float32,
	remainingMembers []string,
	rulesByName map[string]config.EmbeddingRule,
	centroids map[string][]float32,
	threshold float64,
) ([]SignalGroupCentroidWarning, error) {
	warnings := make([]SignalGroupCentroidWarning, 0)
	for _, right := range remainingMembers {
		rightCentroid, err := c.embeddingRuleCentroid(right, rulesByName[right], centroids)
		if err != nil {
			return nil, err
		}
		similarity := float64(cosineSimilarity(leftCentroid, rightCentroid))
		if similarity < threshold {
			continue
		}
		warnings = append(warnings, SignalGroupCentroidWarning{
			GroupName:   group.Name,
			LeftMember:  left,
			RightMember: right,
			Similarity:  similarity,
		})
	}
	return warnings, nil
}

func embeddingRulesByName(rules []config.EmbeddingRule) map[string]config.EmbeddingRule {
	result := make(map[string]config.EmbeddingRule, len(rules))
	for _, rule := range rules {
		result[rule.Name] = rule
	}
	return result
}

func softmaxEmbeddingSignalGroups(
	groups []config.ProjectionPartition,
	rulesByName map[string]config.EmbeddingRule,
) []config.ProjectionPartition {
	result := make([]config.ProjectionPartition, 0)
	for _, group := range groups {
		if !strings.EqualFold(group.Semantics, "softmax_exclusive") {
			continue
		}
		members := make([]string, 0, len(group.Members))
		for _, member := range group.Members {
			if _, ok := rulesByName[member]; ok {
				members = append(members, member)
			}
		}
		if len(members) < 2 {
			continue
		}
		group.Members = members
		result = append(result, group)
	}
	return result
}

func (c *Classifier) embeddingRuleCentroid(
	name string,
	rule config.EmbeddingRule,
	cache map[string][]float32,
) ([]float32, error) {
	if centroid, ok := cache[name]; ok {
		return centroid, nil
	}
	centroid, err := c.keywordEmbeddingClassifier.ruleCentroid(rule)
	if err != nil {
		return nil, fmt.Errorf("failed to compute centroid for embedding rule %q: %w", name, err)
	}
	cache[name] = centroid
	return centroid, nil
}

func (c *EmbeddingClassifier) ensureCandidateEmbeddings() error {
	if len(c.candidateEmbeddings) > 0 {
		if len(c.rulePrototypeBanks) == 0 {
			c.rebuildRulePrototypeBanks()
		}
		return nil
	}
	return c.preloadCandidateEmbeddings()
}

func (c *EmbeddingClassifier) ruleCentroid(rule config.EmbeddingRule) ([]float32, error) {
	if centroid, ok := prototypeBankCentroid(c.rulePrototypeBanks[rule.Name]); ok {
		return centroid, nil
	}
	if err := c.ensureCandidateEmbeddings(); err != nil {
		return nil, err
	}
	return c.candidateRuleCentroid(rule)
}

func prototypeBankCentroid(bank *prototypeBank) ([]float32, bool) {
	if bank == nil || len(bank.prototypes) == 0 {
		return nil, false
	}
	embeddings := make([][]float32, 0, len(bank.prototypes))
	for _, prototype := range bank.prototypes {
		embeddings = append(embeddings, prototype.Embedding)
	}
	return averageEmbeddings(embeddings), true
}

func (c *EmbeddingClassifier) candidateRuleCentroid(rule config.EmbeddingRule) ([]float32, error) {
	if len(rule.Candidates) == 0 {
		return nil, fmt.Errorf("embedding rule %q has no candidates", rule.Name)
	}

	embeddings := make([][]float32, 0, len(rule.Candidates))
	for _, candidate := range rule.Candidates {
		embedding, ok := c.candidateEmbeddings[candidate]
		if !ok {
			return nil, fmt.Errorf("candidate %q has no embedding", candidate)
		}
		embeddings = append(embeddings, embedding)
	}
	return averageEmbeddings(embeddings), nil
}

func averageEmbeddings(embeddings [][]float32) []float32 {
	if len(embeddings) == 0 {
		return nil
	}

	centroid := make([]float32, len(embeddings[0]))
	for _, embedding := range embeddings {
		for i, value := range embedding {
			centroid[i] += value
		}
	}

	scale := float32(len(embeddings))
	for i := range centroid {
		centroid[i] /= scale
	}
	return centroid
}
