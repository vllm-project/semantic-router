package evaluationplane

import (
	"fmt"
	"sort"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// routingRecipePlanFromSnapshot freezes the decision-time graph reachable from
// one recipe. It deliberately derives TargetSnapshotDigest only from the six
// immutable mixture component digests so the plan does not hash itself.
func routingRecipePlanFromSnapshot(
	routing routerconfig.CanonicalRouting,
	mixture ManifestMixture,
) (RoutingRecipePlan, error) {
	graph := newRoutingRecipePlanGraph(routing.Projections)
	for index := range routing.Decisions {
		if err := graph.collectRuleNode(&routing.Decisions[index].Rules); err != nil {
			return RoutingRecipePlan{}, err
		}
	}

	targetSnapshotDigest, err := routingRecipeTargetSnapshotDigest(mixture)
	if err != nil {
		return RoutingRecipePlan{}, err
	}
	armIDs := make([]string, 0, len(mixture.ModelArms))
	for _, arm := range mixture.ModelArms {
		armIDs = append(armIDs, arm.ID)
	}
	plan, err := canonicalRoutingRecipePlan(RoutingRecipePlan{
		ContractVersion:      RoutingRecipePlanContractVersion,
		TargetSnapshotDigest: targetSnapshotDigest,
		ArmIDs:               armIDs,
		FallbackArmID:        mixture.FallbackArmID,
		Signals:              graph.signalSpecs(),
		Projections:          graph.projectionSpecs(),
		TopK:                 routingRecipeTopK(len(armIDs)),
	})
	if err != nil {
		return RoutingRecipePlan{}, err
	}
	if err := ValidateRoutingRecipePlan(plan); err != nil {
		return RoutingRecipePlan{}, err
	}
	return plan, nil
}

func routingRecipeTargetSnapshotDigest(mixture ManifestMixture) (string, error) {
	return canonicalValueDigest(map[string]any{
		"adaptation_digest":      mixture.AdaptationDigest,
		"binding_digest":         mixture.BindingDigest,
		"pool_digest":            mixture.PoolDigest,
		"recipe_digest":          mixture.RecipeDigest,
		"selector_digest":        mixture.SelectorDigest,
		"selector_policy_digest": mixture.SelectorPolicyDigest,
	})
}

func routingRecipeTopK(armCount int) []int {
	if armCount <= 0 {
		return []int{}
	}
	values := map[int]struct{}{1: {}}
	for _, candidate := range []int{min(3, armCount), min(5, armCount)} {
		values[candidate] = struct{}{}
	}
	result := make([]int, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Ints(result)
	return result
}

type routingRecipePlanGraph struct {
	scoresByName    map[string]routerconfig.ProjectionScore
	sourcesByOutput map[string]string
	signals         map[string]struct{}
	projections     map[string]struct{}
	visitingScores  map[string]bool
	visitedScores   map[string]bool
	visitingOutputs map[string]bool
	visitedOutputs  map[string]bool
}

func newRoutingRecipePlanGraph(projections routerconfig.CanonicalProjections) *routingRecipePlanGraph {
	graph := &routingRecipePlanGraph{
		scoresByName:    make(map[string]routerconfig.ProjectionScore, len(projections.Scores)),
		sourcesByOutput: make(map[string]string),
		signals:         make(map[string]struct{}),
		projections:     make(map[string]struct{}),
		visitingScores:  make(map[string]bool),
		visitedScores:   make(map[string]bool),
		visitingOutputs: make(map[string]bool),
		visitedOutputs:  make(map[string]bool),
	}
	for _, score := range projections.Scores {
		graph.scoresByName[strings.TrimSpace(score.Name)] = score
	}
	for _, mapping := range projections.Mappings {
		for _, output := range mapping.Outputs {
			graph.sourcesByOutput[strings.TrimSpace(output.Name)] = strings.TrimSpace(mapping.Source)
		}
	}
	return graph
}

func (graph *routingRecipePlanGraph) collectRuleNode(node *routerconfig.RuleNode) error {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		signalType := strings.ToLower(strings.TrimSpace(node.Type))
		if signalType == routerconfig.SignalTypeProjection {
			return graph.collectProjectionOutput(strings.TrimSpace(node.Name))
		}
		graph.signals[routingRecipeRuntimeInputID(signalType, strings.TrimSpace(node.Name), strings.TrimSpace(node.Label))] = struct{}{}
		return nil
	}
	for index := range node.Conditions {
		if err := graph.collectRuleNode(&node.Conditions[index]); err != nil {
			return err
		}
	}
	return nil
}

func (graph *routingRecipePlanGraph) collectProjectionOutput(outputName string) error {
	if graph.visitedOutputs[outputName] {
		return nil
	}
	if graph.visitingOutputs[outputName] {
		return fmt.Errorf("projection output dependency %q is cyclic", outputName)
	}
	source, ok := graph.sourcesByOutput[outputName]
	if !ok {
		return fmt.Errorf("projection output %q has no mapping source", outputName)
	}
	graph.visitingOutputs[outputName] = true
	graph.projections[routingRecipeRuntimeInputID(routerconfig.SignalTypeProjection, outputName, "")] = struct{}{}
	if err := graph.collectProjectionScore(source); err != nil {
		return err
	}
	delete(graph.visitingOutputs, outputName)
	graph.visitedOutputs[outputName] = true
	return nil
}

func (graph *routingRecipePlanGraph) collectProjectionScore(scoreName string) error {
	if graph.visitedScores[scoreName] {
		return nil
	}
	if graph.visitingScores[scoreName] {
		return fmt.Errorf("projection score dependency %q is cyclic", scoreName)
	}
	score, ok := graph.scoresByName[scoreName]
	if !ok {
		return fmt.Errorf("projection score %q is unavailable", scoreName)
	}
	graph.visitingScores[scoreName] = true
	for _, input := range score.Inputs {
		signalType := strings.ToLower(strings.TrimSpace(input.Type))
		switch signalType {
		case routerconfig.SignalTypeProjection:
			if strings.EqualFold(strings.TrimSpace(input.ValueSource), routerconfig.ProjectionValueSourceConfidence) {
				if err := graph.collectProjectionOutput(strings.TrimSpace(input.Name)); err != nil {
					return err
				}
			} else if err := graph.collectProjectionScore(strings.TrimSpace(input.Name)); err != nil {
				return err
			}
		case routerconfig.ProjectionInputKBMetric:
			graph.signals[routingRecipeRuntimeInputID(signalType, strings.TrimSpace(input.KB), strings.TrimSpace(input.Metric))] = struct{}{}
		default:
			graph.signals[routingRecipeRuntimeInputID(signalType, strings.TrimSpace(input.Name), "")] = struct{}{}
		}
	}
	delete(graph.visitingScores, scoreName)
	graph.visitedScores[scoreName] = true
	return nil
}

func (graph *routingRecipePlanGraph) signalSpecs() []RoutingRecipeInputSpec {
	ids := sortedRoutingRecipePlanGraphIDs(graph.signals)
	result := make([]RoutingRecipeInputSpec, 0, len(ids))
	for _, id := range ids {
		result = append(result, RoutingRecipeInputSpec{ID: id, ValueKind: "numeric"})
	}
	return result
}

func (graph *routingRecipePlanGraph) projectionSpecs() []RoutingRecipeProjectionSpec {
	ids := sortedRoutingRecipePlanGraphIDs(graph.projections)
	result := make([]RoutingRecipeProjectionSpec, 0, len(ids))
	for _, id := range ids {
		result = append(result, RoutingRecipeProjectionSpec{
			ID: id, ValueKind: "probability", OutcomeBinding: "selected_is_oracle",
		})
	}
	return result
}

func sortedRoutingRecipePlanGraphIDs(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func routingRecipeRuntimeInputID(signalType, name, detail string) string {
	if detail == "" {
		return signalType + ":" + name
	}
	return signalType + ":" + name + ":" + detail
}
