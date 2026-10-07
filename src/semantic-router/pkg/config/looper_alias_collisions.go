package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// looperAliasFamily is one list of direct Looper aliases and the decision
// algorithm a request for one of its names evaluates.
type looperAliasFamily struct {
	field     string
	algorithm string
	names     func(*RouterConfig) []string
}

// looperAliasFamilies are in the order request routing resolves them, so the
// first family that lists a name is the one that captures it.
var looperAliasFamilies = []looperAliasFamily{
	{
		field:     "global.integrations.looper.remom.model_names",
		algorithm: DecisionAlgorithmReMoM,
		names:     func(c *RouterConfig) []string { return c.Looper.ReMoM.EffectiveModelNames() },
	},
	{
		field:     "global.integrations.looper.fusion.model_names",
		algorithm: DecisionAlgorithmFusion,
		names:     func(c *RouterConfig) []string { return c.Looper.Fusion.EffectiveModelNames() },
	},
	{
		field:     "global.integrations.looper.flow.model_names",
		algorithm: DecisionAlgorithmWorkflows,
		names:     func(c *RouterConfig) []string { return c.Looper.Flow.EffectiveModelNames() },
	},
}

// looperAliasCollision is a direct Looper alias that is also the name of a
// model. The alias wins: a request for the name evaluates only the alias's
// decisions, so the model cannot be requested directly, and a request that
// matches none of those decisions has no route.
type looperAliasCollision struct {
	field     string
	alias     string
	algorithm string
	// served reports that providers.models serves the name, as a model or as
	// one of its LoRA adapters.
	served bool
	// routedBy are the decisions whose modelRefs name it.
	routedBy []string
	// evaluated are the decisions a request for the name evaluates.
	evaluated []string
}

func (c looperAliasCollision) reason() string {
	return fmt.Sprintf(
		"requests for %q evaluate only %s decisions, so the model cannot be requested directly and a request that matches none of them fails with no_route; give the alias a name that no model uses",
		c.alias, c.algorithm,
	)
}

// looperAliasCollisions returns every direct Looper alias that providers.models
// serves or that a decision's modelRefs name, in any recipe.
func (c *RouterConfig) looperAliasCollisions() []looperAliasCollision {
	if c == nil {
		return nil
	}
	routedBy := c.decisionsByModelRefName()
	claimed := make(map[string]bool)
	var collisions []looperAliasCollision
	for _, family := range looperAliasFamilies {
		for _, alias := range family.names(c) {
			if claimed[alias] {
				continue
			}
			claimed[alias] = true
			served := c.servesModelName(alias)
			if !served && len(routedBy[alias]) == 0 {
				continue
			}
			collisions = append(collisions, looperAliasCollision{
				field:     family.field,
				alias:     alias,
				algorithm: family.algorithm,
				served:    served,
				routedBy:  routedBy[alias],
				evaluated: c.defaultDecisionNamesWithAlgorithm(family.algorithm),
			})
		}
	}
	return collisions
}

func (c *RouterConfig) servesModelName(name string) bool {
	if _, ok := c.ModelConfig[name]; ok {
		return true
	}
	_, _, ok := c.resolveLoRABaseModel(name)
	return ok
}

// decisionsByModelRefName maps each model or LoRA adapter a decision's
// modelRefs name to those decisions, keyed as RoutingDecisionKey keys them.
func (c *RouterConfig) decisionsByModelRefName() map[string][]string {
	decisions := make(map[string][]string)
	for _, ref := range c.RoutingDecisionRefs() {
		key := RoutingDecisionKey(ref.Recipe, ref.Decision.Name)
		named := make(map[string]bool)
		for _, modelRef := range ref.Decision.ModelRefs {
			for _, name := range []string{modelRef.Model, modelRef.LoRAName} {
				name = strings.TrimSpace(name)
				if name == "" || named[name] {
					continue
				}
				named[name] = true
				decisions[name] = append(decisions[name], key)
			}
		}
	}
	return decisions
}

// defaultDecisionNamesWithAlgorithm lists the default recipe's decisions of
// one algorithm, the only ones a direct Looper alias evaluates.
func (c *RouterConfig) defaultDecisionNamesWithAlgorithm(algorithm string) []string {
	var names []string
	for _, decision := range c.Decisions {
		if decision.Algorithm != nil && decision.Algorithm.Type == algorithm {
			names = append(names, decision.Name)
		}
	}
	return names
}

// warnLooperAliasCollisions reports each direct Looper alias that is also a
// model name. The configuration is valid as written, since the alias simply
// wins, so this warns instead of refusing to load. Recipe views share the
// aliases of the configuration they come from, which already reported them.
func warnLooperAliasCollisions(cfg *RouterConfig) error {
	if cfg == nil || cfg.RoutingScope != "" {
		return nil
	}
	for _, collision := range cfg.looperAliasCollisions() {
		logging.ComponentWarnEvent("config", "looper_alias_shadows_model", map[string]interface{}{
			"field":               collision.field,
			"alias":               collision.alias,
			"served_by_providers": collision.served,
			"routed_by_decisions": collision.routedBy,
			"evaluated_decisions": collision.evaluated,
			"reason":              collision.reason(),
		})
	}
	return nil
}
