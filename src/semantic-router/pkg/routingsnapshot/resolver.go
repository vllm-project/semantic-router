package routingsnapshot

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ruleSpecificity struct {
	claims    int
	exactPath int
	prefixLen int
}

func validateRuleAmbiguity(rules []EntrypointRule) error {
	seen := make(map[string]string, len(rules))
	for _, rule := range rules {
		keyBytes, _ := json.Marshal(rule.Matchers)
		key := string(keyBytes)
		actionBytes, _ := json.Marshal(struct {
			RecipeID       string
			RecipeRevision int64
			Assignments    map[string]AssignmentSet
		}{rule.RecipeID, rule.RecipeRevision, rule.Assignments})
		if previous, exists := seen[key]; exists && previous != string(actionBytes) {
			return fmt.Errorf("equally specific rules with identical matchers have different actions")
		}
		seen[key] = string(actionBytes)
	}
	return nil
}

func (snapshot *Snapshot) Resolve(input ResolveInput) (Resolution, error) {
	if snapshot == nil {
		return Resolution{}, fmt.Errorf("routing snapshot is required")
	}
	entrypointID, claimed := snapshot.aliases[strings.TrimSpace(input.Alias)]
	if !claimed {
		return Resolution{Outcome: ResolveUnclaimed}, nil
	}
	entrypoint := snapshot.entrypointsByID[entrypointID]
	var selected *EntrypointRule
	var selectedSpecificity ruleSpecificity
	for i := range entrypoint.Rules {
		rule := &entrypoint.Rules[i]
		matched, specificity := ruleMatches(*rule, input)
		if !matched {
			continue
		}
		if selected == nil || compareSpecificity(specificity, selectedSpecificity) > 0 {
			copy := *rule
			selected = &copy
			selectedSpecificity = specificity
			continue
		}
		if compareSpecificity(specificity, selectedSpecificity) == 0 {
			left, _ := json.Marshal(selected)
			right, _ := json.Marshal(rule)
			if string(left) != string(right) {
				return Resolution{}, fmt.Errorf("ambiguous routing rules in active snapshot")
			}
		}
	}
	entrypointCopy := entrypoint
	if selected == nil {
		return Resolution{Outcome: ResolveClaimedNoMatch, Entrypoint: &entrypointCopy}, nil
	}
	recipe := snapshot.recipesByID[selected.RecipeID]
	recipeCopy := recipe
	return Resolution{Outcome: ResolveMatched, Entrypoint: &entrypointCopy, Rule: selected, Recipe: &recipeCopy}, nil
}

func ruleMatches(rule EntrypointRule, input ResolveInput) (bool, ruleSpecificity) {
	var specificity ruleSpecificity
	for _, matcher := range rule.Matchers {
		switch {
		case matcher.Claim != nil:
			actual, exists := input.Claims[matcher.Claim.Name]
			if !exists || actual != matcher.Claim.Value {
				return false, ruleSpecificity{}
			}
			specificity.claims++
		case matcher.ExactPath != "":
			if input.Path != matcher.ExactPath {
				return false, ruleSpecificity{}
			}
			specificity.exactPath = 1
		case matcher.PathPrefix != "":
			if !segmentPrefix(input.Path, matcher.PathPrefix) {
				return false, ruleSpecificity{}
			}
			specificity.prefixLen = len(matcher.PathPrefix)
		}
	}
	return true, specificity
}

func segmentPrefix(path, prefix string) bool {
	if path == prefix {
		return true
	}
	if prefix == "/" {
		return strings.HasPrefix(path, "/")
	}
	return strings.HasPrefix(path, strings.TrimSuffix(prefix, "/")+"/")
}

func compareSpecificity(left, right ruleSpecificity) int {
	if left.claims != right.claims {
		if left.claims > right.claims {
			return 1
		}
		return -1
	}
	if left.exactPath != right.exactPath {
		if left.exactPath > right.exactPath {
			return 1
		}
		return -1
	}
	if left.prefixLen != right.prefixLen {
		if left.prefixLen > right.prefixLen {
			return 1
		}
		return -1
	}
	return 0
}

func (snapshot *Snapshot) Model(id string) (Model, bool) {
	if snapshot == nil {
		return Model{}, false
	}
	model, ok := snapshot.modelsByID[id]
	return model, ok
}

func (snapshot *Snapshot) Entrypoint(id string) (Entrypoint, bool) {
	if snapshot == nil {
		return Entrypoint{}, false
	}
	entrypoint, ok := snapshot.entrypointsByID[id]
	return entrypoint, ok
}
