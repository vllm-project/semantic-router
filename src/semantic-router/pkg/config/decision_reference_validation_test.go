package config

import (
	"strings"
	"testing"
)

func TestEntrypointAssignmentUnknownModelRejected(t *testing.T) {
	document := strings.Replace(entrypointRulesYAML, "- model: model-c", "- model: missing", 1)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("unknown Model assignment error = %v", err)
	}
}

func TestEntrypointAssignmentKnownModelAccepted(t *testing.T) {
	if _, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML)); err != nil {
		t.Fatalf("known Model assignment was rejected: %v", err)
	}
}

func TestDuplicateDecisionNamesRejected(t *testing.T) {
	document := strings.Replace(entrypointRulesYAML, "        - name: finish", "        - name: choose", 1)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "choose") {
		t.Fatalf("duplicate decision name error = %v", err)
	}
}

func TestDistinctDecisionNamesAccepted(t *testing.T) {
	if _, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML)); err != nil {
		t.Fatalf("distinct decision names were rejected: %v", err)
	}
}

func TestEntrypointMustAssignEveryRecipeDecision(t *testing.T) {
	document := strings.Replace(entrypointRulesYAML, "          finish: {models: [{model: model-b}]}\n", "", 1)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "must assign every decision") {
		t.Fatalf("incomplete assignment error = %v", err)
	}
}
