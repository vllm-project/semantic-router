package dsl

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const topicContinuityDSL = `
MODEL "chat-model" {}

SIGNAL topic_continuity "topic_boundary" {
  description: "Whether the live turn still depends on retained history."
}

SIGNAL topic_continuity "strict" {
  include_assistant: false
  thresholds: { change: 0, continuation: 0.4 }
  limits: { max_prior_turns: 4, max_turn_bytes: 4096 }
}

ROUTE "default-route" {
  PRIORITY 100
  MODEL "chat-model"
}`

func TestTopicContinuityDSLRoundTrip(t *testing.T) {
	cfg := mustCompilePolicyDSL(t, topicContinuityDSL)
	if len(cfg.TopicContinuityRules) != 2 {
		t.Fatalf("rules = %+v", cfg.TopicContinuityRules)
	}
	defaults, strict := cfg.TopicContinuityRules[0], cfg.TopicContinuityRules[1]
	if defaults.IncludeAssistant != nil || defaults.Thresholds != nil || defaults.Limits != nil {
		t.Fatalf("omitted fields were filled in: %+v", defaults)
	}
	if strict.IncludeAssistant == nil || *strict.IncludeAssistant ||
		strict.Thresholds == nil || strict.Thresholds.Change == nil || *strict.Thresholds.Change != 0 ||
		strict.Limits == nil || strict.Limits.MaxPriorTurns != 4 || strict.Limits.MaxInputBytes != 0 {
		t.Fatalf("explicit values not preserved: %+v", strict)
	}

	text, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("Decompile error: %v", err)
	}
	if strings.Contains(text, "max_input_bytes") {
		t.Fatalf("an omitted default was written out:\n%s", text)
	}
	again := mustCompilePolicyDSL(t, text)
	if !reflect.DeepEqual(cfg.TopicContinuityRules, again.TopicContinuityRules) {
		t.Fatalf("round trip changed the rules:\n first %+v\nsecond %+v\n%s",
			cfg.TopicContinuityRules, again.TopicContinuityRules, text)
	}
}

func TestTopicContinuityDSLRejectsInvalidRules(t *testing.T) {
	_, errs := Compile(`
SIGNAL topic_continuity "bad" {
  thresholds: { continuation: 1 }
}`)
	if len(errs) == 0 || !strings.Contains(errs[0].Error(), "0 <= change < continuation < 1") {
		t.Fatalf("expected a threshold error, got %v", errs)
	}
}

func TestEmitUserYAMLNestsTopicContinuitySignals(t *testing.T) {
	cfg := mustCompilePolicyDSL(t, topicContinuityDSL)
	userYAML, err := EmitUserYAML(cfg)
	if err != nil {
		t.Fatalf("EmitUserYAML error: %v", err)
	}
	var raw map[string]interface{}
	if err := yaml.Unmarshal(userYAML, &raw); err != nil {
		t.Fatalf("emitted YAML is invalid: %v\n%s", err, userYAML)
	}
	routing, _ := raw["routing"].(map[string]interface{})
	signals, _ := routing["signals"].(map[string]interface{})
	rules, _ := signals["topic_continuity"].([]interface{})
	if len(rules) != 2 {
		t.Fatalf("signals.topic_continuity = %v:\n%s", signals["topic_continuity"], userYAML)
	}
}
