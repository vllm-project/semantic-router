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

package classification

import (
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// severityKeywords are the well-known severity level terms used in event payloads
// and structured logs. Matched case-insensitively against request text.
var severityKeywords = map[string]*regexp.Regexp{
	"critical": regexp.MustCompile(`(?i)\bcritical\b`),
	"high":     regexp.MustCompile(`(?i)\bhigh\b`),
	"medium":   regexp.MustCompile(`(?i)\bmedium\b`),
	"low":      regexp.MustCompile(`(?i)\blow\b`),
}

// temporalMarkers matches urgency indicators common in incident and alert payloads.
var temporalMarkers = regexp.MustCompile(`(?i)\b(urgent|immediate|asap|deadline|time.sensitive|now|critical.window)\b`)

// EventContextClassifier evaluates EventContextRules against request text.
// It extracts structured event metadata — event type, severity, temporal urgency,
// and domain-specific action codes — without requiring an ML model.
type EventContextClassifier struct {
	rules []compiledEventContextRule
}

type compiledEventContextRule struct {
	name        string
	eventTypes  []*regexp.Regexp
	severities  []string // matched against severityKeywords
	actionCodes []*regexp.Regexp
	temporal    bool
}

// NewEventContextClassifier compiles EventContextRules into regex patterns.
// Empty or whitespace-only event type and action code strings are skipped to
// prevent the degenerate pattern \b\b from matching unintended text.
// Severity values are normalised (trimmed, lowercased) at compile time; unknown
// severities are dropped and do not count as a configured criterion.
func NewEventContextClassifier(rules []config.EventContextRule) *EventContextClassifier {
	compiled := make([]compiledEventContextRule, 0, len(rules))
	for _, r := range rules {
		cr := compiledEventContextRule{
			name:     r.Name,
			temporal: r.Temporal,
		}
		for _, et := range r.EventTypes {
			et = strings.TrimSpace(et)
			if et == "" {
				continue
			}
			pattern := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(et) + `\b`)
			cr.eventTypes = append(cr.eventTypes, pattern)
		}
		for _, ac := range r.ActionCodes {
			ac = strings.TrimSpace(ac)
			if ac == "" {
				continue
			}
			pattern := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(ac) + `\b`)
			cr.actionCodes = append(cr.actionCodes, pattern)
		}
		// Normalise severities at compile time — trim, lowercase, drop unknowns.
		for _, sev := range r.Severities {
			key := strings.ToLower(strings.TrimSpace(sev))
			if _, ok := severityKeywords[key]; ok {
				cr.severities = append(cr.severities, key)
			}
		}
		compiled = append(compiled, cr)
	}
	return &EventContextClassifier{rules: compiled}
}

// EventContextMatch holds the details of a single rule match.
type EventContextMatch struct {
	RuleName        string
	MatchedSeverity string
	TemporalMatch   bool
	Confidence      float64
}

// Classify evaluates all rules against text and returns matches.
// A rule matches when at least one of its configured criteria is satisfied.
// Confidence is proportional to how many criteria matched (0.5 base + bonus per criterion).
func (ec *EventContextClassifier) Classify(text string) []EventContextMatch {
	var matches []EventContextMatch
	for _, rule := range ec.rules {
		match, ok := ec.evaluateRule(rule, text)
		if ok {
			matches = append(matches, match)
		}
	}
	return matches
}

func (ec *EventContextClassifier) evaluateRule(rule compiledEventContextRule, text string) (EventContextMatch, bool) {
	match := EventContextMatch{RuleName: rule.name}
	criteriaCount, matchedCount := 0, 0

	criteriaCount, matchedCount = matchEventTypes(rule, text, criteriaCount, matchedCount)
	criteriaCount, matchedCount, match.MatchedSeverity = matchSeverities(rule, text, criteriaCount, matchedCount)
	criteriaCount, matchedCount = matchActionCodes(rule, text, criteriaCount, matchedCount)
	criteriaCount, matchedCount, match.TemporalMatch = matchTemporal(rule, text, criteriaCount, matchedCount)

	if criteriaCount == 0 || matchedCount == 0 {
		return EventContextMatch{}, false
	}

	// Confidence: 0.5 base + up to 0.5 proportional to matched criteria.
	match.Confidence = 0.5 + 0.5*float64(matchedCount)/float64(criteriaCount)
	return match, true
}

func matchEventTypes(rule compiledEventContextRule, text string, criteria, matched int) (int, int) {
	if len(rule.eventTypes) == 0 {
		return criteria, matched
	}
	criteria++
	for _, re := range rule.eventTypes {
		if re.MatchString(text) {
			return criteria, matched + 1
		}
	}
	return criteria, matched
}

func matchSeverities(rule compiledEventContextRule, text string, criteria, matched int) (int, int, string) {
	// Severities are already normalised (trimmed, lowercased, validated) at
	// compile time in NewEventContextClassifier; unknown values were dropped.
	if len(rule.severities) == 0 {
		return criteria, matched, ""
	}
	criteria++
	for _, key := range rule.severities {
		if re := severityKeywords[key]; re.MatchString(text) {
			return criteria, matched + 1, key
		}
	}
	return criteria, matched, ""
}

func matchActionCodes(rule compiledEventContextRule, text string, criteria, matched int) (int, int) {
	if len(rule.actionCodes) == 0 {
		return criteria, matched
	}
	criteria++
	for _, re := range rule.actionCodes {
		if re.MatchString(text) {
			return criteria, matched + 1
		}
	}
	return criteria, matched
}

func matchTemporal(rule compiledEventContextRule, text string, criteria, matched int) (int, int, bool) {
	if !rule.temporal {
		return criteria, matched, false
	}
	criteria++
	if temporalMarkers.MatchString(text) {
		return criteria, matched + 1, true
	}
	return criteria, matched, false
}
