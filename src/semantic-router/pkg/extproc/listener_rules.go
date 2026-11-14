package extproc

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

// listenerRuleWorkers bounds the rule evaluations one rules call runs at
// once; each rule's deployment still admits its own calls.
const listenerRuleWorkers = 4

// ruleOutcome is one bound rule on one part; done is false until the rule
// answered, so a rule the deadline cut short stays unchecked.
type ruleOutcome struct {
	done    bool
	matched bool
	spans   []modelservice.Span
	err     error
}

// EvaluateListenerRules evaluates the rules a listener binds to app on each
// part with the classifiers of the recipe they bind: whether each guard rule
// matches, or what each redact rule finds, merged across rules. Each rule is
// asked on its own, with its own timeout.
func (r *OpenAIRouter) EvaluateListenerRules(
	ctx context.Context, listener string, rules *config.VSRRules, app string, parts []systemone.RulePart,
) ([]systemone.RulePartResult, error) {
	classifier, ok := r.RecipeClassifiers.ForRecipe(config.RecipeName(rules.Recipe))
	if !ok || classifier == nil {
		return nil, fmt.Errorf("recipe %q has no classifier", rules.Recipe)
	}
	started := time.Now()
	bound := rules.App(app)
	outcomes := make([][]ruleOutcome, len(parts))
	type job struct{ part, rule int }
	jobs := make(chan job)
	var wg sync.WaitGroup
	for range min(listenerRuleWorkers, len(parts)*len(bound)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				// A job taken after the deadline stays unchecked.
				if ctx.Err() != nil {
					continue
				}
				ref, text := bound[j.rule].SignalRef, parts[j.part].Text
				outcome := ruleOutcome{done: true}
				if app == config.VSRAppRedact {
					outcome.spans, outcome.err = classifier.RuleSpans(ctx, ref.Type, ref.Name, text)
				} else {
					outcome.matched, outcome.err = classifier.RuleMatches(ctx, ref.Type, ref.Name, text)
				}
				outcomes[j.part][j.rule] = outcome
			}
		}()
	}
feed:
	for p := range parts {
		outcomes[p] = make([]ruleOutcome, len(bound))
		for i := range bound {
			if ctx.Err() != nil {
				break feed
			}
			select {
			case jobs <- job{p, i}:
			case <-ctx.Done():
				break feed
			}
		}
	}
	close(jobs)
	wg.Wait()
	for p := range outcomes {
		if outcomes[p] == nil {
			outcomes[p] = make([]ruleOutcome, len(bound))
		}
	}
	results := listenerRuleResults(bound, parts, outcomes)
	recordListenerRules(listener, rules.Recipe, app, bound, results, time.Since(started))
	return results, nil
}

// listenerRuleResults turns each part's outcomes into what the plugins read.
func listenerRuleResults(bound []config.VSRRule, parts []systemone.RulePart, outcomes [][]ruleOutcome) []systemone.RulePartResult {
	results := make([]systemone.RulePartResult, len(parts))
	for p, part := range parts {
		result := systemone.RulePartResult{ID: part.ID, Matches: []string{}, Spans: []systemone.RuleSpan{}, Unchecked: []systemone.RuleUnchecked{}}
		var found []ruleSpan
		for i, rule := range bound {
			outcome := outcomes[p][i]
			switch {
			case !outcome.done:
				result.Unchecked = append(result.Unchecked, systemone.RuleUnchecked{Rule: rule.Name, Reason: "not checked before the deadline"})
			case outcome.err != nil:
				result.Unchecked = append(result.Unchecked, systemone.RuleUnchecked{Rule: rule.Name, Reason: uncheckedReason(outcome.err)})
			case outcome.matched:
				result.Matches = append(result.Matches, rule.Name)
			}
			for _, span := range outcome.spans {
				found = append(found, ruleSpan{start: span.Start, end: span.End, rule: i})
			}
		}
		for _, span := range mergeRuleSpans(found) {
			result.Spans = append(result.Spans, systemone.RuleSpan{Start: span.start, End: span.end, Rule: bound[span.rule].Name})
		}
		switch len(result.Unchecked) {
		case 0:
			result.Coverage = "complete"
		case len(bound):
			result.Coverage = "failed"
		default:
			result.Coverage = "partial"
		}
		results[p] = result
	}
	return results
}

// ruleSpan is a span of the rule at its index in the bound list.
type ruleSpan struct{ start, end, rule int }

// mergeRuleSpans joins overlapping spans into one, as masking does
// (masking.MergeSpans): the earliest start keeps its rule, the longer span at
// an equal start, then the rule bound first.
func mergeRuleSpans(spans []ruleSpan) []ruleSpan {
	sort.Slice(spans, func(i, j int) bool {
		a, b := spans[i], spans[j]
		if a.start != b.start {
			return a.start < b.start
		}
		if a.end != b.end {
			return a.end > b.end
		}
		return a.rule < b.rule
	})
	var merged []ruleSpan
	for _, span := range spans {
		if last := len(merged) - 1; last >= 0 && span.start < merged[last].end {
			merged[last].end = max(merged[last].end, span.end)
			continue
		}
		merged = append(merged, span)
	}
	return merged
}

// uncheckedReason says why a rule could not be checked, without the text.
func uncheckedReason(err error) string {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "the decision model did not answer in time"
	case errors.Is(err, context.Canceled):
		return "the call was cancelled"
	case errors.Is(err, modelservice.ErrOverloaded):
		return "the decision model is busy"
	case errors.Is(err, modelservice.ErrUnavailable):
		return "the decision model is unavailable"
	case strings.Contains(err.Error(), "read only part"), strings.Contains(err.Error(), "longer than one window"):
		return "the text is too long to read whole"
	}
	return "the rule could not be evaluated"
}

// recordListenerRules logs one rules call by rule name and count, never text.
func recordListenerRules(listener, recipe, app string, bound []config.VSRRule, results []systemone.RulePartResult, took time.Duration) {
	names := make([]string, len(bound))
	for i, rule := range bound {
		names[i] = rule.Name
	}
	matched, spans, unchecked := 0, 0, 0
	for _, result := range results {
		matched += len(result.Matches)
		spans += len(result.Spans)
		unchecked += len(result.Unchecked)
	}
	logging.ComponentEvent("systemone", "listener_rules_evaluated", map[string]interface{}{
		"listener": listener, "recipe": recipe, "app": app, "rules": names, "parts": len(results),
		"matched": matched, "spans": spans, "unchecked": unchecked, "latency_ms": took.Milliseconds(),
	})
}
