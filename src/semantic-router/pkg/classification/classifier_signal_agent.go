package classification

import (
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// AgentFacts is the agent a request belongs to, as its harness's headers
// name it: Role is main or subagent, empty when no agent session is named;
// Harness is claude_code or codex, empty for any other client; Type is the
// type the harness gives a subagent, and RequestClass the class Claude Code
// gives the request, each empty when the harness sends none.
type AgentFacts struct {
	Role         string
	Harness      string
	Type         string
	RequestClass string
}

// evaluateAgentSignal matches agent rules against the request's agent facts.
// Purely deterministic: a rule matches when every field it sets matches.
func (c *Classifier) evaluateAgentSignal(
	results *SignalResults,
	mu *sync.Mutex,
	facts RequestFacts,
	usedSignals map[string]bool,
) {
	start := time.Now()
	for _, rule := range c.Config.AgentRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeAgent, rule.Name) || !agentRuleMatches(rule, facts.Agent) {
			continue
		}
		mu.Lock()
		results.MatchedAgentRules = append(results.MatchedAgentRules, rule.Name)
		results.SignalConfidences[signalConfidenceKey(config.SignalTypeAgent, rule.Name)] = 1.0
		mu.Unlock()
		c.recordSignalMatch(config.SignalTypeAgent, rule.Name)
	}
	results.Metrics.Agent.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	results.Metrics.Agent.Method = "harness_headers"
}

func agentRuleMatches(rule config.AgentRule, facts AgentFacts) bool {
	if rule.Role != "" && rule.Role != facts.Role || rule.Harness != "" && rule.Harness != facts.Harness {
		return false
	}
	return namedIn(rule.Types, facts.Type) && namedIn(rule.RequestClasses, facts.RequestClass)
}

// namedIn reports whether value, ignoring case, is one of names; any value
// is when names is empty.
func namedIn(names []string, value string) bool {
	return len(names) == 0 || value != "" && slices.ContainsFunc(names, func(name string) bool {
		return strings.EqualFold(name, value)
	})
}
