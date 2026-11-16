package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/spendbudget"
)

// DefaultBudgetOutputReserveTokens is the output a budget reserves for a call
// whose request and model name no output limit.
const DefaultBudgetOutputReserveTokens = 4096

// DefaultRunBudgetIdleTimeout is how long an agent run may go unused before
// its spend is forgotten.
const DefaultRunBudgetIdleTimeout = 24 * time.Hour

// Budget unpriced policies: what a dispatch to a model with no price in the
// budgets' currency gets.
const (
	BudgetUnpricedRefuse = "refuse"
	BudgetUnpricedAllow  = "allow"
)

// BudgetsConfig is global.services.budgets: spend budgets set once and
// enforced on every model call, across an agent run and per user over a
// window. A call reserves its estimated cost before it is dispatched and
// settles its actual cost after, so calls in flight together cannot spend
// past a budget.
type BudgetsConfig struct {
	// Currency is the one every budget counts in; default USD. A call
	// priced in another currency is unpriced.
	Currency string `yaml:"currency,omitempty" json:"currency,omitempty"`
	// Unpriced is refuse (the default) or allow: allow admits a call to a
	// model with no price at zero and counts it as unpriced.
	Unpriced string `yaml:"unpriced,omitempty" json:"unpriced,omitempty" jsonschema:"enum=refuse,enum=allow"`
	// OutputReserveTokens is the output a call reserves when neither its
	// request nor its model sets a limit; default 4096.
	OutputReserveTokens int `yaml:"output_reserve_tokens,omitempty" json:"output_reserve_tokens,omitempty" jsonschema:"minimum=0"`
	// Run is the budget of one agent run: a user's session with every
	// subagent, Looper hop, retry and Router model call in it.
	Run *RunBudget `yaml:"run,omitempty" json:"run,omitempty"`
	// Users are budgets per user over a calendar window. Every rule that
	// matches a request applies.
	Users []UserBudget `yaml:"users,omitempty" json:"users,omitempty"`
}

// RunBudget is the spend limit of one agent run.
type RunBudget struct {
	Limit float64 `yaml:"limit" json:"limit" jsonschema:"required,exclusiveMinimum=0"`
	// IdleTimeoutSeconds forgets a run unused this long, so a session
	// resumed later starts a new run; default 86400.
	IdleTimeoutSeconds int `yaml:"idle_timeout_seconds,omitempty" json:"idle_timeout_seconds,omitempty" jsonschema:"minimum=0"`
}

// UserBudget is the spend limit of each user a rule matches, per window.
type UserBudget struct {
	Name  string      `yaml:"name" json:"name" jsonschema:"required"`
	Match BudgetMatch `yaml:"match,omitempty" json:"match,omitempty"`
	Limit float64     `yaml:"limit" json:"limit" jsonschema:"required,exclusiveMinimum=0"`
	// Window is the calendar span, in UTC, the budget resets over.
	Window string `yaml:"window" json:"window" jsonschema:"required,enum=hour,enum=day,enum=week,enum=month"`
}

// BudgetMatch selects the requests a user budget applies to, as a rate-limit
// rule's match does: empty or "*" matches every request.
type BudgetMatch struct {
	User  string `yaml:"user,omitempty" json:"user,omitempty"`
	Group string `yaml:"group,omitempty" json:"group,omitempty"`
}

// Enabled reports whether any budget is configured.
func (b BudgetsConfig) Enabled() bool {
	return b.Run != nil || len(b.Users) > 0
}

// EffectiveCurrency is the budgets' currency, upper case.
func (b BudgetsConfig) EffectiveCurrency() string {
	if currency := strings.ToUpper(strings.TrimSpace(b.Currency)); currency != "" {
		return currency
	}
	return "USD"
}

// RefusesUnpriced reports whether a call with no price is refused.
func (b BudgetsConfig) RefusesUnpriced() bool {
	return b.Unpriced != BudgetUnpricedAllow
}

// EffectiveOutputReserveTokens is OutputReserveTokens or its default.
func (b BudgetsConfig) EffectiveOutputReserveTokens() int {
	if b.OutputReserveTokens > 0 {
		return b.OutputReserveTokens
	}
	return DefaultBudgetOutputReserveTokens
}

// IdleTimeout is how long the run may go unused before it is forgotten.
func (r *RunBudget) IdleTimeout() time.Duration {
	if r == nil || r.IdleTimeoutSeconds <= 0 {
		return DefaultRunBudgetIdleTimeout
	}
	return time.Duration(r.IdleTimeoutSeconds) * time.Second
}

// Matches reports whether the rule applies to a user in groups.
func (m BudgetMatch) Matches(user string, groups []string) bool {
	if m.User != "" && m.User != "*" && m.User != user {
		return false
	}
	if m.Group == "" || m.Group == "*" {
		return true
	}
	for _, group := range groups {
		if group == m.Group {
			return true
		}
	}
	return false
}

func validateBudgetsContract(cfg *RouterConfig) error {
	const path = "global.services.budgets"
	budgets := cfg.Budgets
	if budgets.Unpriced != "" && budgets.Unpriced != BudgetUnpricedRefuse && budgets.Unpriced != BudgetUnpricedAllow {
		return fmt.Errorf("%s.unpriced %q must be refuse or allow", path, budgets.Unpriced)
	}
	if budgets.OutputReserveTokens < 0 {
		return fmt.Errorf("%s.output_reserve_tokens must not be negative", path)
	}
	if run := budgets.Run; run != nil {
		if !(run.Limit > 0) {
			return fmt.Errorf("%s.run.limit must be greater than 0", path)
		}
		if run.IdleTimeoutSeconds < 0 {
			return fmt.Errorf("%s.run.idle_timeout_seconds must not be negative", path)
		}
	}
	names := make(map[string]bool, len(budgets.Users))
	for i, rule := range budgets.Users {
		rulePath := fmt.Sprintf("%s.users[%d]", path, i)
		switch {
		case strings.TrimSpace(rule.Name) == "":
			return fmt.Errorf("%s.name is required", rulePath)
		case names[rule.Name]:
			return fmt.Errorf("%s.name %q is used twice", rulePath, rule.Name)
		case !(rule.Limit > 0):
			return fmt.Errorf("%s.limit must be greater than 0", rulePath)
		case !spendbudget.Window(rule.Window).Valid():
			return fmt.Errorf("%s.window %q must be hour, day, week or month", rulePath, rule.Window)
		}
		names[rule.Name] = true
	}
	return nil
}
