package config

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/spendbudget"
)

func parseBudgetsDoc(budgets string) (*RouterConfig, error) {
	doc := drainTimeoutDoc + "global:\n  services:\n    budgets:\n" + budgets
	return ParseYAMLBytes([]byte(doc))
}

func TestBudgetsLoadWithDefaultsAndExportAsWritten(t *testing.T) {
	cfg, err := parseBudgetsDoc(`      run:
        limit: 2.5
      users:
        - name: daily
          match:
            group: eng
          limit: 10
          window: day
`)
	if err != nil {
		t.Fatal(err)
	}
	budgets := cfg.Budgets
	if !budgets.Enabled() || budgets.EffectiveCurrency() != "USD" || !budgets.RefusesUnpriced() ||
		budgets.EffectiveOutputReserveTokens() != DefaultBudgetOutputReserveTokens {
		t.Fatalf("defaults: %+v", budgets)
	}
	if budgets.Run.IdleTimeout() != 24*time.Hour {
		t.Fatalf("run idle timeout %v, want a day", budgets.Run.IdleTimeout())
	}
	if rule := budgets.Users[0]; rule.Match.Group != "eng" || rule.Limit != 10 || spendbudget.Window(rule.Window) != spendbudget.WindowDay {
		t.Fatalf("user rule %+v", rule)
	}
	exported := CanonicalGlobalFromRouterConfig(cfg).Services.Budgets
	if exported.Run == nil || exported.Run.Limit != 2.5 || exported.Currency != "" || len(exported.Users) != 1 {
		t.Fatalf("exported %+v", exported)
	}
	if (BudgetsConfig{}).Enabled() {
		t.Fatal("an empty budgets block enforces a budget")
	}
}

func TestBudgetsRejectWhatCannotBeEnforced(t *testing.T) {
	for name, test := range map[string]struct{ block, want string }{
		"zero run limit": {"      run:\n        limit: 0\n", "global.services.budgets.run.limit must be greater than 0"},
		"negative user limit": {
			"      users:\n        - name: a\n          limit: -1\n          window: day\n",
			"global.services.budgets.users[0].limit must be greater than 0",
		},
		"unknown window": {
			"      users:\n        - name: a\n          limit: 1\n          window: fortnight\n",
			`global.services.budgets.users[0].window "fortnight" must be hour, day, week or month`,
		},
		"duplicate name": {
			"      users:\n        - name: a\n          limit: 1\n          window: day\n        - name: a\n          limit: 2\n          window: month\n",
			`global.services.budgets.users[1].name "a" is used twice`,
		},
		"missing name": {
			"      users:\n        - limit: 1\n          window: day\n",
			"global.services.budgets.users[0].name is required",
		},
		"unknown unpriced policy": {
			"      unpriced: ignore\n      run:\n        limit: 1\n",
			`global.services.budgets.unpriced "ignore" must be refuse or allow`,
		},
		"negative idle timeout": {
			"      run:\n        limit: 1\n        idle_timeout_seconds: -5\n",
			"global.services.budgets.run.idle_timeout_seconds must not be negative",
		},
	} {
		_, err := parseBudgetsDoc(test.block)
		if err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("%s: error %v, want it to contain %q", name, err, test.want)
		}
	}
}
