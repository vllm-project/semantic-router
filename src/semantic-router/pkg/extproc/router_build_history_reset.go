package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// verifyHistoryResetTriggerWiring refuses to activate an enabled policy that
// has no topic-continuity producer in this process. Static validation proves
// the configuration is well formed and that its trigger resolves inside the
// recipe; it cannot prove that this router has a producer wired, and a route
// without one could only preserve or reject every request.
func (r *OpenAIRouter) verifyHistoryResetTriggerWiring(cfg *config.RouterConfig) error {
	if r == nil || cfg == nil {
		return nil
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		if !decision.GetHistoryResetConfig().IsEnabled() {
			continue
		}
		if r.HistoryResetTriggers == nil {
			return fmt.Errorf(
				"decision %q: %s: no topic-continuity producer is wired in this router",
				decision.Name,
				config.HistoryResetTriggerUnavailable,
			)
		}
	}
	return nil
}

// verifyContextRecoveryAgreement enforces one recovery backend and total-byte
// budget across every reachable decision. The store is built once and reused
// for the process, so two decisions naming different backends would silently
// share whichever one the first recovery-using request created.
func verifyContextRecoveryAgreement(cfg *config.RouterConfig) error {
	var (
		store    string
		total    int
		declared string
	)
	for _, decision := range cfg.AllRoutingDecisions() {
		settings, err := resolveContextRecoverySettings(&decision)
		if err != nil {
			return fmt.Errorf("decision %q: %w", decision.Name, err)
		}
		if settings == nil {
			continue
		}
		current := strings.TrimSpace(settings.Store)
		if declared == "" {
			store, total, declared = current, settings.MaxTotalBytes, decision.Name
			continue
		}
		if !strings.EqualFold(store, current) {
			return fmt.Errorf(
				"decisions %q and %q request different context recovery stores (%q and %q)",
				declared, decision.Name, store, current,
			)
		}
		if settings.MaxTotalBytes != total {
			return fmt.Errorf(
				"decisions %q and %q request different context recovery max_total_bytes (%d and %d)",
				declared, decision.Name, total, settings.MaxTotalBytes,
			)
		}
	}
	return nil
}
