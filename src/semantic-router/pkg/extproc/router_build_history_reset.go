package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
