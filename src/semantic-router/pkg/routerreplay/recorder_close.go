package routerreplay

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Close drains receipts before releasing storage. If the drain budget expires,
// ownership remains with a cleanup goroutine until the shared writer exits.
// Repeated calls return the same result and never close storage twice.
func (r *Recorder) Close() error {
	r.closeOnce.Do(func() {
		r.closeErr = r.DrainOutcomes()
		if r.closeErr == nil {
			r.closeErr = r.storage.Close()
			return
		}
		go func() {
			<-r.outcomes.done
			if err := r.storage.Close(); err != nil {
				logging.Errorf("Deferred replay storage cleanup failed: %v", err)
			}
		}()
	})
	return r.closeErr
}
