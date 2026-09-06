package ratelimit

import (
	"errors"
	"io"
)

// Close releases closeable providers.
func (r *RateLimitResolver) Close() error {
	if r == nil {
		return nil
	}
	var errs []error
	for _, provider := range r.providers {
		closer, ok := provider.(io.Closer)
		if !ok {
			continue
		}
		if err := closer.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
