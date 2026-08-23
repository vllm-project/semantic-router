package outcomefeedback

import (
	"context"
	"errors"
	"fmt"
	"time"
)

type Repository interface {
	Record(context.Context, Caller, string, Request) (Receipt, error)
}

type AbuseDecision struct {
	Allowed    bool
	RetryAfter time.Duration
}

type AbuseLimiter interface {
	Allow(context.Context, Caller) (AbuseDecision, error)
}

type ServiceOptions struct {
	Repository Repository
	Limiter    AbuseLimiter
}

// Service is the public inference boundary. It validates only bounded public
// fields, applies a dedicated abuse budget, and delegates durable ownership
// and idempotency to one transaction. It never invokes inference quota APIs.
type Service struct {
	repository Repository
	limiter    AbuseLimiter
}

func NewService(options ServiceOptions) (*Service, error) {
	if options.Repository == nil {
		return nil, errors.New("outcome repository is required")
	}
	if options.Limiter == nil {
		return nil, errors.New("outcome abuse limiter is required")
	}
	return &Service{repository: options.Repository, limiter: options.Limiter}, nil
}

func (service *Service) Submit(
	ctx context.Context,
	caller Caller,
	idempotencyKey string,
	request Request,
) (Receipt, error) {
	if service == nil || service.repository == nil || service.limiter == nil {
		return Receipt{}, ErrUnavailable
	}
	if err := caller.Validate(); err != nil {
		return Receipt{}, err
	}
	if err := ValidateIdempotencyKey(idempotencyKey); err != nil {
		return Receipt{}, err
	}
	if err := request.Validate(); err != nil {
		return Receipt{}, err
	}
	decision, err := service.limiter.Allow(ctx, caller)
	if err != nil {
		return Receipt{}, fmt.Errorf("%w: apply outcome abuse limit", ErrUnavailable)
	}
	if !decision.Allowed {
		return Receipt{}, &RateLimitError{RetryAfter: decision.RetryAfter}
	}
	receipt, err := service.repository.Record(ctx, caller, idempotencyKey, request)
	if err != nil {
		return Receipt{}, err
	}
	return receipt, nil
}

type RateLimitError struct {
	RetryAfter time.Duration
}

func (err *RateLimitError) Error() string { return ErrRateLimited.Error() }

func (err *RateLimitError) Unwrap() error { return ErrRateLimited }
