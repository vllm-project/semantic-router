// Package policymanagement owns the Router-native AccessPolicy,
// RateLimitPolicy, and policy-binding management use cases.
package policymanagement

import "errors"

var (
	ErrInvalidRequest     = errors.New("policy management request is invalid")
	ErrNotFound           = errors.New("policy management resource not found")
	ErrAlreadyExists      = errors.New("policy management resource already exists")
	ErrRevisionConflict   = errors.New("policy management revision conflict")
	ErrResourceInUse      = errors.New("policy management resource is in use")
	ErrAllocationConflict = errors.New("subject already has an active quota allocation")
	ErrCounterSemantics   = errors.New("rate-limit rule counter semantics require a new rule ID")
	ErrUnknownUsageFence  = errors.New("an unresolved usage fence protects this quota resource")
	ErrUnavailable        = errors.New("policy management service is unavailable")
)
