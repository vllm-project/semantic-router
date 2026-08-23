package managementapi

// ResourceReference is the immutable result of a synchronous mutation. It is
// intentionally smaller than the mutable resource representation so a retry
// can return the original result after later edits.
type ResourceReference struct {
	Kind     string `json:"kind"`
	ID       string `json:"id"`
	Revision uint64 `json:"revision"`
}

// OperationReference is the immutable result of an asynchronous mutation.
// DesiredRevision exists only when the operation publishes a new runtime
// revision; ordinary synchronous CRUD never reports one.
type OperationReference struct {
	OperationID     string  `json:"operationId"`
	DesiredRevision *uint64 `json:"desiredRevision,omitempty"`
}

// MutationReceipt contains exactly one result kind. Idempotency metadata is
// present only for operations whose contract requires Idempotency-Key.
type MutationReceipt struct {
	Resource    *ResourceReference   `json:"resource,omitempty"`
	Operation   *OperationReference  `json:"operation,omitempty"`
	Idempotency *IdempotencyMetadata `json:"idempotency,omitempty"`
}

func NewResourceMutationReceipt(kind, id string, revision uint64, replayed *bool) MutationReceipt {
	receipt := MutationReceipt{Resource: &ResourceReference{Kind: kind, ID: id, Revision: revision}}
	if replayed != nil {
		receipt.Idempotency = &IdempotencyMetadata{Replayed: *replayed}
	}
	return receipt
}

func NewOperationMutationReceipt(operationID string, desiredRevision *uint64, replayed *bool) MutationReceipt {
	operation := &OperationReference{OperationID: operationID}
	if desiredRevision != nil {
		value := *desiredRevision
		operation.DesiredRevision = &value
	}
	receipt := MutationReceipt{Operation: operation}
	if replayed != nil {
		receipt.Idempotency = &IdempotencyMetadata{Replayed: *replayed}
	}
	return receipt
}
