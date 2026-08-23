package accesscontrol

import (
	"strings"
	"time"
)

// Namespace is the isolation and quota-atomicity boundary.
type Namespace struct {
	ID               NamespaceID
	Name             string
	QuotaPartitionID QuotaPartitionID
	BillingCurrency  string
	Status           NamespaceStatus
	Revision         Revision
	RuntimeEpoch     uint64
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

func (n Namespace) Validate() error {
	return joinValidation(
		validateRequired("id", string(n.ID)),
		validateRequired("name", n.Name),
		validateRequired("quota_partition_id", string(n.QuotaPartitionID)),
		validateCurrency(n.BillingCurrency),
		validateNamespaceStatus(n.Status),
		validateRevision(n.Revision),
		validateRuntimeEpoch(n.RuntimeEpoch),
		validateTimestamps(n.CreatedAt, n.UpdatedAt),
	)
}

func validateRuntimeEpoch(epoch uint64) error {
	if epoch == 0 {
		return invalid("runtime_epoch", "must be positive")
	}
	return nil
}

func validateCurrency(currency string) error {
	if len(currency) != 3 || strings.ToUpper(currency) != currency {
		return invalid("billing_currency", "must be a three-letter uppercase ISO-4217 code")
	}
	for _, char := range currency {
		if char < 'A' || char > 'Z' {
			return invalid("billing_currency", "must contain only ASCII letters")
		}
	}
	return nil
}

func validateNamespaceStatus(status NamespaceStatus) error {
	if !status.Valid() {
		return invalid("status", "is not a valid namespace status")
	}
	return nil
}
