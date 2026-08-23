package managementapi

import "time"

// AccessStatistics is a permission-projected control-plane cardinality
// snapshot. Optional fields are omitted when the caller lacks that resource's
// read permission; an omitted field never means zero.
type AccessStatistics struct {
	AsOf               time.Time      `json:"asOf"`
	ExpiringBefore     time.Time      `json:"expiringBefore"`
	Users              *WholeQuantity `json:"users,omitempty"`
	Teams              *WholeQuantity `json:"teams,omitempty"`
	ActiveAPIKeys      *WholeQuantity `json:"activeApiKeys,omitempty"`
	ExpiringAPIKeys    *WholeQuantity `json:"expiringApiKeys,omitempty"`
	AccessPolicies     *WholeQuantity `json:"accessPolicies,omitempty"`
	ActiveRatePolicies *WholeQuantity `json:"activeRatePolicies,omitempty"`
}
