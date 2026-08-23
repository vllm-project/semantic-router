package providercatalog

import (
	"fmt"
	"regexp"
	"sort"
)

type CapabilityPlane string

const (
	CapabilityPlaneControl CapabilityPlane = "control"
	CapabilityPlaneData    CapabilityPlane = "data"
)

var rolloutGroupIDPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)

// RolloutGroup is a stable deployment identity. Replica or Pod names are
// deliberately absent: autoscaling must not change catalog activation policy.
type RolloutGroup struct {
	Plane CapabilityPlane
	ID    string
}

func (group RolloutGroup) Validate() error {
	if group.Plane != CapabilityPlaneControl && group.Plane != CapabilityPlaneData {
		return fmt.Errorf("provider Catalog capability plane is invalid")
	}
	if !rolloutGroupIDPattern.MatchString(group.ID) {
		return fmt.Errorf("provider Catalog rollout group ID is invalid")
	}
	return nil
}

func (group RolloutGroup) Key() string {
	return string(group.Plane) + "/" + group.ID
}

func CanonicalRolloutGroups(groups []RolloutGroup) ([]RolloutGroup, error) {
	if len(groups) == 0 {
		return nil, fmt.Errorf("at least one Provider Catalog rollout group is required")
	}
	result := append([]RolloutGroup(nil), groups...)
	for _, group := range result {
		if err := group.Validate(); err != nil {
			return nil, err
		}
	}
	sort.Slice(result, func(left, right int) bool {
		return result[left].Key() < result[right].Key()
	})
	for index := 1; index < len(result); index++ {
		if result[index].Key() == result[index-1].Key() {
			return nil, fmt.Errorf("provider Catalog rollout group %q is duplicated", result[index].Key())
		}
	}
	return result, nil
}
