package accesspublisher

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	FieldPublicationID        = "publication_id"
	FieldPendingPublicationID = "pending_publication_id"
	FieldState                = "state"
	FieldPendingState         = "pending_state"
)

type PointerState string

const (
	PointerStateActive    PointerState = "active"
	PointerStateTombstone PointerState = "tombstone"
)

// PublicationGate is the coupled access/routing switch read by every data
// plane before it selects pointer fields. Access and routing gates are changed
// by one partition-local Lua operation and must name the same publication.
type PublicationGate struct {
	PublicationID     string
	Revision          uint64
	RuntimeEpoch      uint64
	PublicationDigest string
	ManifestDigest    string
	SnapshotDigest    string
	SnapshotKey       string
}

func ParsePublicationGate(values map[string]string) (PublicationGate, error) {
	var gate PublicationGate
	gate.PublicationID = values[FieldPublicationID]
	gate.PublicationDigest = values["publication_digest"]
	gate.ManifestDigest = values["manifest_digest"]
	gate.SnapshotDigest = values["snapshot_digest"]
	gate.SnapshotKey = values["snapshot_key"]
	var err error
	gate.Revision, err = strconv.ParseUint(values["revision"], 10, 64)
	if err != nil || gate.Revision == 0 {
		return PublicationGate{}, fmt.Errorf("publication gate revision is invalid")
	}
	gate.RuntimeEpoch, err = strconv.ParseUint(values["runtime_epoch"], 10, 64)
	if err != nil || gate.RuntimeEpoch == 0 {
		return PublicationGate{}, fmt.Errorf("publication gate runtime epoch is invalid")
	}
	if strings.TrimSpace(gate.PublicationID) == "" || !validDigest(gate.PublicationDigest) {
		return PublicationGate{}, fmt.Errorf("publication gate identity is invalid")
	}
	if gate.ManifestDigest != "" && !validDigest(gate.ManifestDigest) {
		return PublicationGate{}, fmt.Errorf("publication gate manifest digest is invalid")
	}
	if gate.SnapshotDigest != "" && !validDigest(gate.SnapshotDigest) {
		return PublicationGate{}, fmt.Errorf("publication gate snapshot digest is invalid")
	}
	return gate, nil
}

// SelectPointer returns the security fields selected by the namespace gate.
// A staged future publication is invisible. During bounded compaction the
// active publication is selected from pending_ fields; after compaction it is
// selected from unprefixed fields. Any mismatch fails closed.
func SelectPointer(values map[string]string, gatePublicationID string) (map[string]string, PointerState, error) {
	if strings.TrimSpace(gatePublicationID) == "" {
		return nil, "", fmt.Errorf("publication gate is required")
	}
	if values[FieldPendingPublicationID] == gatePublicationID {
		return selectFieldPrefix(values, "pending_")
	}
	if values[FieldPublicationID] == gatePublicationID {
		return selectFieldPrefix(values, "")
	}
	return nil, "", fmt.Errorf("pointer does not match the active publication")
}

func selectFieldPrefix(values map[string]string, prefix string) (map[string]string, PointerState, error) {
	state := PointerState(values[prefix+FieldState])
	if state != PointerStateActive && state != PointerStateTombstone {
		return nil, "", fmt.Errorf("pointer state is invalid")
	}
	selected := make(map[string]string)
	for field, value := range values {
		if prefix == "" {
			if strings.HasPrefix(field, "pending_") {
				continue
			}
			selected[field] = value
			continue
		}
		if strings.HasPrefix(field, prefix) {
			selected[strings.TrimPrefix(field, prefix)] = value
		}
	}
	if selected[FieldPublicationID] == "" {
		return nil, "", fmt.Errorf("selected pointer publication is absent")
	}
	return selected, state, nil
}
