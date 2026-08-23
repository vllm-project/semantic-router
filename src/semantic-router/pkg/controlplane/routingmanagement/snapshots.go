package routingmanagement

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// SnapshotStatus is the immutable publication lifecycle recorded for one
// routing revision. A revision's bytes and members never change; activation
// only advances its lifecycle.
type SnapshotStatus string

const (
	SnapshotStatusStaged  SnapshotStatus = "staged"
	SnapshotStatusActive  SnapshotStatus = "active"
	SnapshotStatusFailed  SnapshotStatus = "failed"
	SnapshotStatusRetired SnapshotStatus = "retired"
)

type SnapshotMetadata struct {
	NamespaceID     string
	RoutingRevision int64
	ContentDigest   string
	Status          SnapshotStatus
	FailureReason   string
	MemberCount     int
	CreatedAt       time.Time
	ActivatedAt     *time.Time
}

type SnapshotMember struct {
	ResourceType     string
	ResourceID       string
	ResourceRevision int64
}

// SnapshotDetail is one immutable, self-contained routing export together
// with the exact resource revisions that produced it.
type SnapshotDetail struct {
	Metadata SnapshotMetadata
	Members  []SnapshotMember
	Export   routingsnapshot.Snapshot
}

type SnapshotPageRequest struct {
	PageSize int
	Cursor   string
}

type SnapshotListQuery struct {
	Limit          int
	BeforeRevision *int64
}

// Valid reports whether status belongs to the closed publication lifecycle.
func (status SnapshotStatus) Valid() bool {
	return status == SnapshotStatusStaged || status == SnapshotStatusActive ||
		status == SnapshotStatusFailed || status == SnapshotStatusRetired
}
