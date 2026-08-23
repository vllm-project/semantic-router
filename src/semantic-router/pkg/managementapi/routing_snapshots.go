package managementapi

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type RoutingSnapshotMetadata struct {
	NamespaceID     string     `json:"namespaceId"`
	RoutingRevision int64      `json:"routingRevision"`
	ContentDigest   string     `json:"contentDigest"`
	Status          string     `json:"status"`
	FailureReason   string     `json:"failureReason,omitempty"`
	MemberCount     int        `json:"memberCount"`
	CreatedAt       time.Time  `json:"createdAt"`
	ActivatedAt     *time.Time `json:"activatedAt,omitempty"`
}

type RoutingSnapshotMember struct {
	ResourceType     string `json:"resourceType"`
	ResourceID       string `json:"resourceId"`
	ResourceRevision int64  `json:"resourceRevision"`
}

type RoutingSnapshotRecord struct {
	Metadata RoutingSnapshotMetadata  `json:"metadata"`
	Members  []RoutingSnapshotMember  `json:"members"`
	Export   routingsnapshot.Snapshot `json:"export"`
}

type RoutingSnapshotPage = Page[RoutingSnapshotMetadata]

type RoutingSnapshotDetail struct {
	Data RoutingSnapshotRecord `json:"data"`
}
