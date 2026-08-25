package accesscapacity

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func runReplicaFailover(
	ctx context.Context,
	options *redis.Options,
	config Config,
	replicas []*replica,
	fixture Fixture,
	prefix string,
	result *Failover,
) (int64, error) {
	if len(replicas) < 2 {
		return 0, fmt.Errorf("replica failover requires at least two replicas")
	}
	credential, target := fixture.Credentials[0], fixture.Targets[0]
	failed := replicas[0]
	if err := failed.close(); err != nil {
		result.Errors++
		return 0, err
	}
	transitionStarted := time.Now()
	failedRequest := executeEvent(ctx, failed, credential, target, "capacity-failover-reroute")
	result.FailedReplicaRequestRejected = failedRequest.err != nil
	if !result.FailedReplicaRequestRejected {
		result.Errors++
	}
	routed := executeEvent(ctx, replicas[1], credential, target, "capacity-failover-reroute")
	result.TransitionMS = milliseconds(time.Since(transitionStarted))
	result.ReroutedRequestAllowed = routed.err == nil && routed.produced
	produced := int64(0)
	if result.ReroutedRequestAllowed {
		produced++
	} else {
		result.Errors++
	}
	allowedForSentinel := 1 + int(produced)
	for allowedForSentinel < config.RequestLimit-1 {
		admissionID := fmt.Sprintf("capacity-failover-%06d", allowedForSentinel)
		current := replicas[1+(allowedForSentinel%(len(replicas)-1))]
		event := executeEvent(ctx, current, credential, target, admissionID)
		if event.err != nil || !event.produced {
			result.Errors++
			break
		}
		produced++
		allowedForSentinel++
	}
	replacementList, err := newReplicas(options, 1, fixture, prefix)
	if err != nil {
		result.Errors++
		return produced, err
	}
	replacement := replacementList[0]
	defer func() { _ = replacement.close() }()
	lastAllowed := executeEvent(ctx, replacement, credential, target, "capacity-failover-replacement")
	result.ReplacementRequestAllowed = lastAllowed.err == nil && lastAllowed.produced
	if result.ReplacementRequestAllowed {
		produced++
		allowedForSentinel++
	} else {
		result.Errors++
	}
	blocked := executeEvent(ctx, replacement, credential, target, "capacity-failover-over-limit")
	result.PostLimitRequestDenied = blocked.err == nil &&
		blocked.disposition == quotaruntime.AdmissionRateLimited && !blocked.produced
	if !result.PostLimitRequestDenied {
		result.Errors++
	}
	result.ExpectedAllowedForSentinel = config.RequestLimit
	result.ObservedAllowedForSentinel = allowedForSentinel
	result.ReplicasAfter = config.Replicas
	result.GlobalQuotaStateConsistent = result.FailedReplicaRequestRejected &&
		result.ReroutedRequestAllowed && result.ReplacementRequestAllowed &&
		result.PostLimitRequestDenied && result.ObservedAllowedForSentinel == result.ExpectedAllowedForSentinel
	if result.Errors != 0 || !result.GlobalQuotaStateConsistent {
		return produced, fmt.Errorf("router-replica failover did not preserve global quota state")
	}
	return produced, nil
}
