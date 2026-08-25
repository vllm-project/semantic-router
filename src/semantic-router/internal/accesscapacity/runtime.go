package accesscapacity

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

type staticDelegationBarriers struct{}

func (staticDelegationBarriers) CheckDelegation(
	context.Context,
	managementauth.DelegationBarrierCheck,
) (managementauth.DelegationBarrierState, error) {
	return managementauth.DelegationBarrierState{Ready: true}, nil
}

type replica struct {
	client  *redis.Client
	runtime *accessruntime.Runtime
}

func (r *replica) close() error {
	if r == nil || r.client == nil {
		return nil
	}
	return r.client.Close()
}

func publishCapacityProjection(
	ctx context.Context,
	client *redis.Client,
	prefix string,
	publication accesspublisher.Publication,
) error {
	store, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: client, KeyPrefix: prefix, ReplicaLease: 30 * time.Second,
	})
	if err != nil {
		return err
	}
	plan, err := store.Prepare(ctx, publication)
	if err != nil {
		return fmt.Errorf("prepare access projection: %w", err)
	}
	if plan.Restrictive() {
		if err := store.InstallBarriers(ctx, plan); err != nil {
			return fmt.Errorf("install access projection barriers: %w", err)
		}
	}
	if err := store.Stage(ctx, plan); err != nil {
		return fmt.Errorf("stage access projection: %w", err)
	}
	if err := store.ValidateStaged(ctx, plan); err != nil {
		return fmt.Errorf("validate access projection: %w", err)
	}
	if err := store.Activate(ctx, plan); err != nil {
		return fmt.Errorf("activate access projection: %w", err)
	}
	for {
		complete, compactErr := store.Compact(ctx, plan, 1000)
		if compactErr != nil {
			return fmt.Errorf("compact access projection: %w", compactErr)
		}
		if complete {
			break
		}
	}
	if err := store.MarkApplied(ctx, plan); err != nil {
		return fmt.Errorf("mark access projection applied: %w", err)
	}
	if err := store.ClearAppliedBarriers(ctx, plan); err != nil {
		return fmt.Errorf("clear access projection barriers: %w", err)
	}
	readiness, err := store.Readiness(ctx, publication.NamespaceID, publication.QuotaPartition)
	if err != nil || !readiness.Ready || readiness.ProjectorLag != 0 {
		return fmt.Errorf("access projection is not ready: readiness=%+v error=%w", readiness, err)
	}
	return nil
}

func newReplicas(
	options *redis.Options,
	count int,
	fixture Fixture,
	prefix string,
) ([]*replica, error) {
	result := make([]*replica, 0, count)
	for range count {
		client := redis.NewClient(options)
		reader, err := accessruntime.NewRedisProjectionReader(accessruntime.RedisProjectionReaderOptions{
			Client: client, KeyPrefix: prefix,
		})
		if err != nil {
			_ = client.Close()
			closeReplicas(result)
			return nil, err
		}
		engine, err := quotaruntime.NewRedisEngine(client, quotaruntime.RedisEngineOptions{
			KeyPrefix: prefix, FinalizationMarkerTTL: time.Hour,
		})
		if err != nil {
			_ = client.Close()
			closeReplicas(result)
			return nil, err
		}
		runtime, err := accessruntime.New(accessruntime.RuntimeOptions{
			Reader: reader, Engine: engine, APIKeyPeppers: fixture.Keyring,
			DelegationPeppers: fixture.Keyring, DelegationAudience: "vllm-sr-inference",
			DelegationBarriers: staticDelegationBarriers{}, KeyPrefix: prefix,
		})
		if err != nil {
			_ = client.Close()
			closeReplicas(result)
			return nil, err
		}
		result = append(result, &replica{client: client, runtime: runtime})
	}
	return result, nil
}

func closeReplicas(replicas []*replica) {
	for _, current := range replicas {
		_ = current.close()
	}
}

func verifyIsolation(ctx context.Context, replicas []*replica, fixture Fixture) (samples, violations int) {
	sampleCount := min(32, len(fixture.Credentials))
	for sample := range sampleCount {
		index := sample * len(fixture.Credentials) / sampleCount
		current := replicas[sample%len(replicas)]
		authentication, err := current.runtime.Authenticate(ctx, accessruntime.AuthenticationRequest{
			Credential: fixture.Credentials[index],
		})
		if err != nil || !authentication.Result.Allowed() {
			violations++
			continue
		}
		wrongTarget := fixtureEntrypointA
		if fixture.Targets[index] == wrongTarget {
			wrongTarget = fixtureEntrypointB
		}
		result, err := current.runtime.Authorize(ctx, accessruntime.AuthorizationRequest{
			Session: authentication.Session,
			Target: accessruntime.Target{
				ResourceType: accesscontrol.GrantResourceEntrypoint,
				ResourceID:   accesscontrol.ResourceID(wrongTarget),
				Permission:   accesscontrol.GrantPermissionInvoke,
			},
		})
		if err != nil || result.Result.Disposition != quotaruntime.AdmissionForbidden {
			violations++
		}
	}
	return sampleCount, violations
}
