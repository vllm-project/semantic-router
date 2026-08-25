package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

type managedAccessLifecycle struct {
	ctx                 context.Context
	management          *managedAccessClient
	fixture             *managedAccessFixture
	replicaSessions     []managedAccessReplicaSession
	gatewayClient       *http.Client
	gatewayBaseURL      string
	publicationRevision uint64
	startedAt           time.Time
	setDetails          func(map[string]interface{})
	successful          int64
	limited             int64
	inputTokens         int64
	outputTokens        int64
	settledCost         string
}

func (lifecycle *managedAccessLifecycle) run() error {
	if err := lifecycle.verifyAuthenticationAndDiscovery(); err != nil {
		return err
	}
	if err := lifecycle.invokeInitialRequest(); err != nil {
		return err
	}
	if err := lifecycle.rotateCredential(); err != nil {
		return err
	}
	if err := lifecycle.enforceQuotaBoundary(); err != nil {
		return err
	}
	if err := lifecycle.disableCredential(); err != nil {
		return err
	}
	lifecycle.reportDetails()
	return nil
}

func (lifecycle *managedAccessLifecycle) verifyAuthenticationAndDiscovery() error {
	status, _, err := managedAccessModelsRequest(
		lifecycle.ctx, lifecycle.gatewayClient, lifecycle.gatewayBaseURL, "",
	)
	if err != nil {
		return err
	}
	if status != http.StatusUnauthorized {
		return fmt.Errorf("unauthenticated /v1/models status = %d, want 401", status)
	}
	for index := range lifecycle.replicaSessions {
		replica := lifecycle.replicaSessions[index]
		status, _, requestErr := managedAccessModelsRequest(lifecycle.ctx, replica.client, replica.baseURL, "")
		if requestErr != nil {
			return requestErr
		}
		if status != http.StatusUnauthorized {
			return fmt.Errorf("Router replica %d unauthenticated /v1/models status = %d, want 401", index, status)
		}
	}
	fixture := lifecycle.fixture
	if err := waitManagedAccessDiscovery(
		lifecycle.ctx, lifecycle.gatewayClient, lifecycle.gatewayBaseURL, fixture.secret,
		fixture.authorizedName, fixture.hiddenName,
	); err != nil {
		return err
	}
	return waitManagedAccessReplicaDiscovery(
		lifecycle.ctx, lifecycle.replicaSessions, fixture.secret,
		fixture.authorizedName, fixture.hiddenName,
	)
}

func (lifecycle *managedAccessLifecycle) invokeInitialRequest() error {
	fixture := lifecycle.fixture
	invocation := invokeManagedAccessModel(
		lifecycle.ctx, lifecycle.gatewayClient, lifecycle.gatewayBaseURL,
		fixture.secret, fixture.authorizedName, 0,
	)
	if invocation.err != nil {
		return fmt.Errorf("public managed-access invocation: %w", invocation.err)
	}
	if invocation.status != http.StatusOK || invocation.input <= 0 || invocation.output <= 0 {
		return fmt.Errorf("public managed-access invocation omitted authoritative usage")
	}
	lifecycle.successful = 1
	lifecycle.inputTokens = invocation.input
	lifecycle.outputTokens = invocation.output
	return nil
}

func (lifecycle *managedAccessLifecycle) rotateCredential() error {
	fixture := lifecycle.fixture
	oldSecret := fixture.secret
	var rotated managedAccessIssuedKey
	if _, _, err := lifecycle.management.request(
		lifecycle.ctx, fixture.namespaceID, http.MethodPost,
		"/api-keys/"+fixture.keyID+"/credentials:rotate", "managed-access-rotate-"+fixture.keyID,
		map[string]interface{}{"overlapSeconds": 0, "revealable": false},
		http.Header{"If-Match": []string{`"key:` + strconv.FormatUint(fixture.keyRevision, 10) + `"`}},
		[]int{http.StatusOK}, &rotated,
	); err != nil {
		return fmt.Errorf("rotate fixture API-key credential: %w", err)
	}
	if rotated.Data.KeyID != fixture.keyID || rotated.Data.Status != "active" ||
		rotated.Data.Revision <= fixture.keyRevision || rotated.Secret == "" {
		return fmt.Errorf("rotated API-key response is incomplete")
	}
	fixture.secret = rotated.Secret
	fixture.keyRevision = rotated.Data.Revision
	rotated.Secret = ""
	revision, err := waitManagedAccessReplicaConvergence(
		lifecycle.ctx, lifecycle.management, fixture.namespaceID, lifecycle.publicationRevision,
	)
	if err != nil {
		return fmt.Errorf("rotated credential did not converge on both Router replicas: %w", err)
	}
	lifecycle.publicationRevision = revision
	if err := waitManagedAccessCredentialDenied(
		lifecycle.ctx, lifecycle.gatewayClient, lifecycle.gatewayBaseURL, oldSecret,
	); err != nil {
		return fmt.Errorf("rotated credential remained usable: %w", err)
	}
	if err := waitManagedAccessReplicaCredentialDenied(
		lifecycle.ctx, lifecycle.replicaSessions, oldSecret,
	); err != nil {
		return fmt.Errorf("rotated credential remained usable on an exact Router replica: %w", err)
	}
	return lifecycle.verifyRotatedDiscovery()
}

func (lifecycle *managedAccessLifecycle) verifyRotatedDiscovery() error {
	fixture := lifecycle.fixture
	if err := waitManagedAccessDiscovery(
		lifecycle.ctx, lifecycle.gatewayClient, lifecycle.gatewayBaseURL, fixture.secret,
		fixture.authorizedName, fixture.hiddenName,
	); err != nil {
		return fmt.Errorf("rotated credential did not preserve authorized discovery: %w", err)
	}
	if err := waitManagedAccessReplicaDiscovery(
		lifecycle.ctx, lifecycle.replicaSessions, fixture.secret,
		fixture.authorizedName, fixture.hiddenName,
	); err != nil {
		return fmt.Errorf("rotated credential did not converge on exact Router replicas: %w", err)
	}
	return nil
}

func (lifecycle *managedAccessLifecycle) enforceQuotaBoundary() error {
	remaining, err := lifecycle.readRequestCapacity()
	if err != nil {
		return err
	}
	results := managedAccessReplicaBurst(
		lifecycle.ctx, lifecycle.replicaSessions, lifecycle.fixture.secret,
		lifecycle.fixture.authorizedName, int(remaining+1),
	)
	if err := lifecycle.recordBurst(results); err != nil {
		return err
	}
	if lifecycle.successful != managedAccessRequestLimit || lifecycle.limited != 1 {
		return fmt.Errorf(
			"global RPM boundary admitted %d and limited %d requests, want %d and 1",
			lifecycle.successful, lifecycle.limited, managedAccessRequestLimit,
		)
	}
	if lifecycle.inputTokens <= 0 || lifecycle.outputTokens <= 0 {
		return fmt.Errorf("successful managed-access responses omitted authoritative token usage")
	}
	fixture := lifecycle.fixture
	if _, err := waitManagedAccessRequestBoundary(
		lifecycle.ctx, lifecycle.management, fixture.namespaceID, fixture.keyID,
	); err != nil {
		return err
	}
	return lifecycle.verifySettlement()
}

func (lifecycle *managedAccessLifecycle) readRequestCapacity() (int64, error) {
	fixture := lifecycle.fixture
	_, meter, err := waitManagedAccessRequestMeter(
		lifecycle.ctx, lifecycle.management, fixture.namespaceID, fixture.keyID,
	)
	if err != nil {
		return 0, err
	}
	used, err := strconv.ParseInt(meter.Used, 10, 64)
	if err != nil || used != lifecycle.successful {
		return 0, fmt.Errorf("live request meter used = %q before burst, want %d", meter.Used, lifecycle.successful)
	}
	remaining := managedAccessRequestLimit - lifecycle.successful
	if meter.Remaining == nil || *meter.Remaining != strconv.FormatInt(remaining, 10) {
		return 0, fmt.Errorf("live request meter remaining = %v, want %d", meter.Remaining, remaining)
	}
	if meter.Completeness != "complete" || meter.Capacity != "available" {
		return 0, fmt.Errorf(
			"live request meter state = %s/%s, want complete/available",
			meter.Completeness, meter.Capacity,
		)
	}
	return remaining, nil
}

func (lifecycle *managedAccessLifecycle) recordBurst(results []managedAccessInvocation) error {
	for _, result := range results {
		if result.err != nil {
			return result.err
		}
		switch result.status {
		case http.StatusOK:
			lifecycle.successful++
			lifecycle.inputTokens += result.input
			lifecycle.outputTokens += result.output
		case http.StatusTooManyRequests:
			lifecycle.limited++
		default:
			return fmt.Errorf("managed-access burst returned unexpected status %d", result.status)
		}
	}
	return nil
}

func (lifecycle *managedAccessLifecycle) verifySettlement() error {
	expectedTokens := lifecycle.inputTokens + lifecycle.outputTokens
	usage, quota, err := waitManagedAccessSettlement(
		lifecycle.ctx, lifecycle.management, *lifecycle.fixture, lifecycle.startedAt,
		lifecycle.successful, lifecycle.inputTokens, lifecycle.outputTokens, expectedTokens,
	)
	if err != nil {
		return err
	}
	if err := assertManagedAccessActualQuota(quota, expectedTokens, expectedTokens); err != nil {
		return err
	}
	lifecycle.settledCost = usage.Totals.Costs[0].KnownAmount
	return nil
}

func (lifecycle *managedAccessLifecycle) disableCredential() error {
	fixture := lifecycle.fixture
	var disabled managedAccessKeyDetail
	if _, _, err := lifecycle.management.request(
		lifecycle.ctx, fixture.namespaceID, http.MethodPost,
		"/api-keys/"+fixture.keyID+":disable", "managed-access-disable-"+fixture.keyID,
		map[string]interface{}{},
		http.Header{"If-Match": []string{`"key:` + strconv.FormatUint(fixture.keyRevision, 10) + `"`}},
		[]int{http.StatusOK}, &disabled,
	); err != nil {
		return fmt.Errorf("disable fixture API key: %w", err)
	}
	if disabled.Data.KeyID != fixture.keyID || disabled.Data.Status != "disabled" ||
		disabled.Data.Revision <= fixture.keyRevision {
		return fmt.Errorf("disabled API-key response is incomplete")
	}
	if _, err := waitManagedAccessReplicaConvergence(
		lifecycle.ctx, lifecycle.management, fixture.namespaceID, lifecycle.publicationRevision,
	); err != nil {
		return fmt.Errorf("disabled credential did not converge on both Router replicas: %w", err)
	}
	if err := waitManagedAccessCredentialDenied(
		lifecycle.ctx, lifecycle.gatewayClient, lifecycle.gatewayBaseURL, fixture.secret,
	); err != nil {
		return fmt.Errorf("disabled credential remained usable: %w", err)
	}
	if err := waitManagedAccessReplicaCredentialDenied(
		lifecycle.ctx, lifecycle.replicaSessions, fixture.secret,
	); err != nil {
		return fmt.Errorf("disabled credential remained usable on an exact Router replica: %w", err)
	}
	return nil
}

func (lifecycle *managedAccessLifecycle) reportDetails() {
	if lifecycle.setDetails == nil {
		return
	}
	lifecycle.setDetails(map[string]interface{}{
		"router_replicas": 2, "public_gateway_requests": 1, "authorized_models": 1,
		"rpm_limit": managedAccessRequestLimit, "successful_requests": lifecycle.successful,
		"rate_limited_requests": lifecycle.limited, "settled_input_tokens": lifecycle.inputTokens,
		"settled_output_tokens": lifecycle.outputTokens, "settled_cost_usd": lifecycle.settledCost,
		"old_credential_denied": true, "disabled_key_denied": true,
	})
}
