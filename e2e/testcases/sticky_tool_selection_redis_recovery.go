package testcases

import (
	"context"
	"errors"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	stickyRedisNamespace        = "default"
	stickyRedisDeployment       = "redis"
	stickyRedisScaleTimeout     = 2 * time.Minute
	stickyRedisScaleInterval    = 2 * time.Second
	stickyRedisReconnectTimeout = 90 * time.Second
)

func init() {
	pkgtestcases.Register("sticky-tool-selection-redis-recovery", pkgtestcases.TestCase{
		Description: "Verify Redis-backed sticky state survives router restart and unavailable Redis falls back safely",
		Tags:        []string{"kubernetes", "plugin", "tool-selection", "sticky", "redis", "restart", "recovery", "fallback"},
		Fn:          testStickyToolSelectionRedisRecovery,
	})
}

func testStickyToolSelectionRedisRecovery(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] sticky tool-selection Redis recovery")
	}

	sessionID, headers, tools, retained, err := seedStickyRedisState(ctx, client, opts)
	if err != nil {
		return err
	}

	if restartErr := restartStickySemanticRouterContainer(ctx, client, opts); restartErr != nil {
		return restartErr
	}
	if readinessErr := waitForSemanticRouterReady(ctx, client, opts); readinessErr != nil {
		return readinessErr
	}

	sessions, err := openStickySessionPair(ctx, client, opts)
	if err != nil {
		return err
	}
	defer sessions.gateway.Close()
	defer sessions.backend.Close()

	afterRestart, restartBaseline, err := verifyStickyRedisRestartReuse(
		ctx,
		sessions,
		sessionID,
		headers,
		tools,
		retained,
	)
	if err != nil {
		return err
	}

	fallback, fallbackBaseline, err := runStickyRedisUnavailableFallback(
		ctx,
		client,
		sessions,
		sessionID,
		headers,
		tools,
		afterRestart,
		opts,
	)
	if err != nil {
		return err
	}

	recovered, err := waitForStickyRedisReuse(ctx, sessions, tools, opts)
	if err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"pre_restart_tools":           retained.Names,
			"post_restart_tools":          afterRestart.Names,
			"restart_stateless_baseline":  restartBaseline.Names,
			"unavailable_store_tools":     fallback.Names,
			"fallback_stateless_baseline": fallbackBaseline.Names,
			"post_recovery_tools":         recovered.Names,
			"state_survived_restart":      true,
			"fallback_request_ok":         true,
			"store_recovered":             true,
		})
	}
	return nil
}

func seedStickyRedisState(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (string, map[string]string, []fixtures.ChatTool, stickyToolSnapshot, error) {
	sessions, err := openStickySessionPair(ctx, client, opts)
	if err != nil {
		return "", nil, nil, stickyToolSnapshot{}, err
	}
	defer sessions.gateway.Close()
	defer sessions.backend.Close()

	sessionID := fmt.Sprintf("sticky-redis-restart-%d", time.Now().UnixNano())
	headers := stickyRequestHeaders(sessionID, true)
	tools := stickyContractTools(1, "")
	_, retained, err := runStickyTrustedTurns(ctx, sessions, sessionID, headers, tools)
	if err != nil {
		return "", nil, nil, stickyToolSnapshot{}, err
	}
	if len(retained.Tools) != stickyToolSelectionMaxTools {
		return "", nil, nil, stickyToolSnapshot{}, fmt.Errorf(
			"redis restart precondition retained %d tools, want %d",
			len(retained.Tools),
			stickyToolSelectionMaxTools,
		)
	}
	return sessionID, headers, tools, retained, nil
}

func verifyStickyRedisRestartReuse(
	ctx context.Context,
	sessions *stickySessionPair,
	sessionID string,
	headers map[string]string,
	tools []fixtures.ChatTool,
	retained stickyToolSnapshot,
) (stickyToolSnapshot, stickyToolSnapshot, error) {
	restartRequest := stickyNormalRequest("__STICKY_TOOL_SELECTION__ Calculate 29 times 31.", tools)
	afterRestart, err := runStickyTurn(ctx, sessions, sessionID, restartRequest, headers, "post-restart Redis-backed turn")
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	if comparisonErr := assertStickySnapshotsEqual(afterRestart, retained); comparisonErr != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, fmt.Errorf("redis-backed sticky state did not survive router restart: %w", comparisonErr)
	}

	restartBaselineID := sessionID + "-baseline"
	restartBaseline, err := runStickyTurn(
		ctx,
		sessions,
		restartBaselineID,
		restartRequest,
		stickyRequestHeaders(restartBaselineID, false),
		"post-restart stateless baseline",
	)
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	if err := assertStickySnapshotsDifferent(afterRestart, restartBaseline); err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, fmt.Errorf(
			"post-restart request did not demonstrate Redis-backed reuse: %w",
			err,
		)
	}
	return afterRestart, restartBaseline, nil
}

func runStickyRedisUnavailableFallback(
	ctx context.Context,
	client *kubernetes.Clientset,
	sessions *stickySessionPair,
	sessionID string,
	headers map[string]string,
	tools []fixtures.ChatTool,
	retained stickyToolSnapshot,
	opts pkgtestcases.TestCaseOptions,
) (fallback stickyToolSnapshot, baseline stickyToolSnapshot, err error) {
	scale, err := client.AppsV1().Deployments(stickyRedisNamespace).GetScale(
		ctx,
		stickyRedisDeployment,
		metav1.GetOptions{},
	)
	if err != nil {
		return fallback, baseline, fmt.Errorf("get Redis deployment scale: %w", err)
	}
	originalReplicas := scale.Spec.Replicas
	if originalReplicas < 1 {
		return fallback, baseline, fmt.Errorf("redis deployment has %d replicas before unavailable-store test", originalReplicas)
	}

	defer func() {
		restoreCtx, cancel := context.WithTimeout(context.Background(), stickyRedisScaleTimeout)
		defer cancel()
		restoreErr := setStickyRedisReplicas(restoreCtx, client, originalReplicas, opts.Verbose)
		if restoreErr != nil {
			err = errors.Join(err, fmt.Errorf("restore Redis deployment to %d replicas: %w", originalReplicas, restoreErr))
		}
	}()

	if scaleErr := setStickyRedisReplicas(ctx, client, 0, opts.Verbose); scaleErr != nil {
		return fallback, baseline, fmt.Errorf("make Redis unavailable: %w", scaleErr)
	}

	request := stickyNormalRequest("__STICKY_TOOL_SELECTION__ Search recent weather reports.", tools)
	fallback, err = runStickyTurn(ctx, sessions, sessionID, request, headers, "Redis-unavailable trusted turn")
	if err != nil {
		return fallback, baseline, err
	}
	baselineID := fmt.Sprintf("sticky-redis-fallback-baseline-%d", time.Now().UnixNano())
	baseline, err = runStickyTurn(
		ctx,
		sessions,
		baselineID,
		request,
		stickyRequestHeaders(baselineID, false),
		"Redis-unavailable stateless baseline",
	)
	if err != nil {
		return fallback, baseline, err
	}
	if err := assertStickySnapshotsEqual(fallback, baseline); err != nil {
		return fallback, baseline, fmt.Errorf("unavailable Redis did not fall back to stateless selection: %w", err)
	}
	if err := assertStickySnapshotsDifferent(fallback, retained); err != nil {
		return fallback, baseline, fmt.Errorf("unavailable Redis reused retained sticky state: %w", err)
	}
	return fallback, baseline, nil
}

func setStickyRedisReplicas(
	ctx context.Context,
	client *kubernetes.Clientset,
	replicas int32,
	verbose bool,
) error {
	scale, err := client.AppsV1().Deployments(stickyRedisNamespace).GetScale(
		ctx,
		stickyRedisDeployment,
		metav1.GetOptions{},
	)
	if err != nil {
		return fmt.Errorf("get scale: %w", err)
	}
	scale.Spec.Replicas = replicas
	if _, err := client.AppsV1().Deployments(stickyRedisNamespace).UpdateScale(
		ctx,
		stickyRedisDeployment,
		scale,
		metav1.UpdateOptions{},
	); err != nil {
		return fmt.Errorf("update scale to %d: %w", replicas, err)
	}

	if verbose {
		fmt.Printf("[Test] Waiting for Redis deployment to reach %d replicas\n", replicas)
	}
	return waitForStickyRedisReplicas(ctx, client, replicas)
}

func waitForStickyRedisReplicas(
	ctx context.Context,
	client *kubernetes.Clientset,
	replicas int32,
) error {
	timer := time.NewTimer(stickyRedisScaleTimeout)
	defer timer.Stop()
	ticker := time.NewTicker(stickyRedisScaleInterval)
	defer ticker.Stop()

	var lastState string
	for {
		ready, state := stickyRedisReadyState(ctx, client, replicas)
		lastState = state
		if ready {
			return nil
		}

		select {
		case <-ctx.Done():
			return errors.Join(ctx.Err(), fmt.Errorf("last Redis deployment state: %s", lastState))
		case <-timer.C:
			return fmt.Errorf(
				"redis deployment and endpoints did not reach %d replicas after %s: %s",
				replicas,
				stickyRedisScaleTimeout,
				lastState,
			)
		case <-ticker.C:
		}
	}
}

func stickyRedisReadyState(
	ctx context.Context,
	client *kubernetes.Clientset,
	replicas int32,
) (bool, string) {
	deployment, err := client.AppsV1().Deployments(stickyRedisNamespace).Get(
		ctx,
		stickyRedisDeployment,
		metav1.GetOptions{},
	)
	if err != nil {
		return false, err.Error()
	}
	state := fmt.Sprintf(
		"generation=%d observed=%d replicas=%d updated=%d ready=%d available=%d unavailable=%d",
		deployment.Generation,
		deployment.Status.ObservedGeneration,
		deployment.Status.Replicas,
		deployment.Status.UpdatedReplicas,
		deployment.Status.ReadyReplicas,
		deployment.Status.AvailableReplicas,
		deployment.Status.UnavailableReplicas,
	)
	if !stickyRedisDeploymentAtReplicas(deployment, replicas) {
		return false, state
	}

	readyEndpoints, err := countReadyEndpointAddresses(ctx, client, stickyRedisNamespace, stickyRedisDeployment)
	if err != nil {
		return false, fmt.Sprintf("%s ready_endpoints_error=%v", state, err)
	}
	state = fmt.Sprintf("%s ready_endpoints=%d", state, readyEndpoints)
	return stickyRedisEndpointsAtReplicas(readyEndpoints, replicas), state
}

func stickyRedisDeploymentAtReplicas(deployment *appsv1.Deployment, replicas int32) bool {
	if deployment.Status.ObservedGeneration < deployment.Generation {
		return false
	}
	if replicas == 0 {
		return deployment.Status.Replicas == 0 && deployment.Status.UpdatedReplicas == 0 &&
			deployment.Status.ReadyReplicas == 0 && deployment.Status.AvailableReplicas == 0
	}
	return deployment.Status.Replicas == replicas && deployment.Status.UpdatedReplicas >= replicas &&
		deployment.Status.ReadyReplicas >= replicas && deployment.Status.AvailableReplicas >= replicas &&
		deployment.Status.UnavailableReplicas == 0
}

func stickyRedisEndpointsAtReplicas(readyEndpoints int, replicas int32) bool {
	if replicas == 0 {
		return readyEndpoints == 0
	}
	return readyEndpoints >= int(replicas)
}

func waitForStickyRedisReuse(
	ctx context.Context,
	sessions *stickySessionPair,
	tools []fixtures.ChatTool,
	opts pkgtestcases.TestCaseOptions,
) (stickyToolSnapshot, error) {
	recoveryCtx, cancel := context.WithTimeout(ctx, stickyRedisReconnectTimeout)
	defer cancel()

	var lastErr error
	for {
		sessionID := fmt.Sprintf("sticky-redis-recovered-%d", time.Now().UnixNano())
		_, retained, err := runStickyTrustedTurns(
			recoveryCtx,
			sessions,
			sessionID,
			stickyRequestHeaders(sessionID, true),
			tools,
		)
		if err == nil {
			return retained, nil
		}
		lastErr = err
		if opts.Verbose {
			fmt.Printf("[Test] Redis-backed sticky reuse not ready: %v\n", err)
		}

		select {
		case <-recoveryCtx.Done():
			return stickyToolSnapshot{}, errors.Join(recoveryCtx.Err(), lastErr)
		case <-time.After(stickyRedisScaleInterval):
		}
	}
}
