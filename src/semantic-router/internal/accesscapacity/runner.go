package accesscapacity

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
)

func Run(
	parent context.Context,
	redisOptions *redis.Options,
	config Config,
) (report Report, returnErr error) {
	startedAt := time.Now().UTC()
	transport := "redis"
	if redisOptions != nil && redisOptions.TLSConfig != nil {
		transport = "rediss"
	}
	prefix := config.KeyPrefix
	if prefix == "" {
		prefix = fmt.Sprintf("access-capacity:%d", startedAt.UnixNano())
	}
	report = NewReport(config, startedAt, transport)
	defer func() { report.Complete(config) }()
	if redisOptions == nil {
		return fail(&report, fmt.Errorf("Redis options are required"))
	}
	if err := config.Validate(); err != nil {
		return fail(&report, err)
	}
	ctx, cancel := context.WithTimeout(parent, config.OperationTimeout)
	defer cancel()
	control := redis.NewClient(redisOptions)
	defer func() { _ = control.Close() }()
	if err := control.Ping(ctx).Err(); err != nil {
		return fail(&report, fmt.Errorf("connect to isolated Redis/Valkey: %w", err))
	}
	if existing, err := readMemorySnapshot(ctx, control, prefix); err != nil {
		return fail(&report, err)
	} else if existing.Keys != 0 {
		return fail(&report, fmt.Errorf("capacity key prefix already contains %d keys", existing.Keys))
	}
	if !config.KeepData {
		defer func() {
			cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cleanupCancel()
			if err := deletePrefix(cleanupContext, control, prefix); err != nil {
				appendReportError(&report, err)
				returnErr = errors.Join(returnErr, err)
			}
		}()
	}
	version, mode, err := readRedisEnvironment(ctx, control)
	if err != nil {
		return fail(&report, err)
	}
	report.Environment.RedisVersion = version
	report.Environment.RedisMode = mode

	fixture, projectionMemory, err := runProjection(ctx, control, prefix, config, &report)
	if err != nil {
		return fail(&report, err)
	}
	if err := runRuntimeGate(
		ctx, control, redisOptions, prefix, config, fixture, projectionMemory, &report,
	); err != nil {
		return fail(&report, err)
	}
	return report, nil
}

func fail(report *Report, err error) (Report, error) {
	appendReportError(report, err)
	return *report, err
}

func runProjection(
	ctx context.Context,
	control *redis.Client,
	prefix string,
	config Config,
	report *Report,
) (Fixture, memorySnapshot, error) {
	beforeOps, err := readCommandSnapshot(ctx, control)
	if err != nil {
		return Fixture{}, memorySnapshot{}, err
	}
	fixtureStarted := time.Now()
	fixture, err := BuildFixture(config, fixtureStarted)
	if err != nil {
		return Fixture{}, memorySnapshot{}, err
	}
	report.Projection.FixtureBuildMS = milliseconds(time.Since(fixtureStarted))
	compileStarted := time.Now()
	publication, err := accesspublisher.Compile(fixture.Desired)
	if err != nil {
		return Fixture{}, memorySnapshot{}, fmt.Errorf("compile access publication: %w", err)
	}
	report.Projection.CompileMS = milliseconds(time.Since(compileStarted))
	publishStarted := time.Now()
	if err := publishCapacityProjection(ctx, control, prefix, publication); err != nil {
		return Fixture{}, memorySnapshot{}, err
	}
	report.Projection.PublishMS = milliseconds(time.Since(publishStarted))
	report.Projection.TotalMS = report.Projection.CompileMS + report.Projection.PublishMS
	report.Projection.KeysPerSecond = float64(config.KeyCount) / (report.Projection.TotalMS / 1000)
	report.Projection.VisibilitySets = 2
	afterOps, err := readCommandSnapshot(ctx, control)
	if err != nil {
		return Fixture{}, memorySnapshot{}, err
	}
	report.Projection.RedisOps = commandDelta(beforeOps, afterOps)
	report.Projection.RedisOpsPerKey = float64(report.Projection.RedisOps.Total) /
		float64(config.KeyCount)
	memory, err := readMemorySnapshot(ctx, control, prefix)
	if err != nil {
		return Fixture{}, memorySnapshot{}, err
	}
	report.Projection.RedisKeyCount = memory.Keys
	report.Projection.MemoryBytes = memory.Bytes
	report.Projection.MemoryBytesPerKey = float64(memory.Bytes) / float64(config.KeyCount)
	return fixture, memory, nil
}

func runRuntimeGate(
	ctx context.Context,
	control *redis.Client,
	redisOptions *redis.Options,
	prefix string,
	config Config,
	fixture Fixture,
	projectionMemory memorySnapshot,
	report *Report,
) error {
	replicas, err := newReplicas(redisOptions, config.Replicas, fixture, prefix)
	if err != nil {
		return err
	}
	defer closeReplicas(replicas)
	report.Projection.IsolationSamples, report.Projection.IsolationViolations = verifyIsolation(
		ctx, replicas, fixture,
	)
	observerContext, observerCancel := context.WithCancel(ctx)
	defer observerCancel()
	expectedUsage := int64(config.KeyCount + config.RequestLimit - 1)
	observer, err := startUsageObserver(observerContext, control, prefix, expectedUsage)
	if err != nil {
		return fmt.Errorf("start usage observer: %w", err)
	}
	beforeOps, err := readCommandSnapshot(ctx, control)
	if err != nil {
		return err
	}
	workload := runConcurrentWorkload(ctx, config, replicas, fixture)
	summarizeWorkload(report, config, workload)
	producedByFailover, failoverErr := runReplicaFailover(
		ctx, redisOptions, config, replicas, fixture, prefix, &report.Failover,
	)
	if failoverErr != nil {
		appendReportError(report, failoverErr)
	}
	report.Usage.Produced = report.Admission.Allowed + producedByFailover
	observation := awaitUsageObservation(observerContext, observerCancel, observer, config.UsageDrainTimeout)
	if observation.Err != nil {
		appendReportError(report, fmt.Errorf("observe usage stream: %w", observation.Err))
	}
	afterOps, metricsErr := readCommandSnapshot(ctx, control)
	if metricsErr != nil {
		appendReportError(report, metricsErr)
	} else {
		report.Admission.RedisOps = commandDelta(beforeOps, afterOps)
		if report.Usage.Produced > 0 {
			report.Admission.RedisOpsPerEvent = float64(report.Admission.RedisOps.Total) /
				float64(report.Usage.Produced)
		}
	}
	recordUsageState(ctx, control, prefix, observation, report)
	memoryErr := recordEventMemory(ctx, control, prefix, projectionMemory, report)
	if memoryErr != nil {
		appendReportError(report, memoryErr)
	}
	if failoverErr != nil || observation.Err != nil || metricsErr != nil || memoryErr != nil ||
		report.Admission.Failed != 0 || len(report.Errors) != 0 {
		return errors.Join(
			errors.New("capacity gate invariant failed"),
			failoverErr,
			observation.Err,
		)
	}
	return nil
}

func recordEventMemory(
	ctx context.Context,
	control *redis.Client,
	prefix string,
	projectionMemory memorySnapshot,
	report *Report,
) error {
	workloadMemory, err := readMemorySnapshot(ctx, control, prefix)
	if err != nil {
		return err
	}
	report.Admission.MemoryDeltaBytes = max(workloadMemory.Bytes-projectionMemory.Bytes, 0)
	if report.Usage.Produced > 0 {
		report.Admission.MemoryBytesPerEvent = float64(report.Admission.MemoryDeltaBytes) /
			float64(report.Usage.Produced)
	}
	return nil
}

func recordUsageState(
	ctx context.Context,
	control *redis.Client,
	prefix string,
	observation usageObservation,
	report *Report,
) {
	report.Usage.Observed = observation.Observed
	report.Usage.Acknowledged = observation.Acknowledged
	report.Usage.ObservationLag = latency(observation.Lag)
	retained, pending, groupLag, err := usageGroupState(ctx, control, prefix)
	if err != nil {
		appendReportError(report, err)
		return
	}
	report.Usage.RetainedEntries = retained
	report.Usage.PendingEntries = pending
	report.Usage.GroupLag = groupLag
}

func awaitUsageObservation(
	ctx context.Context,
	cancel context.CancelFunc,
	observer <-chan usageObservation,
	timeout time.Duration,
) usageObservation {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case result := <-observer:
		return result
	case <-timer.C:
		cancel()
		select {
		case result := <-observer:
			if result.Err == nil {
				result.Err = fmt.Errorf("usage observer timed out")
			}
			return result
		case <-time.After(time.Second):
			return usageObservation{Err: fmt.Errorf("usage observer did not stop after timeout")}
		}
	case <-ctx.Done():
		cancel()
		return usageObservation{Err: ctx.Err()}
	}
}
