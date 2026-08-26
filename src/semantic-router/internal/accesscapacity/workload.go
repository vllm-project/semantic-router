package accesscapacity

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
)

type eventResult struct {
	authentication time.Duration
	admission      time.Duration
	settlement     time.Duration
	disposition    quotaruntime.AdmissionDisposition
	produced       bool
	err            error
}

type workloadResult struct {
	results  []eventResult
	duration time.Duration
}

func runConcurrentWorkload(
	ctx context.Context,
	config Config,
	replicas []*replica,
	fixture Fixture,
) workloadResult {
	startedAt := time.Now()
	jobs := make(chan int)
	results := make(chan eventResult, config.Concurrency)
	var workers sync.WaitGroup
	for worker := range config.Concurrency {
		workers.Add(1)
		go func(workerID int) {
			defer workers.Done()
			current := replicas[workerID%len(replicas)]
			for index := range jobs {
				results <- executeEvent(
					ctx, current, fixture.Credentials[index], fixture.Targets[index],
					"capacity-admission-"+suffix(index),
				)
			}
		}(worker)
	}
	go func() {
		for index := range config.KeyCount {
			jobs <- index
		}
		close(jobs)
		workers.Wait()
		close(results)
	}()
	workload := workloadResult{results: make([]eventResult, 0, config.KeyCount)}
	for result := range results {
		workload.results = append(workload.results, result)
	}
	workload.duration = time.Since(startedAt)
	return workload
}

func summarizeWorkload(report *Report, config Config, workload workloadResult) {
	authentication := make([]time.Duration, 0, len(workload.results))
	admission := make([]time.Duration, 0, len(workload.results))
	settlement := make([]time.Duration, 0, len(workload.results))
	report.Admission.Attempted = int64(len(workload.results))
	for _, result := range workload.results {
		if result.authentication > 0 {
			authentication = append(authentication, result.authentication)
		}
		if result.admission > 0 {
			admission = append(admission, result.admission)
		}
		if result.settlement > 0 {
			settlement = append(settlement, result.settlement)
		}
		switch {
		case result.err != nil:
			report.Admission.Failed++
		case result.disposition == quotaruntime.AdmissionAllowed && result.produced:
			report.Admission.Allowed++
		case result.disposition == quotaruntime.AdmissionRateLimited:
			report.Admission.RateLimited++
		default:
			report.Admission.Failed++
		}
	}
	report.Admission.Authentication = latency(authentication)
	report.Admission.Admission = latency(admission)
	report.Admission.Settlement = latency(settlement)
	if workload.duration > 0 {
		report.Admission.EventsPerSecond = float64(config.KeyCount) / workload.duration.Seconds()
	}
}

func executeEvent(
	ctx context.Context,
	current *replica,
	credential,
	target,
	admissionID string,
) eventResult {
	result := eventResult{}
	authStarted := time.Now()
	authentication, err := current.runtime.Authenticate(ctx, accessruntime.AuthenticationRequest{Credential: credential})
	result.authentication = time.Since(authStarted)
	if err != nil {
		result.err = err
		return result
	}
	if !authentication.Result.Allowed() {
		result.disposition = authentication.Result.Disposition
		return result
	}
	requestDigest := digest("request", admissionID)
	admitStarted := time.Now()
	admission, err := current.runtime.Admit(ctx, accessruntime.AdmissionRequest{
		Session: authentication.Session,
		Target: accessruntime.Target{
			ResourceType: accesscontrol.GrantResourceEntrypoint,
			ResourceID:   accesscontrol.ResourceID(target),
			Permission:   accesscontrol.GrantPermissionInvoke,
		},
		AdmissionID: admissionID, RequestDigest: requestDigest, LeaseDuration: time.Minute,
	})
	result.admission = time.Since(admitStarted)
	result.disposition = admission.Result.Disposition
	if err != nil || !admission.Result.Allowed() {
		result.err = err
		return result
	}
	settleStarted := time.Now()
	dispatchID := admissionID + "-dispatch"
	dispatchDigest := digest("dispatch", admissionID)
	if _, err := current.runtime.JournalDispatch(ctx, accessruntime.DispatchJournalRequest{
		Admission: admission, DispatchID: dispatchID, Ordinal: 0, Digest: dispatchDigest,
	}); err != nil {
		result.err = err
		return result
	}
	evidence, err := current.runtime.ReadAttemptEvidence(ctx, accessruntime.AttemptEvidenceRequest{
		Admission: admission,
		Dispatches: []accessruntime.AttemptEvidenceDispatch{{
			DispatchID: dispatchID, Ordinal: 0, DispatchPlanDigest: dispatchDigest,
			ModelID: fixtureModelID, ModelRevision: 1,
		}},
	})
	if err != nil {
		result.err = err
		return result
	}
	event, _ := json.Marshal(usageEvent{EmittedAtUnixNano: time.Now().UnixNano()})
	if _, err := current.runtime.Settle(ctx, accessruntime.SettlementRequest{
		Admission: admission, AttemptEvidence: evidence, Aggregate: usageaccounting.Aggregate{},
		FinalizationDigest: digest("finalization", admissionID), Event: string(event),
		EventEvidenceState: "known",
	}); err != nil {
		result.err = err
		return result
	}
	result.settlement = time.Since(settleStarted)
	result.produced = true
	return result
}

func digest(parts ...string) string {
	hash := sha256.New()
	for _, part := range parts {
		_, _ = hash.Write([]byte(part))
		_, _ = hash.Write([]byte{0})
	}
	return hex.EncodeToString(hash.Sum(nil))
}
