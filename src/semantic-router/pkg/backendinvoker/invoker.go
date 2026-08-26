package backendinvoker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const (
	defaultRequestTimeout = 5 * time.Minute
	defaultStreamTimeout  = 5 * time.Minute
	maximumModelTimeout   = 24 * time.Hour
	maximumSafeAttempts   = 6
	maximumChainTimeout   = maximumDispatchCandidates * maximumSafeAttempts * maximumModelTimeout
)

var errEmptyTransportResponse = errors.New("transport returned an empty response")

var strippedHeaders = map[string]struct{}{
	"authorization": {}, "proxy-authorization": {}, "x-api-key": {}, "api-key": {},
	"x-goog-api-key": {}, "x-amz-security-token": {},
	"connection": {}, "content-length": {}, "cookie": {}, "host": {}, "keep-alive": {},
	"set-cookie": {}, "te": {}, "trailer": {}, "transfer-encoding": {}, "upgrade": {},
	"x-authz-user-id": {}, "x-authz-user-groups": {}, "x-vllm-sr-api-key-id": {},
	"x-vllm-sr-user-id": {}, "x-vllm-sr-team-id": {}, "x-vsr-destination-endpoint": {},
	"x-envoy-upstream-rq-timeout-ms": {}, "x-envoy-upstream-rq-per-try-timeout-ms": {},
	"x-envoy-max-retries": {}, "x-envoy-retry-on": {}, "x-envoy-hedge-on-per-try-timeout": {},
}

type Invoker struct {
	Transport   Transport
	Credentials CredentialResolver
	Codecs      *protocolcodec.Registry
	Journal     Journal
	Finalizer   ResponseFinalizer
	Now         func() time.Time
}

func (i *Invoker) Invoke(ctx context.Context, plan Plan) (Result, error) {
	return i.InvokeChain(ctx, PlanChain{Candidates: []Plan{plan}})
}

// InvokeChain executes one signed, snapshot-resolved candidate chain. Each
// physical attempt is capped by its Model timeout. One bounded deadline,
// derived from every candidate and safe attempt, is shared by the complete
// chain and never resets while retries or fallback advance.
func (i *Invoker) InvokeChain(ctx context.Context, chain PlanChain) (Result, error) {
	if err := validatePlanChain(chain); err != nil {
		return Result{}, err
	}
	if i == nil || i.Journal == nil {
		return Result{}, fmt.Errorf("dispatch journal is required")
	}
	formats := make([]llmprotocol.WireFormat, 0)
	for _, plan := range chain.Candidates {
		formats = append(formats, plan.SourceFormat)
		for _, backend := range plan.Backends {
			formats = append(formats, backend.WireFormat)
		}
	}
	if err := i.codecRegistry().Check(formats); err != nil {
		return Result{}, err
	}
	transport := i.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	now := time.Now
	if i.Now != nil {
		now = i.Now
	}
	deadline := now().Add(chainTimeout(chain))
	if parentDeadline, found := ctx.Deadline(); found && parentDeadline.Before(deadline) {
		deadline = parentDeadline
	}
	invokeCtx, cancel := context.WithDeadline(ctx, deadline)
	cancelOnReturn := true
	defer func() {
		if cancelOnReturn {
			cancel()
		}
	}()

	result := Result{Outcomes: make([]CandidateOutcome, 0, len(chain.Candidates))}
	var lastErr error
	var lastPlan *Plan
	for candidateIndex, plan := range chain.Candidates {
		if err := invokeCtx.Err(); err != nil {
			if lastPlan != nil && result.Attempt.ID != "" {
				_ = i.finalizeResponse(ctx, *lastPlan, result.Attempt, failedResponseTerminal(err))
			}
			return result, fmt.Errorf("backend fallback deadline exhausted: %w", err)
		}
		execution := i.invokeCandidate(ctx, invokeCtx, cancel, transport, now, deadline, plan)
		if execution.pinned {
			pinned := execution.plan
			lastPlan = &pinned
		}
		if execution.attempt.ID != "" {
			result.Attempt = execution.attempt
		}
		if execution.appendOutcome {
			result.Outcomes = append(result.Outcomes, execution.outcome)
		}
		lastErr = execution.err
		switch execution.state {
		case candidateExecutionAborted:
			return result, execution.err
		case candidateExecutionResponseStarted:
			selected := execution.plan
			result.Selected = &selected
			if execution.err != nil {
				return result, execution.err
			}
			cancelOnReturn = false
			result.Response = execution.response
			return result, nil
		case candidateExecutionRequestFailed:
			if finalizeErr := i.finalizeResponse(ctx, execution.plan, execution.attempt, failedResponseTerminal(execution.err)); finalizeErr != nil {
				return result, errors.Join(execution.err, finalizeErr)
			}
			return result, execution.err
		case candidateExecutionFallbackEligible:
			if fallbackEnabled(chain.Fallback, execution.outcome.FallbackTrigger) &&
				candidateIndex+1 < len(chain.Candidates) && invokeCtx.Err() == nil {
				continue
			}
			fallthrough
		case candidateExecutionTerminalFailure:
			if finalizeErr := i.finalizeResponse(ctx, execution.plan, execution.attempt, failedResponseTerminal(execution.err)); finalizeErr != nil {
				return result, errors.Join(execution.err, finalizeErr)
			}
			return result, fmt.Errorf("backend attempt failed: %w", execution.err)
		}
	}
	if lastPlan != nil && result.Attempt.ID != "" {
		_ = i.finalizeResponse(ctx, *lastPlan, result.Attempt, failedResponseTerminal(lastErr))
	}
	return result, fmt.Errorf("backend invocation exhausted: %w", lastErr)
}

type candidateExecutionState uint8

const (
	candidateExecutionAborted candidateExecutionState = iota
	candidateExecutionResponseStarted
	candidateExecutionRequestFailed
	candidateExecutionFallbackEligible
	candidateExecutionTerminalFailure
)

type candidateExecution struct {
	state         candidateExecutionState
	plan          Plan
	pinned        bool
	outcome       CandidateOutcome
	attempt       AttemptResult
	response      *http.Response
	err           error
	appendOutcome bool
}

type candidateAttemptState uint8

const (
	candidateAttemptAborted candidateAttemptState = iota
	candidateAttemptResponseStarted
	candidateAttemptRequestFailed
	candidateAttemptTransportFailed
)

type candidateAttemptExecution struct {
	state    candidateAttemptState
	attempt  AttemptResult
	response *http.Response
	err      error
	recorded bool
}

func (i *Invoker) invokeCandidate(
	requestCtx context.Context,
	invokeCtx context.Context,
	cancel context.CancelFunc,
	transport Transport,
	now func() time.Time,
	deadline time.Time,
	sourcePlan Plan,
) candidateExecution {
	plan, err := i.pinProviderCredentials(invokeCtx, sourcePlan)
	if err != nil {
		return candidateExecution{state: candidateExecutionAborted, err: err}
	}
	execution := candidateExecution{state: candidateExecutionAborted, plan: plan, pinned: true, outcome: candidateOutcome(plan)}
	if err := i.Journal.BeginDispatch(invokeCtx, plan, deadline); err != nil {
		execution.err = fmt.Errorf("journal dispatch intent: %w", err)
		return execution
	}
	for number := 1; number <= plan.Execution.MaxRetries+1; number++ {
		backend := selectBackend(plan.Backends, plan.DispatchID, number)
		attempt := Attempt{ID: plan.DispatchID + ":" + strconv.Itoa(number), Number: number, BackendID: backend.ID, StartedAt: now().UTC()}
		attemptCtx, attemptCancel := context.WithDeadline(
			invokeCtx,
			localAttemptDeadline(now(), deadline, plan),
		)
		attemptExecution := i.invokeCandidateAttempt(
			requestCtx,
			attemptCtx,
			cancelTogether(attemptCancel, cancel),
			transport,
			now,
			plan,
			backend,
			attempt,
		)
		if attemptExecution.response == nil {
			attemptCancel()
		}
		execution.attempt, execution.err = attemptExecution.attempt, attemptExecution.err
		if attemptExecution.recorded {
			appendAttempt(&execution.outcome, attemptExecution.attempt)
		}
		switch attemptExecution.state {
		case candidateAttemptAborted:
			execution.appendOutcome = attemptExecution.recorded
			return execution
		case candidateAttemptResponseStarted:
			execution.state, execution.response, execution.appendOutcome = candidateExecutionResponseStarted, attemptExecution.response, true
			return execution
		case candidateAttemptRequestFailed:
			execution.state, execution.appendOutcome = candidateExecutionRequestFailed, true
			return execution
		case candidateAttemptTransportFailed:
			if attemptExecution.attempt.State == AttemptKnownZero &&
				number <= plan.Execution.MaxRetries &&
				fallbackEnabled(FallbackPolicy{On: plan.Execution.RetryOn}, attemptExecution.attempt.FallbackTrigger) {
				continue
			}
		}
		break
	}
	execution.appendOutcome = true
	if execution.outcome.State == AttemptKnownZero {
		execution.state = candidateExecutionFallbackEligible
	} else {
		execution.state = candidateExecutionTerminalFailure
	}
	return execution
}

func (i *Invoker) invokeCandidateAttempt(
	requestCtx context.Context,
	invokeCtx context.Context,
	cancel context.CancelFunc,
	transport Transport,
	now func() time.Time,
	plan Plan,
	backend Backend,
	attempt Attempt,
) candidateAttemptExecution {
	if err := i.Journal.BeginAttempt(invokeCtx, plan, attempt); err != nil {
		return candidateAttemptExecution{state: candidateAttemptAborted, err: fmt.Errorf("journal attempt %d: %w", attempt.Number, err)}
	}
	req, tracker, requestErr := i.request(invokeCtx, plan, backend)
	if requestErr != nil {
		result := AttemptResult{Attempt: attempt, State: AttemptKnownZero, CompletedAt: now().UTC(), ErrorCode: "request_build_failed"}
		_ = i.Journal.FinishAttempt(context.WithoutCancel(requestCtx), plan, result)
		return candidateAttemptExecution{state: candidateAttemptRequestFailed, attempt: result, err: requestErr, recorded: true}
	}
	rawResponse, roundTripErr := transport.RoundTrip(req)
	closeResponseOnReturn := rawResponse != nil && rawResponse.Body != nil
	defer func() {
		if closeResponseOnReturn {
			_ = rawResponse.Body.Close()
		}
	}()
	if rawResponse == nil && roundTripErr == nil {
		roundTripErr = errEmptyTransportResponse
	}
	result := classifyAttempt(attempt, rawResponse, roundTripErr, tracker, now().UTC())
	if err := i.Journal.FinishAttempt(context.WithoutCancel(requestCtx), plan, result); err != nil {
		return candidateAttemptExecution{state: candidateAttemptAborted, attempt: result, err: fmt.Errorf("journal attempt result: %w", err), recorded: true}
	}
	if roundTripErr != nil {
		return candidateAttemptExecution{state: candidateAttemptTransportFailed, attempt: result, err: roundTripErr, recorded: true}
	}
	closeResponseOnReturn = false
	return i.transformCandidateResponse(invokeCtx, cancel, plan, backend, result, rawResponse, tracker.sensitiveValues)
}

func (i *Invoker) transformCandidateResponse(
	invokeCtx context.Context,
	cancel context.CancelFunc,
	plan Plan,
	backend Backend,
	attempt AttemptResult,
	rawResponse *http.Response,
	sensitiveValues []string,
) candidateAttemptExecution {
	response, err := i.transformResponse(invokeCtx, plan, backend, attempt, rawResponse, sensitiveValues)
	closeResponseOnReturn := response != nil && response.Body != nil
	defer func() {
		if closeResponseOnReturn {
			_ = response.Body.Close()
		}
	}()
	result := candidateAttemptExecution{state: candidateAttemptResponseStarted, attempt: attempt, err: err, recorded: true}
	if err != nil {
		result.err = fmt.Errorf("translate backend response from wire format %q: %w", backend.WireFormat, err)
		return result
	}
	if response == nil || response.Body == nil {
		result.err = fmt.Errorf("backend returned an empty response")
		return result
	}
	response.Body = newCancelOnCloseBody(response.Body, cancel)
	closeResponseOnReturn = false
	result.response = response
	return result
}

func failedResponseTerminal(cause error) ResponseTerminal {
	category, code, message := llmprotocol.ErrorUpstreamUnavailable, "upstream_unavailable", "the selected model is temporarily unavailable"
	if errors.Is(cause, context.DeadlineExceeded) {
		category, code, message = llmprotocol.ErrorUpstreamTimeout, "upstream_timeout", "the selected model timed out"
	} else if errors.Is(cause, context.Canceled) {
		code, message = "upstream_canceled", "the selected model request was canceled"
	}
	return ResponseTerminal{
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}, StopReason: llmprotocol.StopError,
		Error: llmprotocol.NewError(category, code, message, cause),
	}
}

func chainTimeout(chain PlanChain) time.Duration {
	total := time.Duration(0)
	for _, plan := range chain.Candidates {
		attempts := plan.Execution.MaxRetries + 1
		if attempts < 1 {
			attempts = 1
		}
		timeout := localAttemptTimeout(plan)
		if timeout > maximumChainTimeout/time.Duration(attempts) {
			return maximumChainTimeout
		}
		candidateBudget := timeout * time.Duration(attempts)
		if candidateBudget > maximumChainTimeout-total {
			return maximumChainTimeout
		}
		total += candidateBudget
	}
	if total > 0 {
		return total
	}
	return defaultRequestTimeout
}

func localAttemptTimeout(plan Plan) time.Duration {
	timeout := plan.Execution.RequestTimeout
	if plan.Streaming {
		timeout = plan.Execution.StreamTimeout
		if timeout <= 0 {
			return defaultStreamTimeout
		}
		return timeout
	}
	if timeout <= 0 {
		return defaultRequestTimeout
	}
	return timeout
}

func localAttemptDeadline(now, chainDeadline time.Time, plan Plan) time.Time {
	deadline := now.Add(localAttemptTimeout(plan))
	if chainDeadline.Before(deadline) {
		return chainDeadline
	}
	return deadline
}

func cancelTogether(first, second context.CancelFunc) context.CancelFunc {
	return func() {
		first()
		second()
	}
}

func candidateOutcome(plan Plan) CandidateOutcome {
	return CandidateOutcome{
		DispatchID: plan.DispatchID, DispatchType: plan.DispatchType,
		Ordinal: plan.Ordinal, DispatchPlanDigest: plan.DispatchPlanDigest,
		ModelID: plan.ModelID, ModelRevision: plan.ModelRevision, Priority: plan.Priority,
		Attempts: make([]AttemptResult, 0, plan.Execution.MaxRetries+1),
	}
}

func appendAttempt(outcome *CandidateOutcome, result AttemptResult) {
	outcome.Attempts = append(outcome.Attempts, result)
	outcome.State = result.State
	outcome.FallbackTrigger = result.FallbackTrigger
}

func (i *Invoker) pinProviderCredentials(ctx context.Context, plan Plan) (Plan, error) {
	pinned := plan
	publication := CredentialPublication{
		NamespaceID: plan.NamespaceID, QuotaPartition: plan.QuotaPartition, PublicationID: plan.PublicationID,
	}
	pinned.Backends = append([]Backend(nil), plan.Backends...)
	for index := range pinned.Backends {
		backend := &pinned.Backends[index]
		backend.Connection = cloneConnection(backend.Connection)
		if backend.ProviderCredentialID == "" {
			if backend.ProviderCredentialVersion != "" {
				return Plan{}, fmt.Errorf("backend without provider credential contains a credential version")
			}
			continue
		}
		if i.Credentials == nil {
			return Plan{}, fmt.Errorf("provider credential resolver is required")
		}
		if err := publication.Validate(); err != nil {
			return Plan{}, err
		}
		version, err := i.Credentials.Pin(
			ctx, publication, backend.ProviderCredentialID, backend.ProviderID, backend.Origin,
		)
		if err != nil {
			return Plan{}, fmt.Errorf("pin provider credential: %w", err)
		}
		if strings.TrimSpace(version) == "" {
			return Plan{}, fmt.Errorf("provider credential resolver returned an empty version")
		}
		backend.ProviderCredentialVersion = version
	}
	return pinned, nil
}

type attemptTracker struct {
	wroteRequest    bool
	gotResponseByte bool
	sensitiveValues []string
}

func (i *Invoker) request(
	ctx context.Context,
	plan Plan,
	backend Backend,
) (*http.Request, *attemptTracker, error) {
	publication := CredentialPublication{
		NamespaceID: plan.NamespaceID, QuotaPartition: plan.QuotaPartition, PublicationID: plan.PublicationID,
	}
	engine, err := protocolcodec.NewEngine(i.codecRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		return nil, nil, err
	}
	translated, err := engine.TranslateRequest(plan.SourceFormat, backend.WireFormat, plan.Body, func(request *llmprotocol.Request) error {
		request.Model = backend.ProviderModelID
		request.Stream = plan.Streaming
		return nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("translate request from %q to %q: %w", plan.SourceFormat, backend.WireFormat, err)
	}
	base, err := url.Parse(backend.Origin)
	if err != nil {
		return nil, nil, fmt.Errorf("parse backend origin: %w", err)
	}
	path, err := url.Parse(backend.Connection.Path)
	if err != nil || !strings.HasPrefix(path.Path, "/") {
		return nil, nil, fmt.Errorf("invalid backend wire path")
	}
	base.Path = strings.TrimRight(base.Path, "/") + path.Path
	base.RawQuery = plan.Query
	request, err := http.NewRequestWithContext(ctx, plan.Method, base.String(), bytes.NewReader(translated.Body))
	if err != nil {
		return nil, nil, fmt.Errorf("build backend request: %w", err)
	}
	request.Header = sanitizedHeaders(plan.Headers)
	applyConnectionHeaders(request.Header, backend.Connection.Headers)
	request.Header.Set("Content-Type", "application/json")
	if plan.Streaming {
		request.Header.Set("Accept", "text/event-stream")
	}
	request.Header.Set("Content-Length", strconv.Itoa(len(translated.Body)))
	request.ContentLength = int64(len(translated.Body))
	request.GetBody = nil
	var sensitiveValues []string
	if backend.ProviderCredentialID != "" {
		if err := publication.Validate(); err != nil {
			return nil, nil, err
		}
		credential, err := i.Credentials.ResolvePinned(
			ctx, publication, backend.ProviderCredentialID, backend.ProviderCredentialVersion,
			backend.ProviderID, backend.Origin,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("resolve provider credential: %w", err)
		}
		if credential.Version != backend.ProviderCredentialVersion {
			return nil, nil, fmt.Errorf("provider credential resolver returned a different version")
		}
		if err := applyCredential(request.Header, credential); err != nil {
			return nil, nil, err
		}
		sensitiveValues = credentialSensitiveValues(credential)
	}
	tracker := &attemptTracker{sensitiveValues: sensitiveValues}
	trace := &httptrace.ClientTrace{
		WroteRequest:         func(httptrace.WroteRequestInfo) { tracker.wroteRequest = true },
		GotFirstResponseByte: func() { tracker.gotResponseByte = true },
	}
	request = request.WithContext(httptrace.WithClientTrace(request.Context(), trace))
	return request, tracker, nil
}

func cloneConnection(source Connection) Connection {
	return Connection{Path: source.Path, Headers: source.Headers.Clone()}
}

func applyCredential(headers http.Header, credential Credential) error {
	header := strings.TrimSpace(credential.Header)
	if header == "" || strings.TrimSpace(credential.Secret) == "" {
		return fmt.Errorf("provider credential header and secret are required")
	}
	if isStrippedHeader(header) && !strings.EqualFold(header, "Authorization") && !strings.EqualFold(header, "X-API-Key") && !strings.EqualFold(header, "API-Key") {
		return fmt.Errorf("provider credential uses forbidden header")
	}
	headers.Set(header, credential.Prefix+credential.Secret)
	for key, values := range credential.Extra {
		if isStrippedHeader(key) {
			return fmt.Errorf("provider credential extra header %q is forbidden", key)
		}
		headers.Del(key)
		for _, value := range values {
			headers.Add(key, value)
		}
	}
	return nil
}

func credentialSensitiveValues(credential Credential) []string {
	values := make([]string, 0, 2+len(credential.Extra))
	secret := strings.TrimSpace(credential.Secret)
	if secret != "" {
		values = append(values, secret)
		if applied := credential.Prefix + credential.Secret; applied != secret {
			values = append(values, applied)
		}
	}
	for _, headers := range credential.Extra {
		for _, value := range headers {
			if value = strings.TrimSpace(value); value != "" {
				values = append(values, value)
			}
		}
	}
	return values
}

func sanitizedHeaders(source http.Header) http.Header {
	result := make(http.Header, len(source))
	for key, values := range source {
		if isStrippedHeader(key) {
			continue
		}
		for _, value := range values {
			result.Add(key, value)
		}
	}
	return result
}

func isStrippedHeader(key string) bool {
	canonical := strings.ToLower(strings.TrimSpace(key))
	if _, forbidden := strippedHeaders[canonical]; forbidden {
		return true
	}
	// Router-owned context is never provider input. Prefix matching keeps new
	// internal headers fail-closed without requiring every data-plane feature to
	// update the backend egress deny list in lockstep.
	return strings.HasPrefix(canonical, "x-vsr-") ||
		strings.HasPrefix(canonical, "x-vllm-sr-") ||
		strings.HasPrefix(canonical, "x-authz-") ||
		strings.HasPrefix(canonical, "x-user-")
}

func classifyAttempt(attempt Attempt, response *http.Response, err error, tracker *attemptTracker, completedAt time.Time) AttemptResult {
	result := AttemptResult{Attempt: attempt, CompletedAt: completedAt}
	if response != nil {
		result.StatusCode = response.StatusCode
	}
	if err == nil {
		result.State = AttemptResponseStarted
		return result
	}
	if errors.Is(err, errEmptyTransportResponse) {
		result.State = AttemptUnknown
		result.ErrorCode = "transport_empty_response"
		return result
	}
	result.ErrorCode = classifyTransportFailure(err)
	trigger, provenKnownZero := knownZeroTrigger(err)
	if provenKnownZero && response == nil && tracker != nil && !tracker.wroteRequest && !tracker.gotResponseByte {
		result.State = AttemptKnownZero
		result.FallbackTrigger = trigger
		return result
	}
	result.State = AttemptUnknown
	return result
}

func classifyTransportFailure(err error) string {
	if errors.Is(err, context.Canceled) {
		return "transport_canceled"
	}
	var networkError net.Error
	if errors.Is(err, context.DeadlineExceeded) || errors.As(err, &networkError) && networkError.Timeout() {
		return "transport_timeout"
	}
	return "transport_unavailable"
}

func validatePlanChain(chain PlanChain) error {
	if err := validateFallbackPolicy(chain.Fallback); err != nil {
		return err
	}
	if len(chain.Candidates) == 0 || len(chain.Candidates) > maximumDispatchCandidates {
		return fmt.Errorf("plan chain must contain between 1 and %d candidates", maximumDispatchCandidates)
	}
	claims := make([]DispatchCandidate, 0, len(chain.Candidates))
	first := chain.Candidates[0]
	for index, plan := range chain.Candidates {
		if err := validatePlan(plan); err != nil {
			return fmt.Errorf("candidate plan %d: %w", index, err)
		}
		if index > 0 && !sameRequestPlan(first, plan) {
			return fmt.Errorf("candidate plan %d does not share one immutable request and routing generation", index)
		}
		claims = append(claims, candidateFromPlan(plan))
	}
	return validateCandidateChain(claims, chain.Fallback)
}

func sameRequestPlan(left, right Plan) bool {
	return left.NamespaceID == right.NamespaceID && left.QuotaPartition == right.QuotaPartition &&
		left.PublicationID == right.PublicationID && left.RuntimeEpoch == right.RuntimeEpoch &&
		left.RoutingRevision == right.RoutingRevision && left.RoutingDigest == right.RoutingDigest &&
		left.AdmissionID == right.AdmissionID && left.AdmissionDigest == right.AdmissionDigest &&
		left.RequestID == right.RequestID && left.Method == right.Method && left.Path == right.Path &&
		left.Query == right.Query && left.RequestDigest == right.RequestDigest &&
		left.Streaming == right.Streaming && left.SourceFormat == right.SourceFormat && bytes.Equal(left.Body, right.Body)
}

func validatePlan(plan Plan) error {
	if plan.NamespaceID == "" || plan.QuotaPartition == "" || plan.PublicationID == "" ||
		plan.RuntimeEpoch == 0 || plan.RoutingRevision <= 0 || !validSHA256Hex(plan.RoutingDigest) ||
		plan.AdmissionID == "" || !validSHA256Hex(plan.AdmissionDigest) ||
		!validBoundedIdentity(plan.RequestID) ||
		plan.DispatchID == "" || plan.DispatchType == "" || plan.Ordinal < 0 ||
		!validSHA256Hex(plan.DispatchPlanDigest) || plan.ModelID == "" || plan.ModelRevision <= 0 ||
		!validRequestDigest(plan.RequestDigest) {
		return fmt.Errorf("complete immutable dispatch, routing, model, and request identities are required")
	}
	if plan.Method == "" || !strings.HasPrefix(plan.Path, "/") || plan.SourceFormat == "" || len(plan.Backends) == 0 {
		return fmt.Errorf("method, absolute path, source wire format, and at least one backend are required")
	}
	if plan.Priority < 0 || plan.Priority > 31 ||
		plan.RequestDigest != RequestDigest(plan.Method, plan.Path, plan.Query, plan.Body) {
		return fmt.Errorf("priority and exact request digest are required")
	}
	if plan.Execution.MaxRetries < 0 || plan.Execution.MaxRetries > 5 {
		return fmt.Errorf("max retries must be between 0 and 5")
	}
	if !validModelTimeout(plan.Execution.RequestTimeout) || !validModelTimeout(plan.Execution.StreamTimeout) {
		return fmt.Errorf("request and stream timeouts must be zero or between %s and %s", time.Second, maximumModelTimeout)
	}
	if plan.Execution.MaxRetries == 0 && len(plan.Execution.RetryOn) != 0 {
		return fmt.Errorf("retry triggers require at least one retry")
	}
	if plan.Execution.MaxRetries > 0 && len(plan.Execution.RetryOn) == 0 {
		return fmt.Errorf("retry triggers are required when retries are enabled")
	}
	if err := validateFallbackPolicy(FallbackPolicy{On: plan.Execution.RetryOn}); err != nil {
		return fmt.Errorf("retry policy: %w", err)
	}
	var totalWeight uint64
	for _, backend := range plan.Backends {
		if backend.ID == "" || backend.Origin == "" || backend.ProviderID == "" || backend.WireFormat == "" || backend.ProviderModelID == "" || backend.Weight == 0 {
			return fmt.Errorf("every backend requires id, origin, provider attribution, wire format, provider model, and positive weight")
		}
		if err := validateRuntimeConnection(backend.Connection); err != nil {
			return fmt.Errorf("backend %q connection: %w", backend.ID, err)
		}
		if backend.ProviderCredentialID == "" && backend.ProviderCredentialVersion != "" {
			return fmt.Errorf("provider credential version requires a provider credential")
		}
		if ^uint64(0)-totalWeight < backend.Weight {
			return fmt.Errorf("combined backend weight exceeds the runtime range")
		}
		totalWeight += backend.Weight
	}
	return nil
}

func validModelTimeout(timeout time.Duration) bool {
	return timeout == 0 || timeout >= time.Second && timeout <= maximumModelTimeout
}

type cancelOnCloseBody struct {
	body   io.ReadCloser
	cancel context.CancelFunc
	once   sync.Once
}

func newCancelOnCloseBody(body io.ReadCloser, cancel context.CancelFunc) io.ReadCloser {
	return &cancelOnCloseBody{body: body, cancel: cancel}
}

func (body *cancelOnCloseBody) Read(target []byte) (int, error) {
	read, err := body.body.Read(target)
	if err != nil {
		body.once.Do(body.cancel)
	}
	return read, err
}

func (body *cancelOnCloseBody) Close() error {
	err := body.body.Close()
	body.once.Do(body.cancel)
	return err
}

func selectBackend(backends []Backend, dispatchID string, attempt int) Backend {
	var total uint64
	for _, backend := range backends {
		total += backend.Weight
	}
	digest := sha256.Sum256([]byte(dispatchID + ":" + strconv.Itoa(attempt)))
	point := binary.BigEndian.Uint64(digest[:8]) % total
	for _, backend := range backends {
		if point < backend.Weight {
			return backend
		}
		point -= backend.Weight
	}
	return backends[len(backends)-1]
}
