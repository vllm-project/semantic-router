package evaluationplane

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const workerSandboxScript = "cli/evaluation/sandbox_worker.py"

const (
	maxWorkerEventLineBytes = 16 * 1024
	maxWorkerMessageBytes   = 512
	maxWorkerRecordCount    = 100_000_000
	workerProcessWaitDelay  = 5 * time.Second
)

const (
	routerEvaluationCredentialEnv = "VLLM_SR_EVALUATION_ROUTER_API_KEY" //nolint:gosec // Environment variable name, not a credential.
	routerManagementCredentialEnv = "VLLM_SR_DASHBOARD_RECIPE_TOKEN"    //nolint:gosec // Environment variable name, not a credential.
)

// CredentialProvider is the narrow recipe-store seam required by evaluation
// execution. Missing credentials must be reported as os.ErrNotExist.
type CredentialProvider interface {
	ManagementCredential() (string, error)
}

type ProcessSpec struct {
	ManifestPath        string
	StorePath           string
	SuiteStorePath      string
	executionContracts  *executionContractRegistry
	controlledPair      *controlledPairRunContext
	evidencePublication func(func() error) error
}

type ProcessResult struct {
	ExecutionTranscript *brokerExecutionTranscript
	publishEvidence     func() error
	discardEvidence     func()
}

func (result ProcessResult) publishStagedEvidence() error {
	if result.publishEvidence == nil {
		return nil
	}
	return result.publishEvidence()
}

func (result ProcessResult) discardStagedEvidence() {
	if result.discardEvidence != nil {
		result.discardEvidence()
	}
}

// Process is injectable for tests, but its input deliberately contains no
// command, argument, environment, or executable override.
type Process interface {
	Run(context.Context, ProcessSpec, func(WorkerEvent) error) (ProcessResult, error)
}

type CommandProcess struct {
	pythonPath       string
	workerScriptPath string
	routerAPIKeyEnv  string
	envoyAPIKeyEnv   string
	cpuSeconds       int
	diagnosticSink   io.Writer
	publishEvidence  func(*workerStaging) error
}

func NewCommandProcess(pythonPath string) *CommandProcess {
	return &CommandProcess{pythonPath: strings.TrimSpace(pythonPath), cpuSeconds: int(defaultWorkerTimeout.Seconds())}
}

func (p *CommandProcess) Run(ctx context.Context, spec ProcessSpec, emit func(WorkerEvent) error) (ProcessResult, error) {
	if err := p.validateSpec(spec); err != nil {
		return ProcessResult{}, err
	}
	staging, err := prepareWorkerStaging(spec)
	if err != nil {
		return ProcessResult{}, err
	}
	retainStaging := false
	defer func() {
		if !retainStaging {
			staging.cleanup()
		}
	}()
	cmd, stdout, brokerSession, err := p.prepareWorkerCommand(ctx, spec, staging)
	if err != nil {
		return ProcessResult{}, err
	}
	defer brokerSession.close()
	if err := cmd.Start(); err != nil {
		return ProcessResult{}, fmt.Errorf("start evaluation worker: %w", err)
	}
	brokerSession.start(ctx)
	if err := consumeWorkerEvents(ctx, cmd, brokerSession, stdout, emit); err != nil {
		return ProcessResult{}, err
	}
	transcript := brokerSession.broker.transcript(time.Now().UTC())
	publishEvidence := staging.importEvidence
	if p.publishEvidence != nil {
		publishEvidence = func() error { return p.publishEvidence(staging) }
	}
	retainStaging = true
	return ProcessResult{
		ExecutionTranscript: &transcript,
		publishEvidence:     publishEvidence,
		discardEvidence:     staging.cleanup,
	}, nil
}

func (p *CommandProcess) validateSpec(spec ProcessSpec) error {
	if p.pythonPath == "" {
		return fmt.Errorf("python interpreter is not configured")
	}
	if strings.TrimSpace(spec.SuiteStorePath) == "" {
		return fmt.Errorf("evaluation suite store is not configured")
	}
	return nil
}

func (p *CommandProcess) prepareWorkerCommand(
	ctx context.Context,
	spec ProcessSpec,
	staging *workerStaging,
) (*exec.Cmd, io.ReadCloser, *workerBrokerSession, error) {
	manifest, _, err := readRunManifestStrict(staging.manifestPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("read evaluation worker broker manifest: %w", err)
	}
	credentials, err := p.credentialsForWorker(spec, manifest)
	if err != nil {
		return nil, nil, nil, err
	}
	broker := newWorkerHTTPBroker(manifest, credentials)
	broker.controlledPair = spec.controlledPair
	brokerSession, err := newWorkerBrokerSession(broker)
	if err != nil {
		return nil, nil, nil, err
	}
	prepared := false
	defer func() {
		if !prepared {
			brokerSession.close()
		}
	}()
	cmd, err := p.newWorkerCommand(ctx, spec, staging, brokerSession)
	if err != nil {
		return nil, nil, nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("open evaluation worker stdout: %w", err)
	}
	prepared = true
	return cmd, stdout, brokerSession, nil
}

func (p *CommandProcess) credentialsForWorker(
	spec ProcessSpec,
	manifest RunManifest,
) (workerBrokerCredentials, error) {
	if spec.controlledPair != nil {
		return spec.controlledPair.credentials, nil
	}
	return p.brokerCredentials(manifest)
}

func (p *CommandProcess) newWorkerCommand(
	ctx context.Context,
	spec ProcessSpec,
	staging *workerStaging,
	brokerSession *workerBrokerSession,
) (*exec.Cmd, error) {
	pythonPath, err := resolveWorkerPython(p.pythonPath)
	if err != nil {
		return nil, err
	}
	workerScript, err := p.resolveWorkerScript()
	if err != nil {
		return nil, err
	}
	cpuSeconds := p.cpuSeconds
	if cpuSeconds < 1 {
		cpuSeconds = 1
	}
	// The executable and argument grammar are server configured and fixed.
	//nolint:gosec
	cmd := exec.CommandContext(
		ctx,
		pythonPath,
		"-I", workerScript,
		"--manifest", staging.manifestPath,
		"--store", staging.storePath,
		"--suite-store", spec.SuiteStorePath,
		"--cpu-seconds", strconv.Itoa(cpuSeconds),
		"--broker-request-fd", strconv.Itoa(workerBrokerRequestFD),
		"--broker-response-fd", strconv.Itoa(workerBrokerResponseFD),
		"--events-stdout",
	)
	cmd.ExtraFiles = brokerSession.childFiles()
	configureWorkerProcessGroup(cmd)
	cmd.Cancel = func() error { return terminateWorkerProcessGroup(cmd) }
	cmd.WaitDelay = workerProcessWaitDelay
	cmd.Env = isolatedWorkerEnvironment(staging.root)
	cmd.Dir = staging.root
	cmd.Stderr = p.diagnosticSink
	if cmd.Stderr == nil {
		cmd.Stderr = io.Discard // Worker stderr may contain private cases or provider diagnostics.
	}
	return cmd, nil
}

func consumeWorkerEvents(
	ctx context.Context,
	cmd *exec.Cmd,
	brokerSession *workerBrokerSession,
	stdout io.Reader,
	emit func(WorkerEvent) error,
) error {
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 4*1024), maxWorkerEventLineBytes)
	for scanner.Scan() {
		event, err := decodeWorkerEvent(scanner.Bytes())
		if err != nil {
			stopWorkerProcess(cmd, brokerSession)
			return err
		}
		if err := emit(event); err != nil {
			stopWorkerProcess(cmd, brokerSession)
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		// A protocol line can exceed Scanner's bound while the worker and its
		// descendants keep running. Kill before Wait so malformed stdout cannot
		// retain a concurrency slot until the run TTL.
		stopWorkerProcess(cmd, brokerSession)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("read evaluation worker events: %w", err)
	}
	return waitForWorkerProcess(ctx, cmd, brokerSession)
}

func stopWorkerProcess(cmd *exec.Cmd, brokerSession *workerBrokerSession) {
	_ = terminateWorkerProcessGroup(cmd)
	_ = cmd.Wait()
	_ = brokerSession.wait()
}

func waitForWorkerProcess(ctx context.Context, cmd *exec.Cmd, brokerSession *workerBrokerSession) error {
	waitErr := cmd.Wait()
	brokerErr := brokerSession.wait()
	// A successful worker may still have spawned a detached descendant that no
	// longer owns stdout. Reap the server-owned process group before making its
	// staged evidence available for server-owned publication.
	_ = terminateWorkerProcessGroup(cmd)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if waitErr != nil {
		return fmt.Errorf("evaluation worker exited unsuccessfully: %w", waitErr)
	}
	if brokerErr != nil {
		return fmt.Errorf("evaluation worker HTTP broker failed: %w", brokerErr)
	}
	return nil
}

func decodeWorkerEvent(line []byte) (WorkerEvent, error) {
	if len(line) == 0 || len(line) > maxWorkerEventLineBytes || !utf8.Valid(line) {
		return WorkerEvent{}, fmt.Errorf("evaluation worker event exceeds the safe protocol envelope")
	}
	if err := rejectDuplicateJSONKeys(line); err != nil {
		return WorkerEvent{}, fmt.Errorf("decode evaluation worker event: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	var event WorkerEvent
	if err := decoder.Decode(&event); err != nil {
		return WorkerEvent{}, fmt.Errorf("decode evaluation worker event: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return WorkerEvent{}, err
	}
	return sanitizeWorkerEvent(event)
}

func sanitizeWorkerEvent(event WorkerEvent) (WorkerEvent, error) {
	if !allowedWorkerEventType(event.Type) {
		return WorkerEvent{}, fmt.Errorf("unknown evaluation worker event type %q", event.Type)
	}
	message := strings.TrimSpace(event.Message)
	if message == "" || message != event.Message || len(message) > maxWorkerMessageBytes {
		return WorkerEvent{}, fmt.Errorf("evaluation worker event message is invalid")
	}
	if event.Progress != nil {
		progressMessage := strings.TrimSpace(event.Progress.Message)
		if progressMessage != event.Progress.Message || len(progressMessage) > maxWorkerMessageBytes {
			return WorkerEvent{}, fmt.Errorf("evaluation worker progress message is invalid")
		}
	}
	if err := validateWorkerEventPayload(event.Type, event.Payload); err != nil {
		return WorkerEvent{}, err
	}
	event.Message = publicWorkerEventMessage(event.Type)
	if event.Progress != nil {
		progress := *event.Progress
		progress.Message = event.Message
		event.Progress = &progress
	}
	return event, nil
}

func validateWorkerEventPayload(eventType string, payload *WorkerEventPayload) error {
	if payload == nil {
		if eventType == "track" || eventType == "completed" {
			return fmt.Errorf("%s worker event requires a typed payload", eventType)
		}
		return nil
	}
	if payload.RecordCount != nil && (*payload.RecordCount < 0 || *payload.RecordCount > maxWorkerRecordCount) {
		return fmt.Errorf("evaluation worker record_count is outside the safe protocol envelope")
	}
	switch eventType {
	case "track":
		if payload.RecordCount == nil || payload.Verdict != "" {
			return fmt.Errorf("track worker event requires only record_count payload")
		}
	case "completed":
		if payload.RecordCount != nil || !validDecisionVerdict(payload.Verdict) {
			return fmt.Errorf("completed worker event requires only verdict payload")
		}
	default:
		return fmt.Errorf("worker event type %q does not accept a payload", eventType)
	}
	return nil
}

func publicWorkerEventMessage(eventType string) string {
	switch eventType {
	case "snapshot":
		return "Evaluation snapshot validated"
	case "progress":
		return "Evaluation progress updated"
	case "track":
		return "Evaluation track evidence collected"
	case "gate":
		return "Evaluation gate updated"
	case "artifact":
		return "Evaluation artifact finalized"
	case "completed":
		return "Evaluation worker completed"
	case "failed":
		return "Evaluation worker reported failure"
	case "cancelled":
		return "Evaluation worker reported cancellation"
	default:
		return "Evaluation worker event"
	}
}

func allowedWorkerEventType(eventType string) bool {
	switch eventType {
	case "snapshot", "progress", "track", "gate", "artifact", "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}
