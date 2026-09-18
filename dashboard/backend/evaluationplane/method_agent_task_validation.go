package evaluationplane

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
)

func validateAgentTaskMethod(method agentTaskMethodEvidence, record executionRecordEvidence) error {
	if method.ContractVersion != agentTaskAttemptContractVersion || method.MethodID != agentTaskMethodID ||
		!validMethodID(method.LedgerID) || !validMethodID(method.SourceID) || !validMethodID(method.SuiteID) ||
		!validMethodDigest(method.SuiteRevision) || !validMethodDigest(method.TaskSetDigest) ||
		method.BenchmarkParityClaim != agentTaskBenchmarkParityClaim || method.ExecutionSemantics != agentTaskExecutionSemantics ||
		!validMethodDigest(method.PolicySnapshotDigest) || !validMethodDigest(method.ConfigDigest) ||
		!validMethodID(method.TargetID) || !validMethodDigest(method.BackendTopologyDigest) ||
		!validMethodDigest(method.MixtureSnapshotDigest) || method.LedgerTotalAttemptCount < 1 ||
		method.LedgerTotalDistinctTaskCount < 1 || method.MinimumDistinctTaskCount < minimumAgentTaskDistinctTaskCount ||
		method.MinimumAttemptsPerTask < minimumAgentTaskAttemptsPerTask || !validMethodID(method.TaskID) ||
		!validMethodDigest(method.TaskSpecDigest) || !validMethodID(method.AttemptID) ||
		!validMethodID(method.RepetitionID) || !validMethodID(method.TrajectoryID) ||
		method.Seed < 0 || method.Seed > math.MaxUint32 || !validMethodID(method.SelectedArmID) ||
		!finiteFloat(method.TaskScore) || method.TaskScore < 0 || method.TaskScore > 1 ||
		!finiteFloat(method.SuccessThreshold) || method.SuccessThreshold < 0 || method.SuccessThreshold > 1 ||
		!validMethodID(method.GraderID) || !validMethodDigest(method.GraderRevisionDigest) ||
		!validMethodDigest(method.GradingReceiptDigest) || !validMethodDigest(method.PrivacyAuditReceiptDigest) ||
		!validMethodDigest(method.ExecutionReceiptDigest) || method.TrajectorySteps < 1 ||
		method.ToolCallCount < 0 || method.ToolCallCount > maximumAgentTaskToolCallsPerAttempt ||
		method.ToolCallCount > method.TrajectorySteps ||
		method.InvalidToolCallCount < 0 || method.InvalidToolCallCount > method.ToolCallCount ||
		method.PrivacyExposureCount < 0 || method.InputTokens < 0 || method.OutputTokens < 0 ||
		invalidCost(method.ModelCostUSD) || invalidCost(method.ToolCostUSD) ||
		invalidCost(method.EvaluationCostUSD) || invalidCost(method.TotalCostUSD) ||
		method.StartedAt.IsZero() || method.CompletedAt.Before(method.StartedAt) ||
		method.GradedAt.Before(method.CompletedAt) || method.PrivacyAuditedAt.Before(method.CompletedAt) ||
		method.ToolCalls == nil || int64(len(method.ToolCalls)) != method.ToolCallCount {
		return fmt.Errorf("agent-task method evidence is invalid")
	}
	if method.TaskSuccess != (method.TaskScore >= method.SuccessThreshold) {
		return fmt.Errorf("agent-task outcome contradicts its frozen success threshold")
	}
	if err := validateAgentTaskToolPolicy(method.ToolPolicy, method.ToolCalls); err != nil {
		return err
	}
	invalidCalls := int64(0)
	toolCost := 0.0
	toolIDs := make(map[string]struct{}, len(method.ToolCalls))
	receipts := map[string]struct{}{
		method.ExecutionReceiptDigest:    {},
		method.GradingReceiptDigest:      {},
		method.PrivacyAuditReceiptDigest: {},
	}
	if len(receipts) != 3 {
		return fmt.Errorf("agent-task attempt receipts must be unique")
	}
	for index, tool := range method.ToolCalls {
		if err := validateAgentTaskToolCall(tool, int64(index+1), method); err != nil {
			return err
		}
		if _, duplicate := toolIDs[tool.ToolCallID]; duplicate {
			return fmt.Errorf("agent-task tool-call identities must be unique")
		}
		toolIDs[tool.ToolCallID] = struct{}{}
		if tool.Outcome == "rejected_invalid" {
			invalidCalls++
		}
		if tool.ExecutionReceiptDigest != nil {
			if _, duplicate := receipts[*tool.ExecutionReceiptDigest]; duplicate {
				return fmt.Errorf("agent-task execution receipts must be unique")
			}
			receipts[*tool.ExecutionReceiptDigest] = struct{}{}
		}
		toolCost += tool.CostUSD
		if !finiteFloat(toolCost) {
			return fmt.Errorf("agent-task tool cost aggregate is not finite")
		}
	}
	if invalidCalls != method.InvalidToolCallCount || !reducedFloatsEqual(toolCost, method.ToolCostUSD) ||
		!reducedFloatsEqual(method.ModelCostUSD+method.ToolCostUSD+method.EvaluationCostUSD, method.TotalCostUSD) {
		return fmt.Errorf("agent-task counters or complete cost do not match the trajectory")
	}
	runtimeCost := method.ModelCostUSD + method.ToolCostUSD
	if record.Status == "unavailable" || record.Success == nil || *record.Success != method.TaskSuccess ||
		(record.Status == "succeeded") != method.TaskSuccess || record.Quality == nil ||
		!reducedFloatsEqual(*record.Quality, method.TaskScore) || record.SelectedArmID == nil ||
		*record.SelectedArmID != method.SelectedArmID || record.TrajectorySteps == nil ||
		*record.TrajectorySteps != method.TrajectorySteps || record.ToolCalls == nil ||
		*record.ToolCalls != method.ToolCallCount || record.InvalidToolCalls == nil ||
		*record.InvalidToolCalls != method.InvalidToolCallCount || record.PrivacyViolations == nil ||
		*record.PrivacyViolations != method.PrivacyExposureCount || record.InputTokens == nil ||
		*record.InputTokens != method.InputTokens || record.OutputTokens == nil || *record.OutputTokens != method.OutputTokens ||
		record.RuntimeCost == nil || !reducedFloatsEqual(*record.RuntimeCost, runtimeCost) ||
		record.EvaluationCost == nil || !reducedFloatsEqual(*record.EvaluationCost, method.EvaluationCostUSD) ||
		record.Grader == nil || *record.Grader != method.GraderID || record.EvidenceKind == nil ||
		*record.EvidenceKind != agentTaskEvidenceKind || record.BrokerReceipt == nil {
		return fmt.Errorf("agent-task record does not bind its exact task attempt")
	}
	return nil
}

func validateAgentTaskToolPolicy(policy agentTaskToolPolicy, calls []agentTaskToolCallEvidence) error {
	if policy.ExpectedTools == nil {
		return fmt.Errorf("agent-task tool policy must be explicit")
	}
	expected := make(map[string]struct{}, len(policy.ExpectedTools))
	previous := ""
	for _, toolName := range policy.ExpectedTools {
		if !validMethodID(toolName) || (previous != "" && toolName <= previous) {
			return fmt.Errorf("agent-task expected tools must be unique and sorted")
		}
		expected[toolName] = struct{}{}
		previous = toolName
	}
	if policy.RequiresTools != (len(policy.ExpectedTools) > 0) {
		return fmt.Errorf("agent-task requires_tools contradicts expected_tools")
	}
	if !policy.RequiresTools {
		if len(calls) != 0 {
			return fmt.Errorf("pure-reasoning agent task cannot contain tool calls")
		}
		return nil
	}
	executedExpected := 0
	for _, call := range calls {
		if call.Outcome != "executed" {
			continue
		}
		if _, allowed := expected[call.ToolName]; !allowed {
			return fmt.Errorf("agent-task executed a tool outside its expected-tool policy")
		}
		executedExpected++
	}
	if executedExpected == 0 {
		return fmt.Errorf("tool-required agent-task attempt lacks a provider-executed expected-tool receipt")
	}
	return nil
}

func validateAgentTaskToolCall(tool agentTaskToolCallEvidence, expectedSequence int64, method agentTaskMethodEvidence) error {
	if tool.Sequence != expectedSequence || !validMethodID(tool.ToolCallID) || !validMethodID(tool.ToolName) ||
		!validMethodDigest(tool.ArgumentsDigest) || invalidCost(tool.CostUSD) {
		return fmt.Errorf("agent-task tool-call evidence is invalid")
	}
	switch tool.Outcome {
	case "executed":
		if tool.ResultDigest == nil || !validMethodDigest(*tool.ResultDigest) ||
			tool.ExecutionReceiptDigest == nil || !validMethodDigest(*tool.ExecutionReceiptDigest) ||
			tool.StartedAt == nil || tool.CompletedAt == nil || tool.StartedAt.Before(method.StartedAt) ||
			tool.CompletedAt.Before(*tool.StartedAt) || tool.CompletedAt.After(method.CompletedAt) {
			return fmt.Errorf("executed agent-task tool call lacks a real bounded receipt")
		}
	case "rejected_invalid":
		if tool.ResultDigest != nil || tool.ExecutionReceiptDigest != nil || tool.StartedAt != nil ||
			tool.CompletedAt != nil || tool.CostUSD != 0 {
			return fmt.Errorf("invalid agent-task tool call cannot claim execution")
		}
	default:
		return fmt.Errorf("agent-task tool-call outcome is invalid")
	}
	return nil
}

func validateAgentTaskLedgerPayload(
	payload agentTaskLedgerPayload,
	records []executionRecordEvidence,
	manifest RunManifest,
) error {
	expectedMixture, err := methodManifestMixtureBinding(manifest)
	if err != nil {
		return err
	}
	if payload.ContractVersion != agentTaskLedgerContractVersion || payload.Environment != "production" ||
		!validMethodID(payload.LedgerID) || !validMethodID(payload.SourceID) || !validMethodID(payload.SuiteID) ||
		!validMethodDigest(payload.SuiteRevision) || !validMethodDigest(payload.TaskSetDigest) ||
		payload.BenchmarkParityClaim != agentTaskBenchmarkParityClaim || payload.ExecutionSemantics != agentTaskExecutionSemantics ||
		!validMethodDigest(payload.ProviderAttestationDigest) || payload.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		payload.ConfigDigest != manifest.ConfigDigest || payload.TargetID != manifest.Target.ID ||
		payload.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
		!reflect.DeepEqual(payload.Mixture, expectedMixture) || payload.LedgerTotalAttemptCount != len(payload.Attempts) ||
		payload.LedgerTotalAttemptCount != len(records) || payload.LedgerTotalDistinctTaskCount < 1 ||
		payload.MinimumDistinctTaskCount < minimumAgentTaskDistinctTaskCount ||
		payload.MinimumAttemptsPerTask < minimumAgentTaskAttemptsPerTask ||
		!validSealedMethodWindow(payload.WindowStartedAt, payload.WindowEndedAt, payload.SealedAt) {
		return fmt.Errorf("agent-task ledger envelope violates its sealed contract")
	}
	computedTaskSetDigest, err := agentTaskSetDigest(payload.Attempts)
	if err != nil || computedTaskSetDigest != payload.TaskSetDigest {
		return fmt.Errorf("agent-task ledger task-set digest does not bind membership")
	}
	recordsByAttempt, err := indexAgentTaskRecords(records)
	if err != nil {
		return err
	}
	byAttempt, err := validateAgentTaskLedgerAttempts(payload, manifest, expectedMixture, recordsByAttempt)
	if err != nil {
		return err
	}
	return validateAgentTaskLedgerMembership(payload, records, byAttempt)
}

func indexAgentTaskRecords(records []executionRecordEvidence) (map[string]executionRecordEvidence, error) {
	recordsByAttempt := make(map[string]executionRecordEvidence, len(records))
	for _, record := range records {
		if record.AgentTask == nil {
			return nil, fmt.Errorf("agent-task ledger is bound to a non-task record")
		}
		if _, duplicate := recordsByAttempt[record.AgentTask.AttemptID]; duplicate {
			return nil, fmt.Errorf("agent-task records contain a duplicate ledger attempt")
		}
		recordsByAttempt[record.AgentTask.AttemptID] = record
	}
	return recordsByAttempt, nil
}

func validateAgentTaskLedgerAttempts(
	payload agentTaskLedgerPayload,
	manifest RunManifest,
	expectedMixture methodMixtureBinding,
	recordsByAttempt map[string]executionRecordEvidence,
) (map[string]agentTaskMethodEvidence, error) {
	byAttempt := make(map[string]agentTaskMethodEvidence, len(payload.Attempts))
	taskContracts := make(map[string]agentTaskRepeatedTaskContract)
	taskAttempts := make(map[string]int)
	taskRepetitions := make(map[string]struct{})
	taskSeeds := make(map[string]struct{})
	trajectoryIDs := make(map[string]struct{})
	allReceipts := map[string]struct{}{payload.ProviderAttestationDigest: {}}
	for _, attempt := range payload.Attempts {
		if err := validateAgentTaskAttemptEnvelope(attempt, payload, expectedMixture); err != nil {
			return nil, err
		}
		boundRecord, present := recordsByAttempt[attempt.AttemptID]
		if !present {
			return nil, fmt.Errorf("agent-task ledger contains an unreported attempt")
		}
		if err := validateAgentTaskMethod(attempt, boundRecord); err != nil {
			return nil, err
		}
		arm, present := manifestModelArm(manifest, attempt.SelectedArmID)
		if !present {
			return nil, fmt.Errorf("agent-task attempt selected outside the frozen Mixture")
		}
		expectedModelCost := (float64(attempt.InputTokens)*arm.InputCostPerMillionTokensUSD +
			float64(attempt.OutputTokens)*arm.OutputCostPerMillionTokensUSD) / 1_000_000
		if !finiteFloat(expectedModelCost) || !reducedFloatsEqual(attempt.ModelCostUSD, expectedModelCost) {
			return nil, fmt.Errorf("agent-task model cost differs from the frozen Mixture pricing")
		}
		if _, duplicate := byAttempt[attempt.AttemptID]; duplicate {
			return nil, fmt.Errorf("agent-task ledger contains a duplicate attempt")
		}
		if _, duplicate := trajectoryIDs[attempt.TrajectoryID]; duplicate {
			return nil, fmt.Errorf("agent-task ledger contains a duplicate trajectory")
		}
		trajectoryIDs[attempt.TrajectoryID] = struct{}{}
		repetitionKey := attempt.TaskID + "\x00" + attempt.RepetitionID
		seedKey := fmt.Sprintf("%s\x00%d", attempt.TaskID, attempt.Seed)
		if _, duplicate := taskRepetitions[repetitionKey]; duplicate {
			return nil, fmt.Errorf("agent-task ledger contains a duplicate task repetition")
		}
		if _, duplicate := taskSeeds[seedKey]; duplicate {
			return nil, fmt.Errorf("agent-task ledger reuses a task seed")
		}
		taskRepetitions[repetitionKey] = struct{}{}
		taskSeeds[seedKey] = struct{}{}
		contract := agentTaskRepeatedTaskContract{
			taskSpecDigest: attempt.TaskSpecDigest, graderID: attempt.GraderID,
			graderRevisionDigest: attempt.GraderRevisionDigest, successThreshold: attempt.SuccessThreshold,
			toolPolicy: attempt.ToolPolicy,
		}
		if prior, present := taskContracts[attempt.TaskID]; present && !prior.equal(contract) {
			return nil, fmt.Errorf("agent-task repetitions change their task or grader contract")
		}
		taskContracts[attempt.TaskID] = contract
		taskAttempts[attempt.TaskID]++
		for _, receipt := range agentTaskAttemptReceipts(attempt) {
			if _, duplicate := allReceipts[receipt]; duplicate {
				return nil, fmt.Errorf("agent-task ledger reuses an execution or grading receipt")
			}
			allReceipts[receipt] = struct{}{}
		}
		byAttempt[attempt.AttemptID] = attempt
	}
	if len(taskContracts) != payload.LedgerTotalDistinctTaskCount || len(taskContracts) < payload.MinimumDistinctTaskCount {
		return nil, fmt.Errorf("agent-task ledger lacks its complete decision-grade task cohort")
	}
	for taskID, count := range taskAttempts {
		if count < payload.MinimumAttemptsPerTask {
			return nil, fmt.Errorf("agent-task %s lacks repeated reliability attempts", taskID)
		}
	}
	return byAttempt, nil
}

func validateAgentTaskLedgerMembership(
	payload agentTaskLedgerPayload,
	records []executionRecordEvidence,
	byAttempt map[string]agentTaskMethodEvidence,
) error {
	for _, record := range records {
		if record.AgentTask == nil || record.TrackID != "agentic" || record.BrokerReceipt == nil {
			return fmt.Errorf("agent-task ledger is bound to a non-task record")
		}
		attempt, present := byAttempt[record.AgentTask.AttemptID]
		caseID := methodLedgerCaseID("agent-task", payload.LedgerID, attempt.AttemptID)
		if !present || !canonicalMethodValuesEqual(attempt, *record.AgentTask) ||
			record.CaseID != caseID || record.ID != "agentic-"+caseID || record.AttemptID != "agentic-"+caseID {
			return fmt.Errorf("agent-task ledger membership differs from its emitted records")
		}
		delete(byAttempt, attempt.AttemptID)
	}
	if len(byAttempt) != 0 {
		return fmt.Errorf("agent-task ledger contains an unreported attempt")
	}
	return nil
}

type agentTaskRepeatedTaskContract struct {
	taskSpecDigest       string
	graderID             string
	graderRevisionDigest string
	successThreshold     float64
	toolPolicy           agentTaskToolPolicy
}

func (contract agentTaskRepeatedTaskContract) equal(other agentTaskRepeatedTaskContract) bool {
	return contract.taskSpecDigest == other.taskSpecDigest && contract.graderID == other.graderID &&
		contract.graderRevisionDigest == other.graderRevisionDigest &&
		reducedFloatsEqual(contract.successThreshold, other.successThreshold) &&
		reflect.DeepEqual(contract.toolPolicy, other.toolPolicy)
}

func validateAgentTaskAttemptEnvelope(
	attempt agentTaskMethodEvidence,
	payload agentTaskLedgerPayload,
	mixture methodMixtureBinding,
) error {
	if attempt.LedgerID != payload.LedgerID || attempt.SourceID != payload.SourceID ||
		attempt.SuiteID != payload.SuiteID || attempt.SuiteRevision != payload.SuiteRevision ||
		attempt.TaskSetDigest != payload.TaskSetDigest || attempt.BenchmarkParityClaim != payload.BenchmarkParityClaim ||
		attempt.ExecutionSemantics != payload.ExecutionSemantics || attempt.PolicySnapshotDigest != payload.PolicySnapshotDigest ||
		attempt.ConfigDigest != payload.ConfigDigest || attempt.TargetID != payload.TargetID ||
		attempt.BackendTopologyDigest != payload.BackendTopologyDigest || attempt.MixtureSnapshotDigest != mixture.SnapshotDigest ||
		attempt.LedgerTotalAttemptCount != payload.LedgerTotalAttemptCount ||
		attempt.LedgerTotalDistinctTaskCount != payload.LedgerTotalDistinctTaskCount ||
		attempt.MinimumDistinctTaskCount != payload.MinimumDistinctTaskCount ||
		attempt.MinimumAttemptsPerTask != payload.MinimumAttemptsPerTask ||
		attempt.StartedAt.Before(payload.WindowStartedAt) || attempt.CompletedAt.After(payload.WindowEndedAt) ||
		attempt.GradedAt.After(payload.WindowEndedAt) || attempt.PrivacyAuditedAt.After(payload.WindowEndedAt) {
		return fmt.Errorf("agent-task attempt does not bind the sealed ledger")
	}
	return nil
}

func agentTaskSetDigest(attempts []agentTaskMethodEvidence) (string, error) {
	type taskIdentity struct {
		specDigest string
		toolPolicy agentTaskToolPolicy
	}
	tasks := make(map[string]taskIdentity)
	for _, attempt := range attempts {
		identity := taskIdentity{specDigest: attempt.TaskSpecDigest, toolPolicy: attempt.ToolPolicy}
		if existing, present := tasks[attempt.TaskID]; present && !reflect.DeepEqual(existing, identity) {
			return "", fmt.Errorf("agent-task identity maps to multiple task specs")
		}
		tasks[attempt.TaskID] = identity
	}
	ids := make([]string, 0, len(tasks))
	for taskID := range tasks {
		ids = append(ids, taskID)
	}
	sort.Strings(ids)
	var canonical strings.Builder
	canonical.WriteString("agent-task-set.v1\n")
	for _, taskID := range ids {
		identity := tasks[taskID]
		canonical.WriteString(taskID)
		canonical.WriteByte(0)
		canonical.WriteString(identity.specDigest)
		canonical.WriteByte(0)
		if identity.toolPolicy.RequiresTools {
			canonical.WriteString("requires_tools")
		} else {
			canonical.WriteString("pure_reasoning")
		}
		for _, toolName := range identity.toolPolicy.ExpectedTools {
			canonical.WriteByte(0)
			canonical.WriteString(toolName)
		}
		canonical.WriteByte('\n')
	}
	return digestString(canonical.String()), nil
}

func agentTaskAttemptReceipts(attempt agentTaskMethodEvidence) []string {
	receipts := []string{
		attempt.ExecutionReceiptDigest,
		attempt.GradingReceiptDigest,
		attempt.PrivacyAuditReceiptDigest,
	}
	for _, tool := range attempt.ToolCalls {
		if tool.ExecutionReceiptDigest != nil {
			receipts = append(receipts, *tool.ExecutionReceiptDigest)
		}
	}
	return receipts
}

func manifestModelArm(manifest RunManifest, armID string) (ModelArm, bool) {
	if manifest.Target.Mixture == nil {
		return ModelArm{}, false
	}
	for _, arm := range manifest.Target.Mixture.ModelArms {
		if arm.ID == armID {
			return arm, true
		}
	}
	return ModelArm{}, false
}
