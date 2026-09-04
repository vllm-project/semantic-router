package evaluationplane

import "fmt"

type agentTaskMethodAttestation struct {
	AttemptCount                 int
	LedgerTotalAttemptCount      int
	DistinctTaskCount            int
	LedgerTotalDistinctTaskCount int
	MinimumDistinctTaskCount     int
	MinimumAttemptsPerTask       int
	SuccessfulAttemptCount       int
	ReliableTaskCount            int
	ToolCallCount                int64
	ToolRequiredAttemptCount     int
	PureReasoningAttemptCount    int
	RequiredToolReceiptCoverage  *float64
	TaskSuccessRate              *float64
	TaskSuccessRateLower95       *float64
	TaskReliability              *float64
	TaskReliabilityLower95       *float64
	MeanTaskScore                *float64
	MeanTrajectorySteps          *float64
	InvalidToolCallRate          *float64
	PrivacyExposuresPerAttempt   *float64
	TotalCostUSD                 *float64
	CostPerSuccessfulAttemptUSD  *float64
	PolicySnapshotDigest         string
	ConfigDigest                 string
	TargetID                     string
	BackendTopologyDigest        string
	MixtureSnapshotDigest        string
	Complete                     bool
}

type agentTaskReduction struct {
	first                           *agentTaskMethodEvidence
	attempts                        map[string]struct{}
	trajectories                    map[string]struct{}
	receipts                        map[string]struct{}
	taskAttempts                    map[string]int
	taskAllSuccessful               map[string]bool
	taskContracts                   map[string]agentTaskRepeatedTaskContract
	successful                      int
	totalScore                      float64
	totalSteps                      int64
	totalToolCalls                  int64
	totalInvalidCalls               int64
	totalPrivacy                    int64
	totalCost                       float64
	toolRequiredAttempts            int
	pureReasoningAttempts           int
	toolRequiredAttemptsWithReceipt int
	count                           int
}

func agentTaskContractEqual(left, right agentTaskMethodEvidence) bool {
	return left.LedgerID == right.LedgerID && left.SourceID == right.SourceID &&
		left.SuiteID == right.SuiteID && left.SuiteRevision == right.SuiteRevision &&
		left.TaskSetDigest == right.TaskSetDigest && left.BenchmarkParityClaim == right.BenchmarkParityClaim &&
		left.ExecutionSemantics == right.ExecutionSemantics && left.PolicySnapshotDigest == right.PolicySnapshotDigest &&
		left.ConfigDigest == right.ConfigDigest && left.TargetID == right.TargetID &&
		left.BackendTopologyDigest == right.BackendTopologyDigest && left.MixtureSnapshotDigest == right.MixtureSnapshotDigest &&
		left.LedgerTotalAttemptCount == right.LedgerTotalAttemptCount &&
		left.LedgerTotalDistinctTaskCount == right.LedgerTotalDistinctTaskCount &&
		left.MinimumDistinctTaskCount == right.MinimumDistinctTaskCount &&
		left.MinimumAttemptsPerTask == right.MinimumAttemptsPerTask
}

func reduceAgentTaskMethod(records []executionRecordEvidence) (agentTaskMethodAttestation, error) {
	reduction := newAgentTaskReduction()
	for _, record := range records {
		method := record.AgentTask
		if method == nil {
			continue
		}
		if err := reduction.add(method); err != nil {
			return agentTaskMethodAttestation{}, err
		}
	}
	return reduction.attestation(), nil
}

func newAgentTaskReduction() *agentTaskReduction {
	return &agentTaskReduction{
		attempts:          make(map[string]struct{}),
		trajectories:      make(map[string]struct{}),
		receipts:          make(map[string]struct{}),
		taskAttempts:      make(map[string]int),
		taskAllSuccessful: make(map[string]bool),
		taskContracts:     make(map[string]agentTaskRepeatedTaskContract),
	}
}

func (reduction *agentTaskReduction) add(method *agentTaskMethodEvidence) error {
	if reduction.first == nil {
		copyMethod := *method
		reduction.first = &copyMethod
	} else if !agentTaskContractEqual(*reduction.first, *method) {
		return fmt.Errorf("agent-task rows mix sealed ledger contracts")
	}
	if _, duplicate := reduction.attempts[method.AttemptID]; duplicate {
		return fmt.Errorf("agent-task attempt identities must be unique")
	}
	if _, duplicate := reduction.trajectories[method.TrajectoryID]; duplicate {
		return fmt.Errorf("agent-task trajectory identities must be unique")
	}
	reduction.attempts[method.AttemptID] = struct{}{}
	reduction.trajectories[method.TrajectoryID] = struct{}{}
	for _, receipt := range agentTaskAttemptReceipts(*method) {
		if _, duplicate := reduction.receipts[receipt]; duplicate {
			return fmt.Errorf("agent-task receipts must be unique")
		}
		reduction.receipts[receipt] = struct{}{}
	}
	contract := agentTaskRepeatedTaskContract{
		taskSpecDigest: method.TaskSpecDigest, graderID: method.GraderID,
		graderRevisionDigest: method.GraderRevisionDigest, successThreshold: method.SuccessThreshold,
		toolPolicy: method.ToolPolicy,
	}
	if prior, present := reduction.taskContracts[method.TaskID]; present && !prior.equal(contract) {
		return fmt.Errorf("agent-task repetitions mix task or grader contracts")
	}
	reduction.taskContracts[method.TaskID] = contract
	reduction.taskAttempts[method.TaskID]++
	if reduction.taskAttempts[method.TaskID] == 1 {
		reduction.taskAllSuccessful[method.TaskID] = true
	}
	reduction.taskAllSuccessful[method.TaskID] = reduction.taskAllSuccessful[method.TaskID] && method.TaskSuccess
	if method.TaskSuccess {
		reduction.successful++
	}
	reduction.totalScore += method.TaskScore
	reduction.totalSteps += method.TrajectorySteps
	reduction.totalToolCalls += method.ToolCallCount
	if method.ToolPolicy.RequiresTools {
		reduction.toolRequiredAttempts++
		for _, call := range method.ToolCalls {
			if call.Outcome == "executed" && call.ExecutionReceiptDigest != nil {
				reduction.toolRequiredAttemptsWithReceipt++
				break
			}
		}
	} else {
		reduction.pureReasoningAttempts++
	}
	reduction.totalInvalidCalls += method.InvalidToolCallCount
	reduction.totalPrivacy += method.PrivacyExposureCount
	reduction.totalCost += method.TotalCostUSD
	if !finiteFloat(reduction.totalScore) || !finiteFloat(reduction.totalCost) {
		return fmt.Errorf("agent-task metric aggregate is not finite")
	}
	reduction.count++
	return nil
}

func (reduction *agentTaskReduction) attestation() agentTaskMethodAttestation {
	if reduction.first == nil {
		return agentTaskMethodAttestation{}
	}
	reliableTasks := 0
	minimumAttemptsSatisfied := true
	for taskID, allSuccessful := range reduction.taskAllSuccessful {
		if reduction.taskAttempts[taskID] < reduction.first.MinimumAttemptsPerTask {
			minimumAttemptsSatisfied = false
		}
		if allSuccessful {
			reliableTasks++
		}
	}
	distinctTasks := len(reduction.taskAttempts)
	successRate := float64(reduction.successful) / float64(reduction.count)
	reliability := float64(reliableTasks) / float64(distinctTasks)
	meanScore := reduction.totalScore / float64(reduction.count)
	meanSteps := float64(reduction.totalSteps) / float64(reduction.count)
	privacyPerAttempt := float64(reduction.totalPrivacy) / float64(reduction.count)
	attestation := agentTaskMethodAttestation{
		AttemptCount: reduction.count, LedgerTotalAttemptCount: reduction.first.LedgerTotalAttemptCount,
		DistinctTaskCount: distinctTasks, LedgerTotalDistinctTaskCount: reduction.first.LedgerTotalDistinctTaskCount,
		MinimumDistinctTaskCount: reduction.first.MinimumDistinctTaskCount,
		MinimumAttemptsPerTask:   reduction.first.MinimumAttemptsPerTask,
		SuccessfulAttemptCount:   reduction.successful, ReliableTaskCount: reliableTasks,
		ToolCallCount: reduction.totalToolCalls, ToolRequiredAttemptCount: reduction.toolRequiredAttempts,
		PureReasoningAttemptCount:  reduction.pureReasoningAttempts,
		TaskSuccessRate:            &successRate,
		TaskSuccessRateLower95:     oneSidedWilsonLower(reduction.successful, reduction.count),
		TaskReliability:            &reliability,
		TaskReliabilityLower95:     oneSidedWilsonLower(reliableTasks, distinctTasks),
		MeanTaskScore:              &meanScore,
		MeanTrajectorySteps:        &meanSteps,
		PrivacyExposuresPerAttempt: &privacyPerAttempt, TotalCostUSD: &reduction.totalCost,
		PolicySnapshotDigest: reduction.first.PolicySnapshotDigest, ConfigDigest: reduction.first.ConfigDigest,
		TargetID: reduction.first.TargetID, BackendTopologyDigest: reduction.first.BackendTopologyDigest,
		MixtureSnapshotDigest: reduction.first.MixtureSnapshotDigest,
	}
	if reduction.toolRequiredAttempts > 0 {
		coverage := float64(reduction.toolRequiredAttemptsWithReceipt) / float64(reduction.toolRequiredAttempts)
		attestation.RequiredToolReceiptCoverage = &coverage
	}
	if reduction.totalToolCalls > 0 {
		invalidRate := float64(reduction.totalInvalidCalls) / float64(reduction.totalToolCalls)
		attestation.InvalidToolCallRate = &invalidRate
	}
	if reduction.successful > 0 {
		costPerSuccess := reduction.totalCost / float64(reduction.successful)
		attestation.CostPerSuccessfulAttemptUSD = &costPerSuccess
	}
	attestation.Complete = reduction.count == reduction.first.LedgerTotalAttemptCount &&
		distinctTasks == reduction.first.LedgerTotalDistinctTaskCount &&
		distinctTasks >= reduction.first.MinimumDistinctTaskCount && minimumAttemptsSatisfied &&
		reduction.toolRequiredAttemptsWithReceipt == reduction.toolRequiredAttempts &&
		reduction.toolRequiredAttempts+reduction.pureReasoningAttempts == reduction.count
	return attestation
}
