package evaluationplane

type methodRecordAttestation struct {
	Robustness robustnessMethodAttestation
	AgentTask  agentTaskMethodAttestation
	Recovery   recoveryMethodAttestation
	Production productionMethodAttestation
	HardPolicy hardPolicyMethodAttestation
	R2Outcomes []CompoundModelBudgetOutcome
}

type methodRecordReducer struct {
	records []executionRecordEvidence
}

func newMethodRecordReducer() *methodRecordReducer {
	return &methodRecordReducer{}
}

func (reducer *methodRecordReducer) observe(record executionRecordEvidence) error {
	reducer.records = append(reducer.records, record)
	return nil
}

func (reducer *methodRecordReducer) finalize() (methodRecordAttestation, error) {
	robustness, err := reduceRobustnessMethod(reducer.records)
	if err != nil {
		return methodRecordAttestation{}, err
	}
	agentTask, err := reduceAgentTaskMethod(reducer.records)
	if err != nil {
		return methodRecordAttestation{}, err
	}
	recovery, err := reduceRecoveryMethod(reducer.records)
	if err != nil {
		return methodRecordAttestation{}, err
	}
	production, err := reduceProductionMethod(reducer.records)
	if err != nil {
		return methodRecordAttestation{}, err
	}
	hardPolicy, err := reduceHardPolicyMethod(reducer.records)
	if err != nil {
		return methodRecordAttestation{}, err
	}
	r2Outcomes := make([]CompoundModelBudgetOutcome, 0)
	for _, record := range reducer.records {
		if record.MethodID == nil {
			continue
		}
		// validateV2MethodCoordinates has already made this conversion total.
		slices := make([]SliceRef, len(record.SliceIDs))
		for index, id := range record.SliceIDs {
			slices[index] = SliceRef{SchemaVersion: EvaluationMethodContractVersion, ID: id}
		}
		r2Outcomes = append(r2Outcomes, CompoundModelBudgetOutcome{
			CaseID: record.CaseID, Action: ActionRef{SchemaVersion: EvaluationMethodContractVersion, ID: *record.ActionID}, Budget: *record.BudgetTokens,
			Score: *record.Quality, SliceRefs: slices,
		})
	}
	return methodRecordAttestation{
		Robustness: robustness, AgentTask: agentTask, Recovery: recovery,
		Production: production, HardPolicy: hardPolicy, R2Outcomes: r2Outcomes,
	}, nil
}
