package extproc

const (
	learningPolicyFieldLearning        routerLearningPolicyField = "learning"
	learningPolicyFieldMethod          routerLearningPolicyField = "method"
	learningPolicyFieldMode            routerLearningPolicyField = "mode"
	learningPolicyFieldScope           routerLearningPolicyField = "scope"
	learningPolicyFieldAction          routerLearningPolicyField = "action"
	learningPolicyFieldReason          routerLearningPolicyField = "reason"
	learningPolicyFieldIdentity        routerLearningPolicyField = "identity"
	learningPolicyFieldScores          routerLearningPolicyField = "scores"
	learningPolicyFieldCandidateSet    routerLearningPolicyField = "candidate_set"
	learningPolicyFieldStrategy        routerLearningPolicyField = "strategy"
	learningPolicyFieldBaseModel       routerLearningPolicyField = "base_model"
	learningPolicyFieldProposalModel   routerLearningPolicyField = "proposal_model"
	learningPolicyFieldFinalModel      routerLearningPolicyField = "final_model"
	learningPolicyFieldSampling        routerLearningPolicyField = "sampling"
	learningPolicyFieldSwitchCost      routerLearningPolicyField = "switch_cost"
	learningPolicyFieldSwitchMargin    routerLearningPolicyField = "switch_margin"
	learningPolicyFieldStabilityWeight routerLearningPolicyField = "stability_weight"

	learningPolicyFieldPhase             routerLearningPolicyField = "phase"
	learningPolicyFieldCurrentModel      routerLearningPolicyField = "current_model"
	learningPolicyFieldBaseSelectedModel routerLearningPolicyField = "base_selected_model"
	learningPolicyFieldSelectedModel     routerLearningPolicyField = "selected_model"
	learningPolicyFieldSelectedScore     routerLearningPolicyField = "selected_score"
	learningPolicyFieldHardLocked        routerLearningPolicyField = "hard_locked"
	learningPolicyFieldHardLockReason    routerLearningPolicyField = "hard_lock_reason"
	learningPolicyFieldDecisionReason    routerLearningPolicyField = "decision_reason"
	learningPolicyFieldDecision          routerLearningPolicyField = "decision"
	learningPolicyFieldDecisionTier      routerLearningPolicyField = "decision_tier"
	learningPolicyFieldRescue            routerLearningPolicyField = "rescue"
)

type routerLearningPolicyField string
