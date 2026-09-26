package looper

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// AlgorithmTypeBase identifies the default base/simple sequential execution algorithm.
const AlgorithmTypeBase = "base"

// StageRole identifies the role and intent of a Looper stage.
type StageRole string

const (
	// StageRoleCandidate produces an internal candidate completion that may be evaluated, ranked, or judged.
	StageRoleCandidate StageRole = "candidate"
	// StageRoleVerifier evaluates or validates prior candidate outputs.
	StageRoleVerifier StageRole = "verifier"
	// StageRolePlanning generates execution plans or routing breakdowns.
	StageRolePlanning StageRole = "planning"
	// StageRoleAnalysis produces intermediate analysis or panel deliberation.
	StageRoleAnalysis StageRole = "analysis"
	// StageRoleExecution executes intermediate workflow steps or tool interactions.
	StageRoleExecution StageRole = "execution"
	// StageRoleSynthesis combines prior deliberation or candidates into a final user-visible response.
	StageRoleSynthesis StageRole = "synthesis"
	// StageRoleDirect produces the direct user-visible response without multi-stage deliberation.
	StageRoleDirect StageRole = "direct"
)

// IsUserVisible reports whether a stage with this role can be emitted directly to the end user.
func (r StageRole) IsUserVisible() bool {
	return r == StageRoleSynthesis || r == StageRoleDirect
}

// StreamingEligibility indicates whether a stage is eligible for native streaming forwarding.
type StreamingEligibility string

const (
	// StreamingEligible indicates native streaming deltas may be forwarded from upstream.
	StreamingEligible StreamingEligibility = "eligible"
	// StreamingIneligibleBufferingRequired indicates output must be buffered before release.
	StreamingIneligibleBufferingRequired StreamingEligibility = "ineligible_buffering_required"
)

// BufferingReason explains why native streaming cannot be used and buffering is required.
type BufferingReason string

const (
	// BufferingReasonNone indicates native streaming is permitted.
	BufferingReasonNone BufferingReason = ""
	// BufferingReasonNonStreamingRequest indicates the client did not request streaming.
	BufferingReasonNonStreamingRequest BufferingReason = "non_streaming_request"
	// BufferingReasonNonUserVisibleStage indicates intermediate candidate, verifier, or planner stages cannot stream to clients.
	BufferingReasonNonUserVisibleStage BufferingReason = "non_user_visible_stage"
	// BufferingReasonSelectionPending indicates candidate selection/winner is not yet known (e.g. Best-of-N, Confidence).
	BufferingReasonSelectionPending BufferingReason = "selection_pending"
	// BufferingReasonOutputContractTransformation indicates output contract requires post-processing or mutation (e.g. JSON action, single choice, reference selection).
	BufferingReasonOutputContractTransformation BufferingReason = "output_contract_transformation"
	// BufferingReasonMultiChoiceOutput indicates algorithm produces multiple side-by-side choices (e.g. Ratings).
	BufferingReasonMultiChoiceOutput BufferingReason = "multi_choice_output"
	// BufferingReasonMultiModelAggregation indicates multiple model outputs must be combined into a synthetic aggregation.
	BufferingReasonMultiModelAggregation BufferingReason = "multi_model_aggregation"
	// BufferingReasonUnsupportedAlgorithm indicates the algorithm has no defined streaming path.
	BufferingReasonUnsupportedAlgorithm BufferingReason = "unsupported_algorithm"
	// BufferingReasonNoModels indicates request had no model candidates configured.
	BufferingReasonNoModels BufferingReason = "no_models_configured"
)

// StageOwnership defines the ownership, role, and streaming eligibility of a Looper stage.
type StageOwnership struct {
	AlgorithmType      string               `json:"algorithm_type"`
	StageName          string               `json:"stage_name"`
	StageRole          StageRole            `json:"stage_role"`
	TargetModel        string               `json:"target_model,omitempty"`
	IsFinalUserVisible bool                 `json:"is_final_user_visible"`
	Eligibility        StreamingEligibility `json:"eligibility"`
	BufferingReason    BufferingReason      `json:"buffering_reason,omitempty"`
}

// RequiresOutputContractTransformation reports whether the given output contract
// specification requires mutating, normalizing, or parsing the output after generation,
// which precludes streaming raw upstream tokens directly.
func RequiresOutputContractTransformation(spec *config.OutputContractSpec) bool {
	if spec == nil {
		return false
	}
	if requestsJSONAction(spec) || requestsSingleChoice(spec) || requestsReferenceSelection(spec) {
		return true
	}
	if spec.Render != nil || spec.Normalize != nil || len(spec.Postprocess) > 0 {
		return true
	}
	if strings.TrimSpace(spec.Type) != "" {
		return true
	}
	return false
}

// ClassifyStage classifies an arbitrary Looper stage and enforces that non-user-visible
// stages can never be eligible for streaming to downstream clients.
func ClassifyStage(req *Request, stageName string, role StageRole, targetModel string) StageOwnership {
	algoType := algorithmTypeFromRequest(req)
	if !role.IsUserVisible() {
		return StageOwnership{
			AlgorithmType:      algoType,
			StageName:          stageName,
			StageRole:          role,
			TargetModel:        targetModel,
			IsFinalUserVisible: false,
			Eligibility:        StreamingIneligibleBufferingRequired,
			BufferingReason:    BufferingReasonNonUserVisibleStage,
		}
	}

	if req == nil || !req.IsStreaming {
		return StageOwnership{
			AlgorithmType:      algoType,
			StageName:          stageName,
			StageRole:          role,
			TargetModel:        targetModel,
			IsFinalUserVisible: true,
			Eligibility:        StreamingIneligibleBufferingRequired,
			BufferingReason:    BufferingReasonNonStreamingRequest,
		}
	}

	if RequiresOutputContractTransformation(req.OutputContractSpec) {
		return StageOwnership{
			AlgorithmType:      algoType,
			StageName:          stageName,
			StageRole:          role,
			TargetModel:        targetModel,
			IsFinalUserVisible: true,
			Eligibility:        StreamingIneligibleBufferingRequired,
			BufferingReason:    BufferingReasonOutputContractTransformation,
		}
	}

	return StageOwnership{
		AlgorithmType:      algoType,
		StageName:          stageName,
		StageRole:          role,
		TargetModel:        targetModel,
		IsFinalUserVisible: true,
		Eligibility:        StreamingEligible,
		BufferingReason:    BufferingReasonNone,
	}
}

// ResolveFinalStageOwnership determines the final user-visible stage and its streaming
// eligibility for each supported Looper algorithm.
func ResolveFinalStageOwnership(req *Request) StageOwnership {
	if !hasConfiguredModels(req) {
		return StageOwnership{
			AlgorithmType:      algorithmTypeFromRequest(req),
			StageName:          "",
			StageRole:          StageRoleCandidate,
			IsFinalUserVisible: false,
			Eligibility:        StreamingIneligibleBufferingRequired,
			BufferingReason:    BufferingReasonNoModels,
		}
	}

	algoType := algorithmTypeFromRequest(req)
	switch algoType {
	case config.DecisionAlgorithmFusion:
		return resolveFusionFinalStage(req)
	case config.DecisionAlgorithmReMoM:
		return resolveReMoMFinalStage(req)
	case config.DecisionAlgorithmWorkflows:
		return resolveWorkflowsFinalStage(req)
	case config.DecisionAlgorithmConfidence:
		return resolveConfidenceFinalStage(req)
	case config.DecisionAlgorithmRatings:
		return resolveRatingsFinalStage(req)
	case AlgorithmTypeBase, "simple", "":
		return resolveBaseFinalStage(req)
	default:
		return StageOwnership{
			AlgorithmType:      algoType,
			StageName:          "unknown",
			StageRole:          StageRoleCandidate,
			IsFinalUserVisible: false,
			Eligibility:        StreamingIneligibleBufferingRequired,
			BufferingReason:    BufferingReasonUnsupportedAlgorithm,
		}
	}
}

func hasConfiguredModels(req *Request) bool {
	if req == nil {
		return false
	}
	if len(req.ModelRefs) > 0 {
		return true
	}
	algoType := algorithmTypeFromRequest(req)
	switch algoType {
	case config.DecisionAlgorithmFusion:
		if req.Algorithm != nil && req.Algorithm.Fusion != nil {
			return len(req.Algorithm.Fusion.AnalysisModels) > 0 || strings.TrimSpace(req.Algorithm.Fusion.Model) != ""
		}
	case config.DecisionAlgorithmWorkflows:
		if req.Algorithm != nil && req.Algorithm.Workflows != nil {
			wf := req.Algorithm.Workflows
			return strings.TrimSpace(wf.Final.Model) != "" || strings.TrimSpace(wf.Planner.Model) != "" || len(wf.Roles) > 0
		}
	case config.DecisionAlgorithmReMoM:
		if req.Algorithm != nil && req.Algorithm.ReMoM != nil {
			return strings.TrimSpace(req.Algorithm.ReMoM.SynthesisModel) != ""
		}
	}
	return false
}

func resolveFusionFinalStage(req *Request) StageOwnership {
	targetModel := ""
	if req.Algorithm != nil && req.Algorithm.Fusion != nil {
		targetModel = strings.TrimSpace(req.Algorithm.Fusion.Model)
	}
	if targetModel == "" && len(req.ModelRefs) > 0 {
		targetModel = req.ModelRefs[0].Model
		if req.ModelRefs[0].LoRAName != "" {
			targetModel = req.ModelRefs[0].LoRAName
		}
	}
	if targetModel == "" && req.Algorithm != nil && req.Algorithm.Fusion != nil && len(req.Algorithm.Fusion.AnalysisModels) > 0 {
		targetModel = strings.TrimSpace(req.Algorithm.Fusion.AnalysisModels[0])
	}
	return ClassifyStage(req, "synthesis", StageRoleSynthesis, targetModel)
}

func resolveReMoMFinalStage(req *Request) StageOwnership {
	cfg := getDefaultReMoMConfig()
	if req.Algorithm != nil && req.Algorithm.ReMoM != nil {
		cfg = req.Algorithm.ReMoM
	}
	targetModel := strings.TrimSpace(cfg.SynthesisModel)
	if targetModel == "" {
		looper := &ReMoMLooper{}
		defaultCalls := looper.distributeCallsToModels(cfg, 1, req.ModelRefs)
		calls := remomFinalRoundModelCalls(cfg, defaultCalls, req.ModelRefs)
		if len(calls) > 0 {
			targetModel = calls[0].Model
			if calls[0].LoRAName != "" {
				targetModel = calls[0].LoRAName
			}
		} else if len(req.ModelRefs) > 0 {
			targetModel = req.ModelRefs[0].Model
		}
	}
	return ClassifyStage(req, "synthesis", StageRoleSynthesis, targetModel)
}

func resolveWorkflowsFinalStage(req *Request) StageOwnership {
	cfg := resolveWorkflowsExecutionConfig(req)
	targetModel := strings.TrimSpace(cfg.Final.Model)
	if targetModel == "" && strings.TrimSpace(cfg.PlannerModel) != "" {
		targetModel = strings.TrimSpace(cfg.PlannerModel)
	}
	if targetModel == "" && len(req.ModelRefs) > 0 {
		targetModel = req.ModelRefs[0].Model
		if req.ModelRefs[0].LoRAName != "" {
			targetModel = req.ModelRefs[0].LoRAName
		}
	}
	return ClassifyStage(req, "synthesis", StageRoleSynthesis, targetModel)
}

func resolveConfidenceFinalStage(req *Request) StageOwnership {
	// Confidence routes by evaluating confidence/margins across candidates.
	// The final selected candidate is user-visible, but while candidate evaluation is underway
	// the winning model is pending and cannot be determined before emission.
	// Therefore, selection is pending and native streaming is ineligible.
	return StageOwnership{
		AlgorithmType:      config.DecisionAlgorithmConfidence,
		StageName:          "candidate_selection",
		StageRole:          StageRoleDirect,
		TargetModel:        "",
		IsFinalUserVisible: true,
		Eligibility:        StreamingIneligibleBufferingRequired,
		BufferingReason:    BufferingReasonSelectionPending,
	}
}

func resolveRatingsFinalStage(req *Request) StageOwnership {
	// Ratings executes multiple models concurrently to produce side-by-side comparison choices.
	// The composite multi-choice response is user-visible, but requires buffering because
	// standard streaming deltas cannot multiplex multiple choices simultaneously.
	return StageOwnership{
		AlgorithmType:      config.DecisionAlgorithmRatings,
		StageName:          "ratings_deliberation",
		StageRole:          StageRoleSynthesis,
		TargetModel:        "",
		IsFinalUserVisible: true,
		Eligibility:        StreamingIneligibleBufferingRequired,
		BufferingReason:    BufferingReasonMultiChoiceOutput,
	}
}

func resolveBaseFinalStage(req *Request) StageOwnership {
	algoType := AlgorithmTypeBase
	if req.Algorithm != nil && req.Algorithm.Type != "" {
		algoType = req.Algorithm.Type
	}
	if len(req.ModelRefs) == 1 {
		targetModel := req.ModelRefs[0].Model
		if req.ModelRefs[0].LoRAName != "" {
			targetModel = req.ModelRefs[0].LoRAName
		}
		return ClassifyStage(req, "direct", StageRoleDirect, targetModel)
	}

	// Multiple sequential models in BaseLooper are aggregated into a composite body.
	return StageOwnership{
		AlgorithmType:      algoType,
		StageName:          "aggregation",
		StageRole:          StageRoleSynthesis,
		TargetModel:        "",
		IsFinalUserVisible: true,
		Eligibility:        StreamingIneligibleBufferingRequired,
		BufferingReason:    BufferingReasonMultiModelAggregation,
	}
}

func algorithmTypeFromRequest(req *Request) string {
	if req == nil || req.Algorithm == nil || strings.TrimSpace(req.Algorithm.Type) == "" {
		return AlgorithmTypeBase
	}
	return strings.TrimSpace(req.Algorithm.Type)
}
