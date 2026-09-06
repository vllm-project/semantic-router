package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
)

// EvaluationMethodContractVersion is intentionally independent of the v1 run
// bundle contract.  No v1 method record is silently promoted into this model.
const (
	EvaluationMethodContractVersion = "evaluation-method.v2"
	R2CompoundModelBudgetMethodID   = "r2.compound-model-budget.v2"
)

type ActionRef struct {
	SchemaVersion string `json:"schema_version"`
	ID            string `json:"id"`
}

type SliceRef struct {
	SchemaVersion string `json:"schema_version"`
	ID            string `json:"id"`
}

type AnalysisPlan struct {
	SchemaVersion string     `json:"schema_version"`
	ID            string     `json:"id"`
	AnalysisUnit  string     `json:"analysis_unit"`
	ClusterUnit   string     `json:"cluster_unit"`
	Slices        []SliceRef `json:"slices"`
	CurveDomain   string     `json:"curve_domain"`
	Missingness   string     `json:"missingness"`
}

// EvaluationMethodDefinition makes the execution and evidentiary limits of an
// advertised method explicit.  A live method is only cataloguable when both
// its input and its grading are complete.
type EvaluationMethodDefinition struct {
	SchemaVersion       string        `json:"schema_version"`
	ID                  string        `json:"id"`
	Version             string        `json:"version"`
	Status              string        `json:"status"`
	ExecutionOwner      string        `json:"execution_owner"`
	InputSchema         string        `json:"input_schema"`
	ExportSchema        string        `json:"export_schema"`
	LiveInputComplete   bool          `json:"live_input_complete"`
	LiveGrader          bool          `json:"live_grader"`
	ApplicableTracks    []TrackID     `json:"applicable_tracks"`
	LiveTracks          []TrackID     `json:"live_tracks"`
	ProducedMetricIDs   []string      `json:"produced_metric_ids"`
	EvidenceCeiling     EvidenceLevel `json:"evidence_ceiling"`
	NativeParity        string        `json:"native_parity"`
	RequiredArtifactIDs []string      `json:"required_artifact_ids"`
	AnalysisPlan        AnalysisPlan  `json:"analysis_plan"`
}

var evaluationMethodRequiredFields = [...]string{
	"schema_version",
	"id",
	"version",
	"status",
	"execution_owner",
	"input_schema",
	"export_schema",
	"live_input_complete",
	"live_grader",
	"applicable_tracks",
	"live_tracks",
	"produced_metric_ids",
	"evidence_ceiling",
	"native_parity",
	"required_artifact_ids",
	"analysis_plan",
}

// UnmarshalJSON is the production wire boundary for method declarations. Go
// value fields otherwise collapse an omitted or null boolean/list into the same
// zero value as an explicit false/empty declaration. Preserve typed registry
// construction while making every JSON declaration prove the complete v2
// contract before it can enter a report or admission path.
func (method *EvaluationMethodDefinition) UnmarshalJSON(data []byte) error {
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return fmt.Errorf("decode evaluation method: %w", err)
	}
	var fields map[string]json.RawMessage
	fieldDecoder := json.NewDecoder(bytes.NewReader(data))
	if err := fieldDecoder.Decode(&fields); err != nil {
		return fmt.Errorf("decode evaluation method fields: %w", err)
	}
	if err := ensureJSONEOF(fieldDecoder); err != nil {
		return err
	}
	for _, name := range evaluationMethodRequiredFields {
		value, present := fields[name]
		if !present {
			return fmt.Errorf("evaluation method field %q is required", name)
		}
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("evaluation method field %q cannot be null", name)
		}
	}

	type methodDefinitionWire EvaluationMethodDefinition
	var decoded methodDefinitionWire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return fmt.Errorf("decode evaluation method: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return err
	}
	candidate := EvaluationMethodDefinition(decoded)
	if err := ValidateEvaluationMethodDefinition(candidate); err != nil {
		return err
	}
	*method = candidate
	return nil
}

func validateActionRef(ref ActionRef) error {
	if ref.SchemaVersion != EvaluationMethodContractVersion || !validMethodID(ref.ID) {
		return fmt.Errorf("action ref is invalid")
	}
	return nil
}

func validateSliceRef(ref SliceRef) error {
	if ref.SchemaVersion != EvaluationMethodContractVersion || !validMethodID(ref.ID) {
		return fmt.Errorf("slice ref is invalid")
	}
	return nil
}

func validateAnalysisPlan(plan AnalysisPlan) error {
	if plan.SchemaVersion != EvaluationMethodContractVersion || !validMethodID(plan.ID) ||
		strings.TrimSpace(plan.AnalysisUnit) == "" || strings.TrimSpace(plan.ClusterUnit) == "" ||
		(plan.CurveDomain != "shared_budget" && plan.CurveDomain != "not_applicable") ||
		plan.Missingness != "fail_closed" || len(plan.Slices) == 0 {
		return fmt.Errorf("analysis plan is invalid")
	}
	seen := make(map[string]struct{}, len(plan.Slices))
	for _, slice := range plan.Slices {
		if err := validateSliceRef(slice); err != nil {
			return err
		}
		if _, duplicate := seen[slice.ID]; duplicate {
			return fmt.Errorf("analysis plan slices must be unique")
		}
		seen[slice.ID] = struct{}{}
	}
	return nil
}

func validEvidenceCeiling(value EvidenceLevel) bool {
	for _, ceiling := range []EvidenceLevel{"E0", "E1", "E2", "E3", "E4", "E5"} {
		if value == ceiling {
			return true
		}
	}
	return false
}

func validMethodTrackID(value TrackID) bool {
	for _, track := range allTrackIDs {
		if value == track {
			return true
		}
	}
	return false
}

// ValidateEvaluationMethodDefinition is the catalog/planner admission boundary.
func ValidateEvaluationMethodDefinition(method EvaluationMethodDefinition) error {
	owners := map[string]bool{"server": true, "worker": true, "provider": true, "benchmark_native": true}
	parities := map[string]bool{"native": true, "source_qualified": true, "none": true}
	statuses := map[string]bool{"native-qualified": true, "exploratory-import": true, "data-required": true, "blocked": true}
	if method.SchemaVersion != EvaluationMethodContractVersion || method.Version != EvaluationMethodContractVersion ||
		!validMethodID(method.ID) || !statuses[method.Status] || !owners[method.ExecutionOwner] || !validMethodID(method.InputSchema) ||
		!validMethodID(method.ExportSchema) ||
		!validEvidenceCeiling(method.EvidenceCeiling) || !parities[method.NativeParity] {
		return fmt.Errorf("evaluation method definition is invalid")
	}
	if method.NativeParity == "native" && method.ExecutionOwner != "benchmark_native" {
		return fmt.Errorf("native method parity requires benchmark-native execution")
	}
	if len(method.ApplicableTracks) == 0 || len(method.ProducedMetricIDs) == 0 {
		return fmt.Errorf("evaluation method definition needs produced metrics")
	}
	if len(method.RequiredArtifactIDs) == 0 {
		return fmt.Errorf("evaluation method definition needs required artifacts")
	}
	if method.Status == "native-qualified" && (!method.LiveInputComplete || !method.LiveGrader || len(method.LiveTracks) == 0) {
		return fmt.Errorf("native-qualified method needs complete graded live execution")
	}
	if method.Status != "native-qualified" && (method.LiveInputComplete || method.LiveGrader) {
		return fmt.Errorf("non-qualified method cannot claim complete live execution")
	}
	seenApplicableTracks := make(map[TrackID]struct{}, len(method.ApplicableTracks))
	for _, track := range method.ApplicableTracks {
		if !validMethodTrackID(track) {
			return fmt.Errorf("evaluation method applicable track is invalid")
		}
		if _, duplicate := seenApplicableTracks[track]; duplicate {
			return fmt.Errorf("evaluation method applicable tracks must be unique")
		}
		seenApplicableTracks[track] = struct{}{}
	}
	seenTracks := make(map[TrackID]struct{}, len(method.LiveTracks))
	for _, track := range method.LiveTracks {
		if !validMethodTrackID(track) {
			return fmt.Errorf("evaluation method track is invalid")
		}
		if _, duplicate := seenTracks[track]; duplicate {
			return fmt.Errorf("evaluation method tracks must be unique")
		}
		if _, applicable := seenApplicableTracks[track]; !applicable {
			return fmt.Errorf("evaluation method live track must be applicable")
		}
		seenTracks[track] = struct{}{}
	}
	seenMetrics := make(map[string]struct{}, len(method.ProducedMetricIDs))
	for _, metric := range method.ProducedMetricIDs {
		if metric == "" || metric != strings.TrimSpace(metric) {
			return fmt.Errorf("evaluation method metric id is invalid")
		}
		if _, duplicate := seenMetrics[metric]; duplicate {
			return fmt.Errorf("evaluation method metric ids must be unique")
		}
		seenMetrics[metric] = struct{}{}
	}
	seenArtifacts := make(map[string]struct{}, len(method.RequiredArtifactIDs))
	for _, artifact := range method.RequiredArtifactIDs {
		if !validMethodID(artifact) {
			return fmt.Errorf("evaluation method artifact id is invalid")
		}
		if _, duplicate := seenArtifacts[artifact]; duplicate {
			return fmt.Errorf("evaluation method artifact ids must be unique")
		}
		seenArtifacts[artifact] = struct{}{}
	}
	return validateAnalysisPlan(method.AnalysisPlan)
}

type CompoundModelBudgetOutcome struct {
	CaseID    string     `json:"case_id"`
	Action    ActionRef  `json:"action"`
	Budget    int        `json:"budget"`
	Score     float64    `json:"score"`
	SliceRefs []SliceRef `json:"slice_refs"`
}

type SharedDomainCurvePoint struct {
	Action    ActionRef `json:"action"`
	Budget    int       `json:"budget"`
	MeanScore float64   `json:"mean_score"`
	CaseCount int       `json:"case_count"`
}

type CompoundModelBudgetReport struct {
	Method                       EvaluationMethodDefinition `json:"method"`
	AnalysisPlan                 AnalysisPlan               `json:"analysis_plan"`
	ActionRefs                   []ActionRef                `json:"action_refs"`
	SliceRefs                    []SliceRef                 `json:"slice_refs"`
	RawSharedDomainCurve         []SharedDomainCurvePoint   `json:"raw_shared_domain_curve"`
	AUDC                         float64                    `json:"audc"`
	NAUC                         float64                    `json:"nauc"`
	Peak                         float64                    `json:"peak"`
	QNC                          float64                    `json:"qnc"`
	MissingCaseActionBudgetCells int                        `json:"missing_case_action_budget_cells"`
}

func R2CompoundModelBudgetMethod() EvaluationMethodDefinition {
	benchmark, found := researchBenchmarkByAdapter("r2-router")
	if !found {
		panic("research benchmark inventory is missing r2-router")
	}
	return researchBenchmarkMethod(benchmark)
}

// ReduceCompoundModelBudget refuses a ragged cohort.  This preserves both the
// supplied ActionRef identity and a truly shared case×action×budget domain.
func ReduceCompoundModelBudget(method EvaluationMethodDefinition, outcomes []CompoundModelBudgetOutcome) (CompoundModelBudgetReport, error) {
	if method.ID != R2CompoundModelBudgetMethodID {
		return CompoundModelBudgetReport{}, fmt.Errorf("compound reducer requires the R2 compound method")
	}
	if err := ValidateEvaluationMethodDefinition(method); err != nil {
		return CompoundModelBudgetReport{}, err
	}
	if len(outcomes) == 0 {
		return CompoundModelBudgetReport{}, fmt.Errorf("compound reducer requires outcomes")
	}
	cases, actions, budgets, slices := make(map[string]struct{}), make(map[string]struct{}), make(map[int]struct{}), make(map[string]struct{})
	cells := make(map[string]float64)
	expectedSlices := make(map[string]struct{}, len(method.AnalysisPlan.Slices))
	for _, slice := range method.AnalysisPlan.Slices {
		expectedSlices[slice.ID] = struct{}{}
	}
	for _, outcome := range outcomes {
		if !validMethodID(outcome.CaseID) || validateActionRef(outcome.Action) != nil || outcome.Budget <= 0 ||
			math.IsNaN(outcome.Score) || math.IsInf(outcome.Score, 0) || outcome.Score < 0 || outcome.Score > 1 || len(outcome.SliceRefs) == 0 {
			return CompoundModelBudgetReport{}, fmt.Errorf("compound outcome is invalid")
		}
		localSlices := make(map[string]struct{}, len(outcome.SliceRefs))
		for _, slice := range outcome.SliceRefs {
			if err := validateSliceRef(slice); err != nil {
				return CompoundModelBudgetReport{}, err
			}
			if _, duplicate := localSlices[slice.ID]; duplicate {
				return CompoundModelBudgetReport{}, fmt.Errorf("compound outcome slices must be unique")
			}
			localSlices[slice.ID] = struct{}{}
			slices[slice.ID] = struct{}{}
		}
		if len(localSlices) != len(expectedSlices) {
			return CompoundModelBudgetReport{}, fmt.Errorf("compound outcome slices must exactly match the analysis plan")
		}
		for sliceID := range expectedSlices {
			if _, present := localSlices[sliceID]; !present {
				return CompoundModelBudgetReport{}, fmt.Errorf("compound outcome slices must exactly match the analysis plan")
			}
		}
		key := outcome.CaseID + "\x00" + outcome.Action.ID + "\x00" + fmt.Sprint(outcome.Budget)
		if _, duplicate := cells[key]; duplicate {
			return CompoundModelBudgetReport{}, fmt.Errorf("duplicate case×action×budget outcome: %s×%s×%d", outcome.CaseID, outcome.Action.ID, outcome.Budget)
		}
		cells[key] = outcome.Score
		cases[outcome.CaseID], actions[outcome.Action.ID], budgets[outcome.Budget] = struct{}{}, struct{}{}, struct{}{}
	}
	caseIDs, actionIDs, budgetValues, sliceIDs := methodV2SortedStrings(cases), methodV2SortedStrings(actions), methodV2SortedInts(budgets), methodV2SortedStrings(slices)
	if len(cells) != len(caseIDs)*len(actionIDs)*len(budgetValues) {
		return CompoundModelBudgetReport{}, fmt.Errorf("compound model+budget outcomes must form an exact shared case×action×budget domain")
	}
	if len(budgetValues) < 2 {
		return CompoundModelBudgetReport{}, fmt.Errorf("compound AUDC requires at least two shared budget points")
	}
	curve := make([]SharedDomainCurvePoint, 0, len(actionIDs)*len(budgetValues))
	means := make(map[string]float64)
	for _, actionID := range actionIDs {
		for _, budget := range budgetValues {
			total := 0.0
			for _, caseID := range caseIDs {
				total += cells[caseID+"\x00"+actionID+"\x00"+fmt.Sprint(budget)]
			}
			mean := total / float64(len(caseIDs))
			means[actionID+"\x00"+fmt.Sprint(budget)] = mean
			curve = append(curve, SharedDomainCurvePoint{Action: ActionRef{SchemaVersion: EvaluationMethodContractVersion, ID: actionID}, Budget: budget, MeanScore: mean, CaseCount: len(caseIDs)})
		}
	}
	audc, peak, qnc := 0.0, 0.0, 0.0
	for _, actionID := range actionIDs {
		for index := 0; index < len(budgetValues)-1; index++ {
			lower, upper := budgetValues[index], budgetValues[index+1]
			audc += float64(upper-lower) * (means[actionID+"\x00"+fmt.Sprint(lower)] + means[actionID+"\x00"+fmt.Sprint(upper)]) / 2
		}
		terminal := means[actionID+"\x00"+fmt.Sprint(budgetValues[len(budgetValues)-1])]
		qnc += terminal
	}
	for _, mean := range means {
		if mean > peak {
			peak = mean
		}
	}
	nauc := audc / (float64(len(actionIDs)) * float64(budgetValues[len(budgetValues)-1]-budgetValues[0]))
	qnc /= float64(len(actionIDs))
	actionRefs, sliceRefs := make([]ActionRef, len(actionIDs)), make([]SliceRef, len(sliceIDs))
	for index, id := range actionIDs {
		actionRefs[index] = ActionRef{SchemaVersion: EvaluationMethodContractVersion, ID: id}
	}
	for index, id := range sliceIDs {
		sliceRefs[index] = SliceRef{SchemaVersion: EvaluationMethodContractVersion, ID: id}
	}
	return CompoundModelBudgetReport{Method: method, AnalysisPlan: method.AnalysisPlan, ActionRefs: actionRefs, SliceRefs: sliceRefs, RawSharedDomainCurve: curve, AUDC: audc, NAUC: nauc, Peak: peak, QNC: qnc}, nil
}

// ReduceSealedMethodReports is the only publication path for v2 method
// reports.  It deliberately starts from validated raw records, not a worker
// aggregate, so curves and summary metrics cannot be forged independently.
func ReduceSealedMethodReports(methods methodRecordAttestation) ([]CompoundModelBudgetReport, error) {
	reports := make([]CompoundModelBudgetReport, 0, 1)
	if len(methods.R2Outcomes) == 0 {
		return reports, nil
	}
	report, err := ReduceCompoundModelBudget(R2CompoundModelBudgetMethod(), methods.R2Outcomes)
	if err != nil {
		return nil, err
	}
	return append(reports, report), nil
}

func validateSealedMethodReports(actual []CompoundModelBudgetReport, methods methodRecordAttestation) error {
	if actual == nil {
		return fmt.Errorf("method_reports cannot be null")
	}
	expected, err := ReduceSealedMethodReports(methods)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(actual, expected) {
		return fmt.Errorf("method_reports do not match server-reduced raw method coordinates")
	}
	return nil
}

func methodV2SortedStrings(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func methodV2SortedInts(values map[int]struct{}) []int {
	result := make([]int, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Ints(result)
	return result
}
