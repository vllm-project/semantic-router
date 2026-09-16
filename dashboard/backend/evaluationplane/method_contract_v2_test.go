package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func r2Outcome(caseID, actionID string, budget int, score float64) CompoundModelBudgetOutcome {
	return CompoundModelBudgetOutcome{
		CaseID: caseID, Action: ActionRef{SchemaVersion: EvaluationMethodContractVersion, ID: actionID}, Budget: budget, Score: score,
		SliceRefs: []SliceRef{{SchemaVersion: EvaluationMethodContractVersion, ID: "all"}},
	}
}

func TestR2CompoundModelBudgetPreservesActionIdentityAndSharedCurve(t *testing.T) {
	method := R2CompoundModelBudgetMethod()
	if err := ValidateEvaluationMethodDefinition(method); err != nil {
		t.Fatalf("method should be admissible: %v", err)
	}
	report, err := ReduceCompoundModelBudget(method, []CompoundModelBudgetOutcome{
		r2Outcome("case-a", "small", 100, .4), r2Outcome("case-a", "small", 200, .6),
		r2Outcome("case-a", "large", 100, .6), r2Outcome("case-a", "large", 200, .8),
		r2Outcome("case-b", "small", 100, .2), r2Outcome("case-b", "small", 200, .4),
		r2Outcome("case-b", "large", 100, .4), r2Outcome("case-b", "large", 200, .6),
	})
	if err != nil {
		t.Fatalf("reduce compound model+budget: %v", err)
	}
	if got := []string{report.ActionRefs[0].ID, report.ActionRefs[1].ID}; strings.Join(got, ",") != "large,small" {
		t.Fatalf("action identities changed: %v", got)
	}
	if len(report.RawSharedDomainCurve) != 4 || report.RawSharedDomainCurve[0].Action.ID != "large" || report.RawSharedDomainCurve[0].Budget != 100 {
		t.Fatalf("raw shared curve is malformed: %#v", report.RawSharedDomainCurve)
	}
	for name, values := range map[string]struct{ got, want float64 }{
		"AUDC": {report.AUDC, 100}, "nAUC": {report.NAUC, .5}, "Peak": {report.Peak, .7}, "QNC": {report.QNC, .6},
	} {
		if math.Abs(values.got-values.want) > 1e-12 {
			t.Fatalf("%s=%v, want %v", name, values.got, values.want)
		}
	}
}

func TestR2ReducersFailClosedOnDuplicateAndRaggedDomains(t *testing.T) {
	method := R2CompoundModelBudgetMethod()
	duplicate := r2Outcome("case-a", "small", 100, .5)
	if _, err := ReduceCompoundModelBudget(method, []CompoundModelBudgetOutcome{duplicate, duplicate}); err == nil || !strings.Contains(err.Error(), "duplicate case×action×budget") {
		t.Fatalf("compound reducer accepted duplicate case×action×budget: %v", err)
	}
	if _, err := ReduceCompoundModelBudget(method, []CompoundModelBudgetOutcome{
		r2Outcome("case-a", "small", 100, .4), r2Outcome("case-a", "small", 200, .5),
		r2Outcome("case-a", "large", 100, .6),
	}); err == nil || !strings.Contains(err.Error(), "exact shared") {
		t.Fatalf("compound reducer accepted ragged shared domain: %v", err)
	}
}

func TestSealedR2ReportIsRecomputedFromRawCoordinates(t *testing.T) {
	methods := methodRecordAttestation{R2Outcomes: []CompoundModelBudgetOutcome{
		r2Outcome("case-a", "small", 100, .4), r2Outcome("case-a", "small", 200, .6),
		r2Outcome("case-b", "small", 100, .2), r2Outcome("case-b", "small", 200, .4),
	}}
	reports, err := ReduceSealedMethodReports(methods)
	if err != nil || len(reports) != 1 {
		t.Fatalf("server reduction failed: reports=%#v err=%v", reports, err)
	}
	if err := validateSealedMethodReports(reports, methods); err != nil {
		t.Fatalf("server-reduced report was rejected: %v", err)
	}
	forged := append([]CompoundModelBudgetReport(nil), reports...)
	forged[0].AUDC += 0.01
	if err := validateSealedMethodReports(forged, methods); err == nil || !strings.Contains(err.Error(), "do not match") {
		t.Fatalf("forged R2 aggregate was accepted: %v", err)
	}
}

func TestR2CoordinatesRequireTheCompleteIdentityAndRemainDistinctPerBudget(t *testing.T) {
	methodID, actionID := R2CompoundModelBudgetMethodID, "small"
	budget, quality := 100, .5
	record := executionRecordEvidence{
		MethodID: &methodID, ActionID: &actionID, BudgetTokens: &budget, Quality: &quality,
		SliceIDs: []string{"all"}, TrackID: "model_pool", Status: "succeeded",
	}
	if err := validateV2MethodCoordinates(record); err != nil {
		t.Fatalf("complete R2 coordinates rejected: %v", err)
	}
	secondBudget := 200
	second := record
	second.BudgetTokens = &secondBudget
	if record.semanticKey() == second.semanticKey() {
		t.Fatal("R2 semantic key collapsed distinct budget cells")
	}
	record.ActionID = nil
	if err := validateV2MethodCoordinates(record); err == nil || !strings.Contains(err.Error(), "require") {
		t.Fatalf("incomplete R2 coordinates accepted: %v", err)
	}
}

func TestEmptyMethodSlicesDoNotTurnOrdinaryRecordsIntoV2Coordinates(t *testing.T) {
	record := executionRecordEvidence{TrackID: "routing", Status: "succeeded", SliceIDs: []string{}}
	if err := validateV2MethodCoordinates(record); err != nil {
		t.Fatalf("ordinary record with no method slices rejected: %v", err)
	}
	record.SliceIDs = []string{"all"}
	if err := validateV2MethodCoordinates(record); err == nil || !strings.Contains(err.Error(), "require method_id") {
		t.Fatalf("unbound non-empty method slices accepted: %v", err)
	}
}

func TestResearchMethodDefinitionsDeclareAllReadinessBoundaries(t *testing.T) {
	exploratory, blocked, dataRequired := 0, 0, 0
	for _, benchmark := range ResearchBenchmarkInventory() {
		method := researchBenchmarkMethod(benchmark)
		if ValidateEvaluationMethodDefinition(method) != nil {
			t.Fatalf("adapter %q lacks a valid v2 method: %#v", benchmark.AdapterID, method)
		}
		if len(method.ApplicableTracks) == 0 || len(method.RequiredArtifactIDs) == 0 || len(method.ProducedMetricIDs) == 0 {
			t.Fatalf("adapter %q lacks explicit applicability, artifacts, or metrics", benchmark.AdapterID)
		}
		switch method.Status {
		case "exploratory-import":
			exploratory++
		case "blocked":
			blocked++
		case "data-required":
			dataRequired++
		}
	}
	if exploratory != 8 || blocked != 2 || dataRequired != 3 {
		t.Fatalf("v2 research inventory exploratory=%d blocked=%d data-required=%d", exploratory, blocked, dataRequired)
	}
}

func decodeMethodFixtureJSONPointer(pointer string) ([]string, error) {
	if pointer == "" || pointer[0] != '/' {
		return nil, fmt.Errorf("JSON Pointer must identify a descriptor field: %q", pointer)
	}
	encodedTokens := strings.Split(pointer[1:], "/")
	tokens := make([]string, 0, len(encodedTokens))
	for _, encodedToken := range encodedTokens {
		for index := 0; index < len(encodedToken); index++ {
			if encodedToken[index] != '~' {
				continue
			}
			if index+1 >= len(encodedToken) || (encodedToken[index+1] != '0' && encodedToken[index+1] != '1') {
				return nil, fmt.Errorf("JSON Pointer has an invalid escape: %q", pointer)
			}
			index++
		}
		token := strings.ReplaceAll(encodedToken, "~1", "/")
		tokens = append(tokens, strings.ReplaceAll(token, "~0", "~"))
	}
	return tokens, nil
}

func methodFixtureJSONArrayIndex(token string, length int, pointer string) (int, error) {
	if token == "" || (len(token) > 1 && token[0] == '0') {
		return 0, fmt.Errorf("JSON Pointer has an invalid array index: %q", pointer)
	}
	for _, character := range token {
		if character < '0' || character > '9' {
			return 0, fmt.Errorf("JSON Pointer has an invalid array index: %q", pointer)
		}
	}
	index, err := strconv.Atoi(token)
	if err != nil || index >= length {
		return 0, fmt.Errorf("JSON Pointer array index is out of bounds: %q", pointer)
	}
	return index, nil
}

func removeMethodFixtureJSONPointerValue(value any, tokens []string, pointer string) (any, error) {
	token := tokens[0]
	switch typed := value.(type) {
	case map[string]any:
		child, exists := typed[token]
		if !exists {
			return nil, fmt.Errorf("JSON Pointer field does not exist: %q", pointer)
		}
		if len(tokens) == 1 {
			delete(typed, token)
			return typed, nil
		}
		updated, err := removeMethodFixtureJSONPointerValue(child, tokens[1:], pointer)
		if err != nil {
			return nil, err
		}
		typed[token] = updated
		return typed, nil
	case []any:
		index, err := methodFixtureJSONArrayIndex(token, len(typed), pointer)
		if err != nil {
			return nil, err
		}
		if len(tokens) == 1 {
			return append(typed[:index], typed[index+1:]...), nil
		}
		updated, err := removeMethodFixtureJSONPointerValue(typed[index], tokens[1:], pointer)
		if err != nil {
			return nil, err
		}
		typed[index] = updated
		return typed, nil
	default:
		return nil, fmt.Errorf("JSON Pointer traverses a scalar: %q", pointer)
	}
}

func removeMethodFixtureJSONPointer(document map[string]any, pointer string) error {
	tokens, err := decodeMethodFixtureJSONPointer(pointer)
	if err != nil {
		return err
	}
	_, err = removeMethodFixtureJSONPointerValue(document, tokens, pointer)
	return err
}

func TestMethodV2AdmissionMatchesSharedCrossLanguageConformance(t *testing.T) {
	type conformanceCase struct {
		ID            string          `json:"id"`
		ExpectedValid *bool           `json:"expected_valid"`
		RemoveFields  *[]string       `json:"remove_fields"`
		Overrides     *map[string]any `json:"overrides"`
	}
	type conformanceCorpus struct {
		SchemaVersion         string            `json:"schema_version"`
		MethodContractVersion string            `json:"method_contract_version"`
		BaseDescriptor        map[string]any    `json:"base_descriptor"`
		Cases                 []conformanceCase `json:"cases"`
	}

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve method conformance test location")
	}
	fixturePath := filepath.Join(
		filepath.Dir(currentFile),
		"../../../src/vllm-sr/tests/fixtures/evaluation_method_contract_v2_conformance.v1.json",
	)
	payload, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read shared method conformance corpus: %v", err)
	}
	var corpus conformanceCorpus
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&corpus); err != nil {
		t.Fatalf("decode shared method conformance corpus: %v", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		t.Fatalf("shared method conformance corpus trailing data: %v", err)
	}
	if corpus.SchemaVersion != "evaluation-method-conformance.v1" ||
		corpus.MethodContractVersion != EvaluationMethodContractVersion || len(corpus.BaseDescriptor) == 0 || len(corpus.Cases) == 0 {
		t.Fatalf("shared method conformance corpus has an invalid envelope: %#v", corpus)
	}

	seenCaseIDs := make(map[string]struct{}, len(corpus.Cases))
	for _, testCase := range corpus.Cases {
		t.Run(testCase.ID, func(t *testing.T) {
			if testCase.ID == "" || testCase.ExpectedValid == nil || testCase.RemoveFields == nil || testCase.Overrides == nil {
				t.Fatal("method conformance case is missing a required field")
			}
			if _, duplicate := seenCaseIDs[testCase.ID]; duplicate {
				t.Fatalf("duplicate method conformance case id %q", testCase.ID)
			}
			seenCaseIDs[testCase.ID] = struct{}{}

			basePayload, marshalErr := json.Marshal(corpus.BaseDescriptor)
			if marshalErr != nil {
				t.Fatalf("clone method descriptor: %v", marshalErr)
			}
			var descriptorFields map[string]any
			if err := json.Unmarshal(basePayload, &descriptorFields); err != nil {
				t.Fatalf("clone method descriptor: %v", err)
			}
			for _, pointer := range *testCase.RemoveFields {
				if err := removeMethodFixtureJSONPointer(descriptorFields, pointer); err != nil {
					t.Fatalf("remove method descriptor field: %v", err)
				}
			}
			for field, value := range *testCase.Overrides {
				descriptorFields[field] = value
			}
			descriptorPayload, marshalErr := json.Marshal(descriptorFields)
			if marshalErr != nil {
				t.Fatalf("encode method descriptor: %v", marshalErr)
			}
			var descriptor EvaluationMethodDefinition
			descriptorDecoder := json.NewDecoder(bytes.NewReader(descriptorPayload))
			descriptorDecoder.DisallowUnknownFields()
			decodeErr := descriptorDecoder.Decode(&descriptor)
			if decodeErr == nil {
				decodeErr = ensureJSONEOF(descriptorDecoder)
			}
			accepted := decodeErr == nil && ValidateEvaluationMethodDefinition(descriptor) == nil
			if accepted != *testCase.ExpectedValid {
				t.Fatalf("method descriptor accepted=%t, want %t (decode error: %v)", accepted, *testCase.ExpectedValid, decodeErr)
			}
		})
	}
}
