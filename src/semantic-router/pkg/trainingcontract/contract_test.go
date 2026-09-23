package trainingcontract

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func fixture(t *testing.T, name string) Fixture {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	var f Fixture
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSharedFixtures(t *testing.T) {
	for _, name := range []string{"selector", "neural"} {
		t.Run(name, func(t *testing.T) {
			f := fixture(t, name)
			if err := ValidateRun(f.Submit); err != nil {
				t.Fatal(err)
			}
			if err := ValidateProfile(f.Snapshot.Profile); err != nil {
				t.Fatal(err)
			}
			if err := ValidateModel(*f.Snapshot.Source); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(f.Submit.Spec, f.Graph.Run.Spec) {
				t.Fatal("run spec changed after submission")
			}
			if !reflect.DeepEqual(f.Snapshot, f.WorkerRequest.Snapshot) {
				t.Fatal("worker received a different snapshot")
			}
			if !reflect.DeepEqual(f.Artifact.Profile, f.Snapshot.Profile) {
				t.Fatal("artifact profile changed")
			}
			if f.Artifact.Provenance.AttemptID != f.WorkerRequest.AttemptID {
				t.Fatal("artifact lost attempt identity")
			}
			if !reflect.DeepEqual(f.Variant.ArtifactVariantSpec, f.TrainResult.Artifacts[0].Variants[0]) {
				t.Fatal("published variant differs from worker output")
			}
			if err := ValidateVariant(f.Variant.ArtifactVariantSpec); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(f.Graph.Outputs, RunOutputs{
				ArtifactIDs: []string{f.Artifact.ID}, EvaluationIDs: []string{f.Evaluation.ID},
				QualificationIDs: []string{f.Qualification.ID},
			}) {
				t.Fatal("run does not index its published outputs")
			}
			if f.Variant.ArtifactID != f.Artifact.ID {
				t.Fatal("variant lost artifact identity")
			}
			if f.EvaluateRequest.Inputs[0].ID != f.Variant.ID || f.EvaluateResult.Evaluations[0].VariantID != f.Variant.ID {
				t.Fatal("evaluation must reuse the published variant")
			}
			if len(f.EvaluateResult.Artifacts) != 0 {
				t.Fatal("evaluation must not recreate the training artifact")
			}
			if f.Qualification.VariantID != f.Proposal.VariantID || f.Proposal.QualificationID != f.Qualification.ID {
				t.Fatal("binding proposal lost qualification evidence")
			}
			encoded, err := json.Marshal(f)
			if err != nil {
				t.Fatal(err)
			}
			var roundtrip Fixture
			if err := json.Unmarshal(encoded, &roundtrip); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(f, roundtrip) {
				t.Fatal("Go changed the shared wire representation")
			}
		})
	}
}

func TestRunDependencies(t *testing.T) {
	cases := []struct {
		name   string
		change func(*RunSpec)
	}{
		{"duplicate task", func(s *RunSpec) { s.Tasks[1].Key = s.Tasks[0].Key }},
		{"unknown dependency", func(s *RunSpec) { s.Tasks[1].DependsOn = []string{"missing"} }},
		{"cycle", func(s *RunSpec) { s.Tasks[0].DependsOn = []string{"qualify"} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture(t, "selector")
			tc.change(&f.Submit.Spec)
			if ValidateRun(f.Submit) == nil {
				t.Fatal("accepted invalid task graph")
			}
		})
	}
}

// These structural rejection cases are also checked against JSON Schema by Python.
func TestSharedInvalidInputs(t *testing.T) {
	data, err := os.ReadFile("testdata/invalid.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name       string          `json:"name"`
		Definition string          `json:"definition"`
		Value      json.RawMessage `json:"value"`
	}
	if unmarshalErr := json.Unmarshal(data, &cases); unmarshalErr != nil {
		t.Fatal(unmarshalErr)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			var validationErr error
			switch tc.Definition {
			case "Profile":
				var p Profile
				if unmarshalErr := json.Unmarshal(tc.Value, &p); unmarshalErr != nil {
					t.Fatal(unmarshalErr)
				}
				validationErr = ValidateProfile(p)
			case "ArtifactVariantSpec":
				var spec ArtifactVariantSpec
				if unmarshalErr := json.Unmarshal(tc.Value, &spec); unmarshalErr != nil {
					t.Fatal(unmarshalErr)
				}
				validationErr = ValidateVariant(spec)
			case "SubmitRunRequest":
				var r SubmitRunRequest
				decoder := json.NewDecoder(bytes.NewReader(tc.Value))
				decoder.DisallowUnknownFields()
				validationErr = decoder.Decode(&r)
				if validationErr == nil {
					validationErr = ValidateRun(r)
				}
			default:
				t.Fatalf("unknown fixture definition %s", tc.Definition)
			}
			if validationErr == nil {
				t.Fatal("accepted invalid contract input")
			}
		})
	}
}

func TestRunTransitions(t *testing.T) {
	allowed := map[Status][]Status{
		Pending:    {Running, Cancelling},
		Running:    {Succeeded, Failed, Cancelling},
		Cancelling: {Cancelled},
		Failed:     {Pending},
		Cancelled:  {Pending},
	}
	statuses := []Status{Pending, Running, Cancelling, Succeeded, Failed, Cancelled, Skipped}
	for _, from := range statuses {
		for _, to := range statuses {
			want := false
			for _, candidate := range allowed[from] {
				want = want || candidate == to
			}
			if got := ValidateTransition(from, to) == nil; got != want {
				t.Errorf("%s -> %s: got %v want %v", from, to, got, want)
			}
		}
	}
}

func TestClassifierLabelOrder(t *testing.T) {
	p := fixture(t, "neural").Snapshot.Profile
	p.Classifier.LabelMapping = map[string]int{"negative": 0, "positive": 0}
	if ValidateProfile(p) == nil {
		t.Fatal("accepted ambiguous class order")
	}
}
