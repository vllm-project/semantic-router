package shadowdataset

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func testPolicy() Policy {
	return Policy{
		Seed:   "seed-a",
		Splits: []Split{{Name: "train", Weight: 8}, {Name: "eval", Weight: 2}},
	}
}

func shadowOutcome(model, digest string) store.Outcome {
	return store.Outcome{
		Source:    shadowDispatchSource,
		Target:    "model",
		TargetRef: model,
		Verdict:   shadowVerdictCompleted,
		Metadata: map[string]string{
			"shadow_model":    model,
			"shadow_backend":  "candidate-pool",
			"response_sha256": digest,
			"sample_rate":     "0.1",
		},
	}
}

func comparedRecord(id, request, response string, outcomes ...store.Outcome) store.Record {
	return store.Record{
		ID:            id,
		Timestamp:     time.Date(2026, 9, 18, 9, 0, 0, 0, time.UTC),
		RequestID:     "req-" + id,
		Recipe:        "vault",
		Decision:      "guard",
		SelectedModel: "primary-model",
		RequestBody:   request,
		ResponseBody:  response,
		Outcomes:      outcomes,
	}
}

func buildOrFail(t *testing.T, records []store.Record, policy Policy) Manifest {
	t.Helper()
	manifest, err := Build(records, policy)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return manifest
}

func TestBuildProducesTheSameManifestForTheSameRecords(t *testing.T) {
	records := []store.Record{
		comparedRecord("r1", "ask one", "answer one", shadowOutcome("candidate-a", "d1")),
		comparedRecord("r2", "ask two", "answer two", shadowOutcome("candidate-a", "d2")),
		comparedRecord("r3", "ask three", "answer three", shadowOutcome("candidate-b", "d3")),
	}

	first := buildOrFail(t, records, testPolicy())
	shuffled := []store.Record{records[2], records[0], records[1]}
	second := buildOrFail(t, shuffled, testPolicy())

	if first.Digest != second.Digest {
		t.Fatalf("same records in a different order gave digests %s and %s", first.Digest, second.Digest)
	}
	if len(first.Examples) != 3 {
		t.Fatalf("kept %d examples, want 3", len(first.Examples))
	}
	for i := range first.Examples {
		if first.Examples[i].Split != second.Examples[i].Split {
			t.Fatalf("example %s moved from split %q to %q", first.Examples[i].ID, first.Examples[i].Split, second.Examples[i].Split)
		}
	}
}

func TestBuildKeepsAnExampleInItsSplitWhenMoreRecordsArrive(t *testing.T) {
	first := comparedRecord("r1", "ask one", "answer one", shadowOutcome("candidate-a", "d1"))
	before := buildOrFail(t, []store.Record{first}, testPolicy())

	later := comparedRecord("r2", "ask two", "answer two", shadowOutcome("candidate-a", "d2"))
	after := buildOrFail(t, []store.Record{first, later}, testPolicy())

	placed := map[string]string{}
	for _, example := range after.Examples {
		placed[example.ID] = example.Split
	}
	for _, example := range before.Examples {
		if placed[example.ID] != example.Split {
			t.Fatalf("example %s was in %q and moved to %q", example.ID, example.Split, placed[example.ID])
		}
	}
}

func TestBuildPlacesAnExampleDifferentlyUnderADifferentSeed(t *testing.T) {
	records := make([]store.Record, 0, 40)
	for i := 0; i < 40; i++ {
		id := string(rune('a' + i%26))
		records = append(records, comparedRecord("r"+id+string(rune('0'+i/26)), "ask "+id+string(rune('0'+i/26)), "answer", shadowOutcome("candidate-a", "d")))
	}

	seedA := buildOrFail(t, records, testPolicy())
	other := testPolicy()
	other.Seed = "seed-b"
	seedB := buildOrFail(t, records, other)

	if seedA.Digest == seedB.Digest {
		t.Fatal("two seeds produced the same manifest, so the seed does not reach the split")
	}
}

// TestBuildExcludesEveryPartialObservation is the guarantee that lets an
// Example carry no capture or redaction state: a record that is not whole never
// becomes one. Each case names the reason it is counted under, so a new
// exclusion cannot be added without saying what it means.
func TestBuildExcludesEveryPartialObservation(t *testing.T) {
	damage := map[string]struct {
		reason string
		apply  func(*store.Record)
	}{
		"no shadow arm at all": {ExcludeNoShadowArm, func(r *store.Record) { r.Outcomes = nil }},
		"shadow arm failed": {ExcludeNoShadowArm, func(r *store.Record) {
			r.Outcomes = []store.Outcome{{
				Source:   shadowDispatchSource,
				Verdict:  "failed",
				Metadata: map[string]string{"shadow_model": "candidate-a"},
			}}
		}},
		"request cleared by the redactor":  {ExcludeRequestMissing, func(r *store.Record) { r.RequestBody = "" }},
		"request cut":                      {ExcludeRequestTruncated, func(r *store.Record) { r.RequestBodyTruncated = true }},
		"response cleared by the redactor": {ExcludePrimaryResponseMissing, func(r *store.Record) { r.ResponseBody = "" }},
		"response cut":                     {ExcludeResponseTruncated, func(r *store.Record) { r.ResponseBodyTruncated = true }},
		"primary model unknown":            {ExcludePrimaryModelMissing, func(r *store.Record) { r.SelectedModel = "" }},
	}

	for name, damaged := range damage {
		t.Run(name, func(t *testing.T) {
			record := comparedRecord("r1", "ask one", "answer one", shadowOutcome("candidate-a", "d1"))
			damaged.apply(&record)

			manifest := buildOrFail(t, []store.Record{record}, testPolicy())

			if manifest.Counts.Examples != 0 {
				t.Fatalf("kept %d examples from a partial observation, want 0", manifest.Counts.Examples)
			}
			if manifest.Counts.Excluded[damaged.reason] != 1 {
				t.Fatalf("excluded under %v, want one %s", manifest.Counts.Excluded, damaged.reason)
			}
		})
	}
}

func TestBuildKeepsTheWholeObservationsBesideTheExcludedOnes(t *testing.T) {
	noShadow := comparedRecord("r1", "ask one", "answer one")
	truncated := comparedRecord("r2", "ask two", "answer two", shadowOutcome("candidate-a", "d2"))
	truncated.ResponseBodyTruncated = true
	kept := comparedRecord("r3", "ask three", "answer three", shadowOutcome("candidate-a", "d3"))

	manifest := buildOrFail(t, []store.Record{noShadow, truncated, kept}, testPolicy())

	if manifest.Counts.Records != 3 {
		t.Fatalf("reported %d records, want 3", manifest.Counts.Records)
	}
	if manifest.Counts.Examples != 1 {
		t.Fatalf("kept %d examples, want 1", manifest.Counts.Examples)
	}
	if got := manifest.Examples[0].Lineage.ReplayID; got != "r3" {
		t.Fatalf("kept replay %q, want the whole observation r3", got)
	}
}

func TestBuildKeepsOneExamplePerInput(t *testing.T) {
	early := comparedRecord("r1", "same ask", "answer one", shadowOutcome("candidate-a", "d1"))
	late := comparedRecord("r2", "same ask", "answer two", shadowOutcome("candidate-a", "d2"))
	late.Timestamp = early.Timestamp.Add(time.Hour)

	manifest := buildOrFail(t, []store.Record{late, early}, testPolicy())

	if manifest.Counts.Examples != 1 {
		t.Fatalf("kept %d examples for one input, want 1", manifest.Counts.Examples)
	}
	if manifest.Counts.Excluded[ExcludeDuplicateInput] != 1 {
		t.Fatalf("excluded %d duplicates, want 1", manifest.Counts.Excluded[ExcludeDuplicateInput])
	}
	if got := manifest.Examples[0].Lineage.ReplayID; got != "r1" {
		t.Fatalf("kept replay %q, want the earlier record r1", got)
	}
}

func TestBuildCarriesNoPayloadText(t *testing.T) {
	secret := "carry me into the manifest"
	manifest := buildOrFail(t, []store.Record{
		comparedRecord("r1", secret, secret, shadowOutcome("candidate-a", "d1")),
	}, testPolicy())

	example := manifest.Examples[0]
	if example.InputDigest == secret || example.Primary.OutputDigest == secret {
		t.Fatal("manifest holds payload text where a digest belongs")
	}
	if example.InputDigest != digestOf(secret) {
		t.Fatalf("input digest %s does not hash the request", example.InputDigest)
	}
	if example.Primary.Model != "primary-model" || example.Shadows[0].Model != "candidate-a" {
		t.Fatalf("arm identity lost: %+v", example)
	}
	if example.Shadows[0].OutputDigest != "d1" {
		t.Fatalf("shadow digest %q, want the one the dispatcher recorded", example.Shadows[0].OutputDigest)
	}
}

func TestBuildRejectsAPolicyThatCannotBeReproduced(t *testing.T) {
	for name, policy := range map[string]Policy{
		"no seed":        {Splits: []Split{{Name: "train", Weight: 1}}},
		"no split":       {Seed: "seed-a"},
		"zero weight":    {Seed: "seed-a", Splits: []Split{{Name: "train", Weight: 0}}},
		"repeated split": {Seed: "seed-a", Splits: []Split{{Name: "train", Weight: 1}, {Name: "train", Weight: 1}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Build(nil, policy); err == nil {
				t.Fatal("Build accepted a policy it cannot reproduce a split from")
			}
		})
	}
}

func TestValidateAcceptsWhatBuildProduced(t *testing.T) {
	manifest := buildOrFail(t, []store.Record{
		comparedRecord("r1", "ask one", "answer one", shadowOutcome("candidate-a", "d1")),
		comparedRecord("r2", "ask two", "answer two", shadowOutcome("candidate-b", "d2")),
	}, testPolicy())

	if err := Validate(manifest); err != nil {
		t.Fatalf("Validate rejected a manifest Build produced: %v", err)
	}
}

func TestValidateRejectsAnEditedManifest(t *testing.T) {
	base := buildOrFail(t, []store.Record{
		comparedRecord("r1", "ask one", "answer one", shadowOutcome("candidate-a", "d1")),
	}, testPolicy())

	edited := base
	edited.Examples = append([]Example(nil), base.Examples...)
	edited.Examples[0].Primary.OutputDigest = digestOf("a different answer")

	if err := Validate(edited); err == nil {
		t.Fatal("Validate accepted a manifest whose content no longer hashes to its digest")
	}
}

func TestValidateRejectsTheSameInputInTwoSplits(t *testing.T) {
	base := buildOrFail(t, []store.Record{
		comparedRecord("r1", "ask one", "answer one", shadowOutcome("candidate-a", "d1")),
	}, testPolicy())

	leaked := base
	leaked.Examples = append([]Example(nil), base.Examples...)
	duplicate := leaked.Examples[0]
	duplicate.Lineage.ReplayID = "r2"
	duplicate.ID = exampleID(duplicate.InputDigest, duplicate.Primary.Model, duplicate.Lineage.ReplayID)
	duplicate.Split = otherSplit(base.Examples[0].Split)
	leaked.Examples = append(leaked.Examples, duplicate)
	leaked.Counts.Examples = len(leaked.Examples)
	digest, err := leaked.contentDigest()
	if err != nil {
		t.Fatalf("contentDigest: %v", err)
	}
	leaked.Digest = digest

	if err := Validate(leaked); err == nil {
		t.Fatal("Validate accepted the same input in two splits")
	}
}

func TestValidateRejectsAnExampleWithoutEvidence(t *testing.T) {
	base := buildOrFail(t, []store.Record{
		comparedRecord("r1", "ask one", "answer one", shadowOutcome("candidate-a", "d1")),
	}, testPolicy())

	for name, damage := range map[string]func(*Example){
		"no shadow arm": func(e *Example) { e.Shadows = nil },
		"no lineage":    func(e *Example) { e.Lineage.ReplayID = "" },
		"no primary":    func(e *Example) { e.Primary.Model = "" },
		"unknown split": func(e *Example) { e.Split = "holdout" },
	} {
		t.Run(name, func(t *testing.T) {
			broken := base
			broken.Examples = append([]Example(nil), base.Examples...)
			damage(&broken.Examples[0])
			if err := Validate(broken); err == nil {
				t.Fatal("Validate accepted an example that carries no usable evidence")
			}
		})
	}
}

func otherSplit(name string) string {
	if name == "train" {
		return "eval"
	}
	return "train"
}
