package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

var capacityContractValidationCases = []struct {
	name    string
	mutate  func(*CreateRunRequest)
	message string
}{
	{
		name: "missing SLO",
		mutate: func(request *CreateRunRequest) {
			request.CapacitySLO = nil
		},
		message: "requires capacity_slo",
	},
	{
		name: "missing load protocol",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol = nil
		},
		message: "requires capacity_load_protocol",
	},
	{
		name: "single load level",
		mutate: func(request *CreateRunRequest) {
			request.Concurrency = 1
			request.CapacitySLO.RequiredConcurrency = 1
		},
		message: "concurrency of at least 2",
	},
	{
		name: "required concurrency above run",
		mutate: func(request *CreateRunRequest) {
			request.CapacitySLO.RequiredConcurrency = 3
		},
		message: "invalid operating bounds",
	},
	{
		name: "unbounded error rate",
		mutate: func(request *CreateRunRequest) {
			request.CapacitySLO.MaxErrorRate = 1
		},
		message: "invalid operating bounds",
	},
	{
		name: "zero throughput",
		mutate: func(request *CreateRunRequest) {
			request.CapacitySLO.MinThroughputRPS = 0
		},
		message: "invalid operating bounds",
	},
	{
		name: "zero scaling efficiency",
		mutate: func(request *CreateRunRequest) {
			request.CapacitySLO.MinThroughputScalingEfficiency = 0
		},
		message: "invalid operating bounds",
	},
	{
		name: "wrong load kind",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.Kind = "open-loop"
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "incomplete concurrency ladder",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.ConcurrencyLevels = []int64{1}
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "weak warmup",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.WarmupRequestMultiplier = 1
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "tiny measurement window",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.MeasurementRequestsPerRepetition = 2
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "too few repetitions",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.RepetitionsPerLevel = 2
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "wrong confidence",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.ConfidenceLevel = 0.9
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "wrong minimum independent clusters",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.MinimumMeasurementClustersPerLevel = 2
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "wrong error rate cluster range",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.MaxErrorRateClusterRange = 0.1
		},
		message: "violates the platform measurement contract",
	},
	{
		name: "weak stability threshold",
		mutate: func(request *CreateRunRequest) {
			request.CapacityLoadProtocol.MaxLatencyP95CV = 0.21
		},
		message: "violates the platform measurement contract",
	},
}

func TestCreateRunRequiresCurrentCapacityContract(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	if err := os.WriteFile(service.registrySource.configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write Mixture-of-Models config: %v", err)
	}
	valid := liveCapacityCreateRequest()
	for _, test := range capacityContractValidationCases {
		t.Run(test.name, func(t *testing.T) {
			request := valid
			request.ClientRequestID = newTestClientRequestID()
			request.CapacitySLO = copyCapacitySLO(valid.CapacitySLO)
			request.CapacityLoadProtocol = copyCapacityLoadProtocol(valid.CapacityLoadProtocol)
			test.mutate(&request)
			_, err := service.CreateRunAs(context.Background(), SystemActor(), request)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("CreateRun error=%v, want %q ErrInvalid", err, test.message)
			}
		})
	}

	replay := validCreateRequest()
	replay.CapacitySLO = testCapacitySLO(1)
	replay.CapacityLoadProtocol = defaultCapacityLoadProtocol(2)
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), replay); !errors.Is(err, ErrInvalid) ||
		!strings.Contains(err.Error(), "only for a live capacity track") {
		t.Fatalf("replay capacity contract error=%v, want scope ErrInvalid", err)
	}
}

func TestCreateRunFreezesCapacityContractIntoRunAndManifest(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	if err := os.WriteFile(service.registrySource.configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write Mixture-of-Models config: %v", err)
	}
	request := liveCapacityCreateRequest()
	originalSLO := copyCapacitySLO(request.CapacitySLO)
	originalProtocol := copyCapacityLoadProtocol(request.CapacityLoadProtocol)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	manifest, _, err := service.readDurableManifest(run.ID)
	if err != nil {
		t.Fatalf("readDurableManifest: %v", err)
	}
	if !reflect.DeepEqual(run.CapacitySLO, originalSLO) ||
		!reflect.DeepEqual(manifest.CapacitySLO, originalSLO) ||
		!reflect.DeepEqual(run.CapacityLoadProtocol, originalProtocol) ||
		!reflect.DeepEqual(manifest.CapacityLoadProtocol, originalProtocol) {
		t.Fatalf("capacity contract was not frozen exactly: run=%+v manifest=%+v", run, manifest)
	}

	changed := request
	changed.CapacitySLO = copyCapacitySLO(request.CapacitySLO)
	changed.CapacityLoadProtocol = copyCapacityLoadProtocol(request.CapacityLoadProtocol)
	changed.CapacityLoadProtocol.MeasurementRequestsPerRepetition++
	if _, retryErr := service.CreateRunAs(context.Background(), SystemActor(), changed); !errors.Is(retryErr, ErrConflict) {
		t.Fatalf("changed capacity protocol retry error=%v, want ErrConflict", retryErr)
	}

	request.CapacitySLO.MaxLatencyP95MS = 1
	request.CapacityLoadProtocol.ConcurrencyLevels[0] = 99
	persisted, err := service.GetRunAs(SystemActor(), run.ID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if !reflect.DeepEqual(persisted.CapacitySLO, originalSLO) ||
		!reflect.DeepEqual(persisted.CapacityLoadProtocol, originalProtocol) {
		t.Fatal("caller mutation changed the frozen capacity contract")
	}

	for _, test := range []struct {
		name   string
		mutate func(*RunManifest)
	}{
		{name: "required concurrency", mutate: func(value *RunManifest) { value.CapacitySLO.RequiredConcurrency-- }},
		{name: "latency", mutate: func(value *RunManifest) { value.CapacitySLO.MaxLatencyP95MS++ }},
		{name: "error rate", mutate: func(value *RunManifest) { value.CapacitySLO.MaxErrorRate /= 2 }},
		{name: "throughput", mutate: func(value *RunManifest) { value.CapacitySLO.MinThroughputRPS++ }},
		{name: "scaling", mutate: func(value *RunManifest) { value.CapacitySLO.MinThroughputScalingEfficiency /= 2 }},
		{name: "load kind", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.Kind = "other" }},
		{name: "load levels", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.ConcurrencyLevels[0]++ }},
		{name: "warmup", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.WarmupRequestMultiplier++ }},
		{name: "measurement window", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.MeasurementRequestsPerRepetition++ }},
		{name: "repetitions", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.RepetitionsPerLevel++ }},
		{name: "cluster coverage", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.MinimumMeasurementClustersPerLevel-- }},
		{name: "confidence", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.ConfidenceLevel = 0.9 }},
		{name: "error stability", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.MaxErrorRateClusterRange = 0.1 }},
		{name: "throughput stability", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.MaxThroughputCV /= 2 }},
		{name: "latency stability", mutate: func(value *RunManifest) { value.CapacityLoadProtocol.MaxLatencyP95CV /= 2 }},
	} {
		t.Run("manifest identity binds "+test.name, func(t *testing.T) {
			tampered := manifest
			tampered.CapacitySLO = copyCapacitySLO(manifest.CapacitySLO)
			tampered.CapacityLoadProtocol = copyCapacityLoadProtocol(manifest.CapacityLoadProtocol)
			test.mutate(&tampered)
			digest, digestErr := manifestSemanticDigest(tampered)
			if digestErr != nil {
				t.Fatalf("manifestSemanticDigest: %v", digestErr)
			}
			if digest == manifest.ManifestDigest {
				t.Fatalf("capacity %s is absent from the immutable manifest identity", test.name)
			}
		})
	}
}

func TestCapacitySLOAssessmentProducesAttestedFailInsteadOfUnavailable(t *testing.T) {
	runDir := t.TempDir()
	writeCapacityRecords(t, runDir)
	profile := capacityTestProfile()
	profile.SLO.MaxLatencyP95MS = 5
	for index := range profile.Levels {
		profile.Levels[index].LatencySLOPassed = capacityBoolPointer(false)
		profile.Levels[index].Qualified = capacityBoolPointer(false)
	}
	profile.Assessment = capacityProfileAssessment{
		QualifiedConcurrency:  json.RawMessage("null"),
		SaturationConcurrency: json.RawMessage("1"),
		SLOHeadroom:           capacityInt64Pointer(-1),
		Verdict:               "fail",
		FailureReasons:        []string{"latency_p95"},
	}
	writeCapacityProfile(t, runDir, profile)
	manifest := capacityManifest()
	manifest.CapacitySLO.MaxLatencyP95MS = 5
	attestation, err := validateCapacityProfileArtifact(
		runDir,
		manifest,
		capacityReport(),
		capacityRecordsAttestation(),
	)
	if err != nil {
		t.Fatalf("validate failing profile: %v", err)
	}
	if attestation == nil || attestation.Headroom != -1 || attestation.LevelCount != 2 {
		t.Fatalf("failing capacity attestation=%+v, want headroom -1", attestation)
	}
}

func TestCapacitySLOAssessmentRejectsTampering(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*capacityProfileEvidence)
	}{
		{name: "missing typed SLO", mutate: func(profile *capacityProfileEvidence) { profile.SLO = nil }},
		{name: "forged headroom", mutate: func(profile *capacityProfileEvidence) { profile.Assessment.SLOHeadroom = capacityInt64Pointer(2) }},
		{name: "forged latency decision", mutate: func(profile *capacityProfileEvidence) {
			profile.Levels[0].LatencySLOPassed = capacityBoolPointer(false)
		}},
		{name: "forged qualified envelope", mutate: func(profile *capacityProfileEvidence) { profile.Levels[0].Qualified = capacityBoolPointer(false) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runDir := t.TempDir()
			writeCapacityRecords(t, runDir)
			profile := capacityTestProfile()
			test.mutate(&profile)
			writeCapacityProfile(t, runDir, profile)
			_, err := validateCapacityProfileArtifact(
				runDir,
				capacityManifest(),
				capacityReport(),
				capacityRecordsAttestation(),
			)
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("tampered profile error=%v, want ErrInvalid", err)
			}
		})
	}
}

func liveCapacityCreateRequest() CreateRunRequest {
	return CreateRunRequest{
		ClientRequestID:      newTestClientRequestID(),
		Name:                 "live capacity SLO",
		Description:          "frozen repeated closed-loop capacity contract",
		SuiteIDs:             []string{"live-capacity"},
		TrackIDs:             []TrackID{"capacity"},
		Mode:                 ModeLive,
		TargetID:             mixtureTargetID("default"),
		ChangeProfile:        "runtime_capacity",
		SampleLimit:          4,
		Concurrency:          2,
		CapacitySLO:          testCapacitySLO(2),
		CapacityLoadProtocol: defaultCapacityLoadProtocol(2),
		Seed:                 17,
	}
}
