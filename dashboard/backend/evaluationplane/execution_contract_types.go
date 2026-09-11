package evaluationplane

type executorSuiteClass string

const (
	executorSuiteFixture    executorSuiteClass = "fixture"
	executorSuiteRuntime    executorSuiteClass = "runtime"
	executorSuiteNormalized executorSuiteClass = "normalized-suite"
	executorSuiteMoMCohort  executorSuiteClass = "mom-cohort"
)

type targetExecutionProfile string

const (
	targetProfileRecorded targetExecutionProfile = "recorded-source"
	targetProfileRuntime  targetExecutionProfile = "brokered-runtime"
)

type policySnapshotProfile string

const (
	policySnapshotFixture    policySnapshotProfile = "fixture"
	policySnapshotNormalized policySnapshotProfile = "normalized-suite-revisions"
	policySnapshotRuntime    policySnapshotProfile = "runtime-config"
)

type lineageProfile string

const (
	lineageFixture    lineageProfile = "fixture-replay"
	lineageNormalized lineageProfile = "normalized-suite-replay"
	lineageRuntime    lineageProfile = "runtime"
)

// executorContract is the single server-owned meaning of an executor ID. The
// public catalog only declares which IDs a target accepts; validation and
// lineage derive behavior from this immutable registry entry.
type executorContract struct {
	ID                       string
	Mode                     Mode
	SuiteClass               executorSuiteClass
	TargetProfile            targetExecutionProfile
	LineageProfile           lineageProfile
	TrackIDs                 []TrackID
	NormalizedSuite          bool
	RecordedNormalizedSource bool
	RequiresFixtureRef       bool
	CaseBudgetPerSuite       bool
	EvidenceLevelCeiling     EvidenceLevel
}

type targetContract struct {
	ExecutionProfile  targetExecutionProfile
	PolicySnapshot    policySnapshotProfile
	TrackRequirements map[TrackID][]targetFeature
}

type executionTargetContract struct {
	Definition targetDefinition
}

type executionContractRegistry struct {
	executors map[string]executorContract
	targets   map[string]executionTargetContract
}

type resolvedExecutionContract struct {
	Executor executorContract
	Target   executionTargetContract
}
