package trainingcontract

// Fixture describes a complete management/worker exchange without live trainers.
// The same JSON examples are consumed by Go, Python and TypeScript tests.
type Fixture struct {
	Asset           DataAsset        `json:"asset"`
	Upload          Upload           `json:"upload"`
	Snapshot        DataSnapshot     `json:"snapshot"`
	Experiment      Experiment       `json:"experiment"`
	Submit          SubmitRunRequest `json:"submit"`
	Graph           RunGraph         `json:"graph"`
	WorkerRequest   WorkerRequest    `json:"worker_request"`
	TrainResult     WorkerResult     `json:"train_result"`
	Artifact        Artifact         `json:"artifact"`
	Variant         ArtifactVariant  `json:"variant"`
	EvaluateRequest WorkerRequest    `json:"evaluate_request"`
	EvaluateResult  WorkerResult     `json:"evaluate_result"`
	Evaluation      Evaluation       `json:"evaluation"`
	QualifyResult   WorkerResult     `json:"qualify_result"`
	Qualification   Qualification    `json:"qualification"`
	Proposal        BindingProposal  `json:"proposal"`
}
