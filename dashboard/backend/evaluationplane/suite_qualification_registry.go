package evaluationplane

// normalizedAdapterContract is the narrow source/import projection of a
// research benchmark. It intentionally does not promote a source pin into an
// upstream-native execution claim.
type normalizedAdapterContract struct {
	sourceRevision  string
	datasetRevision string
	decisionUnit    string
	actionSpace     string
	trackIDs        []TrackID
}

var normalizedAdapterContracts = normalizedAdapterContractsFromResearchInventory()

func normalizedAdapterContractsFromResearchInventory() map[string]normalizedAdapterContract {
	benchmarks := ResearchBenchmarkInventory()
	contracts := make(map[string]normalizedAdapterContract, len(benchmarks))
	for _, benchmark := range benchmarks {
		datasetRevision := ""
		if benchmark.DatasetRevision != nil {
			datasetRevision = *benchmark.DatasetRevision
		}
		contracts[benchmark.AdapterID] = normalizedAdapterContract{
			sourceRevision: benchmark.SourceRevision, datasetRevision: datasetRevision,
			decisionUnit: benchmark.DecisionUnit, actionSpace: benchmark.ActionSpace,
			trackIDs: append([]TrackID(nil), benchmark.ImportTracks...),
		}
	}
	return contracts
}

func normalizedAdapterTracksMatch(contract normalizedAdapterContract, trackIDs []TrackID) bool {
	allowed := make(map[TrackID]struct{}, len(contract.trackIDs))
	for _, trackID := range contract.trackIDs {
		allowed[trackID] = struct{}{}
	}
	for _, trackID := range trackIDs {
		if _, supported := allowed[trackID]; !supported {
			return false
		}
	}
	return len(trackIDs) > 0
}
