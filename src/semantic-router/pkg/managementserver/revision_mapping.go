package managementserver

// publicRevision projects signed persistence revisions into the unsigned HTTP
// contract without allowing a corrupt negative value to wrap into a very large
// revision. Domain services only emit positive revisions; zero remains the
// transport sentinel for an invalid result.
func publicRevision(value int64) uint64 {
	if value <= 0 {
		return 0
	}
	// #nosec G115 -- the positive int64 domain is fully representable by uint64.
	return uint64(value)
}
