package classification

import (
	"crypto/sha256"
	"encoding/hex"
)

// PrivacyEvidence describes only the exact input read at one stage. It carries
// no prompt bytes, and cannot certify a response, tool, or different input.
type PrivacyEvidence struct {
	Stage            string
	InputSHA256      string
	Coverage         string
	PersonalDataFree bool
}

func NewPrivacyEvidence(stage, input string, complete, personalDataFree bool) PrivacyEvidence {
	digest := sha256.Sum256([]byte(input))
	coverage := "unknown"
	if complete {
		coverage = "complete"
	}
	return PrivacyEvidence{Stage: stage, InputSHA256: hex.EncodeToString(digest[:]), Coverage: coverage, PersonalDataFree: complete && personalDataFree}
}

func (e PrivacyEvidence) CoversClean(stage, input string) bool {
	return e.Stage == stage && e.Coverage == "complete" && e.PersonalDataFree && e.InputSHA256 == NewPrivacyEvidence(stage, input, true, true).InputSHA256
}
