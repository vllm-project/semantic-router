package decision

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const SignalObservationVersionV1 = "v1"

type SignalObservationAvailability string

const (
	SignalObservationAvailable   SignalObservationAvailability = "available"
	SignalObservationUnavailable SignalObservationAvailability = "unavailable"
)

// SignalObservation is the versioned, recipe-scoped result passed from signal
// evaluation to decision evaluation. The envelope owns availability, bounded
// diagnostics, and provenance; PayloadType selects the declared typed payload.
type SignalObservation struct {
	Version      string
	SignalType   string
	Name         string
	Availability SignalObservationAvailability
	Provenance   SignalObservationProvenance
	PayloadType  string
	Payload      *LabelDistributionObservation
	ErrorCode    string
}

type SignalObservationProvenance struct {
	Evaluator string
	Backend   string
}

// LabelDistributionObservation is the bounded payload produced by a declared
// classifier signal, independent of the backend that computed it.
type LabelDistributionObservation struct {
	Scores        map[string]float64
	Thresholds    map[string]float64
	MatchedLabels []string
	Rationale     string
}

const (
	observationUnsupportedVersion = "signal_observation_unsupported_version"
	observationUnavailable        = "signal_observation_unavailable"
	observationInvalidPayload     = "signal_observation_invalid_payload"
)

func findSignalObservation(signals *SignalMatches, signalType, name string) (*SignalObservation, bool) {
	if signals == nil {
		return nil, false
	}
	for index := range signals.Observations {
		observation := &signals.Observations[index]
		if observation.SignalType == signalType && observation.Name == name {
			return observation, true
		}
	}
	return nil, false
}

func classifierObservation(signals *SignalMatches, name string) (*SignalObservation, bool) {
	return findSignalObservation(signals, config.SignalTypeClassifier, name)
}

func classifierObservationError(observation *SignalObservation) string {
	if observation.Version != SignalObservationVersionV1 {
		return observationUnsupportedVersion
	}
	if observation.Availability != SignalObservationAvailable {
		if observation.ErrorCode != "" {
			return observation.ErrorCode
		}
		return observationUnavailable
	}
	if observation.PayloadType != config.RemoteClassifierContractLabelDistribution || observation.Payload == nil {
		return observationInvalidPayload
	}
	return ""
}

func classifierObservationMatched(observation *SignalObservation, label string) (bool, bool) {
	if classifierObservationError(observation) != "" {
		return false, false
	}
	return slices.Contains(observation.Payload.MatchedLabels, label), true
}

func classifierObservationValue(observation *SignalObservation, label string) (float64, bool) {
	if classifierObservationError(observation) != "" {
		return 0, false
	}
	value, ok := observation.Payload.Scores[label]
	return value, ok
}
