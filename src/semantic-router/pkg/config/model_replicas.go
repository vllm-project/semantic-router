package config

import (
	"fmt"
	"strings"
)

// ModelReplica places one worker. Model identity, input policy and numerical
// profile belong to the logical deployment and cannot vary between workers.
type ModelReplica struct {
	Device   string `yaml:"device,omitempty" json:"device,omitempty"`
	Endpoint string `yaml:"endpoint,omitempty" json:"endpoint,omitempty"`
	// Model selects the served model on an attached worker.
	Model string `yaml:"model,omitempty" json:"model,omitempty"`
}

// Placements returns effective worker placements without changing the authored
// declaration. An empty list means one worker, not an empty deployment.
func (d ModelDeployment) Placements() []ModelReplica {
	placements := append([]ModelReplica(nil), d.Replicas...)
	if len(placements) == 0 {
		placements = []ModelReplica{{Device: d.Device, Endpoint: d.Endpoint, Model: d.Model}}
	}
	for i := range placements {
		if placements[i].Device == "" && placements[i].Endpoint == "" {
			placements[i].Device = "auto"
		}
	}
	return placements
}

func (d ModelDeployment) validateReplicas() error {
	if len(d.Replicas) > 0 && (d.Device != "" || d.Endpoint != "" || d.Model != "") {
		return fmt.Errorf("replicas cannot be combined with deployment device, endpoint or model")
	}
	if len(d.Replicas) > 64 {
		return fmt.Errorf("replicas must contain at most 64 worker placements")
	}
	attached := make(map[string]bool)
	for i, placement := range d.Placements() {
		if placement.Endpoint != "" {
			if placement.Device != "" {
				return fmt.Errorf("replica %d: an attached endpoint cannot set device", i)
			}
			if err := validateModelRuntimeEndpoint(placement.Endpoint); err != nil {
				return fmt.Errorf("replica %d: %w", i, err)
			}
			if placement.Model != "" && (strings.TrimSpace(placement.Model) != placement.Model || strings.ContainsAny(placement.Model, "\x00\r\n")) {
				return fmt.Errorf("replica %d: model must be a trimmed model name", i)
			}
			key := strings.TrimRight(placement.Endpoint, "/") + "\x00" + placement.Model
			if attached[key] {
				return fmt.Errorf("replica %d: duplicate attached worker", i)
			}
			attached[key] = true
			continue
		}
		if placement.Model != "" {
			return fmt.Errorf("replica %d: model requires an attached endpoint", i)
		}
		if !modelRuntimeDevice.MatchString(placement.Device) {
			return fmt.Errorf("replica %d: device must be an accelerator name with an optional index, such as cpu, cuda:0 or rocm:1", i)
		}
	}
	return nil
}
