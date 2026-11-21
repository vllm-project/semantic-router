package classification

import (
	"cmp"
	"encoding/json"
	"slices"
)

type windowModel struct {
	fields         []string
	precisionEpoch uint64 // local validity check, never part of a shared key
}

// windowModelIdentity binds both the model and the connection that selects its
// guard policy. An unknown identity disables local and shared reads and writes.
func (c *Classifier) windowModelIdentity(name string) (windowModel, bool) {
	declared := c.Config.ModelDeployments[name]
	clientCertificate := declared.TLS != nil && (declared.TLS.CertFile != "" || declared.TLS.KeyFile != "")
	// Authentication can select a policy independently of model metadata.
	// Keys can rotate between calls, and certificate files can differ between
	// replicas. This applies to attached model_runtime workers too.
	if declared.Auth != nil || clientCertificate || len(declared.ForwardRequestHeaders) > 0 {
		return windowModel{}, false
	}
	connection := struct {
		Headers    map[string]string
		ServerName string
		Placements [][2]string
	}{Headers: declared.Headers}
	if declared.TLS != nil {
		// ServerName is sent as SNI, so it can select a different backend.
		connection.ServerName = declared.TLS.ServerName
	}
	model := []string{name, declared.Provider, declared.Artifact, declared.Revision}
	var epoch uint64
	switch {
	case declared.IsSystemOne():
		if declared.Endpoint == "" || declared.Model == "" {
			return windowModel{}, false
		}
		model = append(model, declared.Endpoint, declared.Model, declared.EffectiveSchema())
		model = append(model, declared.Deviations...)
	case declared.IsModelRuntime():
		// Include every attached worker, even in a mixed managed/attached pool.
		// The runtime client uses the deployment name when placement.Model is empty.
		for _, placement := range declared.Placements() {
			if placement.Endpoint != "" {
				connection.Placements = append(connection.Placements, [2]string{placement.Endpoint, cmp.Or(placement.Model, name)})
			}
		}
		slices.SortFunc(connection.Placements, func(a, b [2]string) int {
			return cmp.Or(cmp.Compare(a[0], b[0]), cmp.Compare(a[1], b[1]))
		})
		// Prepared task cards describe capabilities, not necessarily the workers
		// that can serve this scan. Only the live observation proves precision.
		if c.models == nil || c.models.runtime == nil {
			return windowModel{}, false
		}
		card, ok := c.models.runtime.CurrentDeploymentCard(name)
		if !ok || card.Dtype == "" || card.Dtype == "auto" {
			return windowModel{}, false
		}
		model = append(model, card.Repo, card.Revision, card.ModelSHA256, card.Dtype)
		if len(declared.Replicas) > 1 {
			epoch = card.PrecisionEpoch
			if epoch == 0 {
				return windowModel{}, false
			}
			// Older Routers could publish mixed-pool results under the baseline precision.
			model = append(model, "uniform-worker-precision:v1")
		}
	default:
		return windowModel{}, false
	}
	// JSON sorts static header keys. Replica order and map insertion order
	// therefore do not invalidate an otherwise identical connection.
	encoded, err := json.Marshal(connection)
	if err != nil {
		return windowModel{}, false
	}
	return windowModel{fields: append(model, string(encoded)), precisionEpoch: epoch}, true
}
