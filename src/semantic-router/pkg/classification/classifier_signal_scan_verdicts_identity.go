package classification

import (
	"cmp"
	"encoding/json"
	"slices"
)

// windowModelIdentity binds both the model and the connection that selects its
// guard policy. A nil identity disables local and shared window reads and writes.
func (c *Classifier) windowModelIdentity(name string) ([]string, bool) {
	declared := c.Config.ModelDeployments[name]
	clientCertificate := declared.TLS != nil && (declared.TLS.CertFile != "" || declared.TLS.KeyFile != "")
	// Authentication can select a policy independently of model metadata.
	// Keys can rotate between calls, and certificate files can differ between
	// replicas. This applies to attached model_runtime workers too.
	if declared.Auth != nil || clientCertificate || len(declared.ForwardRequestHeaders) > 0 {
		return nil, false
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
	switch {
	case declared.IsSystemOne():
		if declared.Endpoint == "" || declared.Model == "" {
			return nil, false
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
		if card, ok := c.decisionTaskCard(name); ok {
			// With dtype: auto, the same weights can run at different precisions.
			model = append(model, card.Repo, card.Revision, card.ModelSHA256, card.Dtype)
		} else if len(connection.Placements) == 0 && declared.Artifact == "" {
			return nil, false
		}
	default:
		return nil, false
	}
	// JSON sorts static header keys. Replica order and map insertion order
	// therefore do not invalidate an otherwise identical connection.
	encoded, err := json.Marshal(connection)
	if err != nil {
		return nil, false
	}
	return append(model, string(encoded)), true
}
