//go:build !windows

package apiserver

import (
	"net/http"
	"net/url"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// modelRuntimeInventoryResponse lists the model_runtime deployments of the
// published configuration: where each one runs, its state and its served card.
type modelRuntimeInventoryResponse struct {
	Deployments []modelRuntimeDeployment `json:"deployments"`
	Count       int                      `json:"count"`
}

// modelRuntimeDeployment is one deployment. Endpoint names only the transport
// (unix, or an http(s) scheme and host), never a socket path or credentials.
// The card fields stay empty until the runtime has reported the model ready.
type modelRuntimeDeployment struct {
	DesiredReplicas int                          `json:"desired_replicas"`
	ReadyReplicas   int                          `json:"ready_replicas"`
	Replicas        []modelservice.ReplicaStatus `json:"replicas,omitempty"`
	Name            string                       `json:"name"`
	Managed         bool                         `json:"managed"`
	Process         string                       `json:"process"`
	ServedName      string                       `json:"served_name"`
	Endpoint        string                       `json:"endpoint,omitempty"`
	Ready           bool                         `json:"ready"`
	State           string                       `json:"state"`
	Reason          string                       `json:"reason,omitempty"`
	Restarts        int                          `json:"restarts"`
	Family          string                       `json:"family,omitempty"`
	Repo            string                       `json:"repo,omitempty"`
	Revision        string                       `json:"revision,omitempty"`
	Surfaces        []string                     `json:"surfaces,omitempty"`
	Heads           []modelRuntimeHead           `json:"heads,omitempty"`
	Embedding       *modelRuntimeEmbedding       `json:"embedding,omitempty"`
	Rerank          *modelRuntimeRerank          `json:"rerank,omitempty"`
	Device          string                       `json:"device,omitempty"`
	Profile         string                       `json:"profile,omitempty"`
	Engine          string                       `json:"engine,omitempty"`
}

type modelRuntimeHead struct {
	Name   string   `json:"name"`
	Kind   string   `json:"kind"`
	Labels []string `json:"labels,omitempty"`
}

type modelRuntimeEmbedding struct {
	Dimensions []int    `json:"dimensions,omitempty"`
	Layers     []int    `json:"layers,omitempty"`
	Modalities []string `json:"modalities,omitempty"`
}

type modelRuntimeRerank struct {
	DefaultLayer int   `json:"default_layer"`
	Layers       []int `json:"layers,omitempty"`
}

// handleModelRuntimeInventory handles GET /api/v1/inventory/model-runtime.
func (s *ClassificationAPIServer) handleModelRuntimeInventory(w http.ResponseWriter, _ *http.Request) {
	var statuses []modelservice.DeploymentStatus
	if manager := modelservice.DefaultManager(); manager != nil {
		statuses = manager.Statuses()
	}
	s.writeJSONResponse(w, http.StatusOK, modelRuntimeInventory(statuses))
}

func modelRuntimeInventory(statuses []modelservice.DeploymentStatus) modelRuntimeInventoryResponse {
	deployments := make([]modelRuntimeDeployment, 0, len(statuses))
	for _, status := range statuses {
		deployment := modelRuntimeDeployment{
			DesiredReplicas: status.DesiredReplicas, ReadyReplicas: status.ReadyReplicas, Replicas: status.Replicas,
			Name: status.Name, Managed: status.Managed, Process: status.Process, ServedName: status.Model,
			Endpoint: inventoryEndpoint(status.Endpoint), Ready: status.Ready, State: status.State,
			Reason: status.Reason, Restarts: status.Restarts,
		}
		if card := status.Card; card != nil {
			deployment.Family, deployment.Repo, deployment.Revision = card.Family, card.Repo, card.Revision
			deployment.Surfaces = card.Surfaces
			deployment.Device, deployment.Profile, deployment.Engine = card.Device, card.Profile, card.Engine
			for _, head := range card.Heads {
				deployment.Heads = append(deployment.Heads, modelRuntimeHead{Name: head.Name, Kind: head.Kind, Labels: head.Labels})
			}
			if embedding := card.Embedding; embedding != nil {
				deployment.Embedding = &modelRuntimeEmbedding{Dimensions: embedding.Dimensions, Layers: embedding.Layers, Modalities: embedding.Modalities}
			}
			if rerank := card.Rerank; rerank != nil {
				deployment.Rerank = &modelRuntimeRerank{DefaultLayer: rerank.Default.Layer}
				for _, exit := range rerank.Exits {
					deployment.Rerank.Layers = append(deployment.Rerank.Layers, exit.Layer)
				}
			}
		}
		deployments = append(deployments, deployment)
	}
	return modelRuntimeInventoryResponse{Deployments: deployments, Count: len(deployments)}
}

// inventoryEndpoint names a deployment's transport without its socket path,
// credentials, path or query.
func inventoryEndpoint(endpoint string) string {
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Scheme == "" {
		return ""
	}
	if parsed.Scheme == "unix" {
		return "unix"
	}
	return parsed.Scheme + "://" + parsed.Host
}
