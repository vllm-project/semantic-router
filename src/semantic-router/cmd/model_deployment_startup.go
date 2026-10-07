package main

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// modelDeploymentProgressInterval is how often startup progress re-reads the
// states of the Router-managed model deployments while it waits for them.
const modelDeploymentProgressInterval = time.Second

// reportModelDeploymentProgress writes startup progress while the first router
// generation waits for its Router-managed model deployments, until the
// returned stop is called: each change of a deployment's state is written,
// naming the deployments that are not ready yet. stop returns once no further
// write can happen, so the caller's next write is the last one.
func reportModelDeploymentProgress(writer startupstatus.StatusWriter, manager *modelservice.Manager) (stop func()) {
	if manager == nil {
		return func() {}
	}
	done, finished := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(finished)
		ticker := time.NewTicker(modelDeploymentProgressInterval)
		defer ticker.Stop()
		var reported []startupstatus.ModelDeploymentStatus
		for {
			deployments := managedDeploymentStatuses(manager.Statuses())
			if len(deployments) > 0 && !slices.Equal(deployments, reported) {
				writeStartupState(writer, modelDeploymentProgress(deployments), "Failed to write model deployment startup status")
				reported = deployments
			}
			select {
			case <-done:
				return
			case <-ticker.C:
			}
		}
	}()
	return func() {
		close(done)
		<-finished
	}
}

// modelDeploymentProgress is the startup state while the Router waits for its
// managed deployments; once all are ready, startup prepares the rest.
func modelDeploymentProgress(deployments []startupstatus.ModelDeploymentStatus) startupstatus.State {
	state := startupstatus.State{Phase: "initializing_models"}
	withModelDeployments(&state, deployments)
	if len(state.PendingModels) == 0 {
		state.Message = fmt.Sprintf("Router-managed model deployments are ready (%d). Starting router services...", state.TotalModels)
		return state
	}
	waiting := make([]string, 0, len(state.PendingModels))
	for _, deployment := range deployments {
		if !deployment.Ready {
			waiting = append(waiting, describeModelDeployment(deployment))
		}
	}
	state.Phase = startupstatus.PhaseLoadingModelDeployments
	state.Message = fmt.Sprintf("Waiting for Router-managed model deployments, %d of %d ready: %s",
		state.ReadyModels, state.TotalModels, strings.Join(waiting, "; "))
	return state
}

func describeModelDeployment(deployment startupstatus.ModelDeploymentStatus) string {
	text := deployment.Name
	if deployment.Artifact != "" {
		text += " (" + deployment.Artifact + ")"
	}
	text += " " + deployment.State
	if deployment.Reason != "" {
		text += ": " + deployment.Reason
	}
	return text
}

// withModelDeployments lists the deployments in state and counts them.
func withModelDeployments(state *startupstatus.State, deployments []startupstatus.ModelDeploymentStatus) {
	state.ModelDeployments = deployments
	state.TotalModels = len(deployments)
	for _, deployment := range deployments {
		if deployment.Ready {
			state.ReadyModels++
		} else {
			state.PendingModels = append(state.PendingModels, deployment.Name)
		}
	}
}

// managedDeploymentStatuses keeps the deployments whose runtime process the
// Router manages, in name order.
func managedDeploymentStatuses(statuses []modelservice.DeploymentStatus) []startupstatus.ModelDeploymentStatus {
	var deployments []startupstatus.ModelDeploymentStatus
	for _, status := range statuses {
		if !status.Managed {
			continue
		}
		deployments = append(deployments, startupstatus.ModelDeploymentStatus{
			Name: status.Name, Artifact: status.Artifact, Process: status.Process,
			State: status.State, Ready: status.Ready, Reason: status.Reason,
		})
	}
	return deployments
}
