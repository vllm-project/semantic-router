package extproc

import (
	"context"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// FrontendModels retains a generation's native model clients independently of
// its optional routing pipeline. The process-wide Manager owns worker lifetime;
// generations keep references, so toggling routing does not restart workers.
type FrontendModels struct{ lease *modelservice.Lease }

func (m *FrontendModels) Close(context.Context) error {
	if m != nil && m.lease != nil {
		return m.lease.Close()
	}
	return nil
}

func (m *FrontendModels) SystemOne(ctx context.Context, deployment string, body json.RawMessage) (modelservice.SystemOneResult, error) {
	if m == nil {
		return modelservice.SystemOneResult{}, modelservice.ErrUnavailable
	}
	return m.lease.SystemOne(ctx, deployment, body)
}

func frontendModelPart() configsnapshot.PartBuilder {
	return configsnapshot.PartBuilder{
		Component: configsnapshot.ComponentModelService,
		Build: func(_ context.Context, candidate *configsnapshot.Snapshot, _ configsnapshot.Part) (configsnapshot.Part, error) {
			models := &FrontendModels{}
			if manager := modelservice.DefaultManager(); manager != nil {
				lease, err := manager.Acquire(candidate.Config())
				if err != nil {
					return nil, err
				}
				models.lease = lease
			}
			return models, nil
		},
		Warm: func(ctx context.Context, part configsnapshot.Part) error {
			if lease := part.(*FrontendModels).lease; lease != nil {
				return lease.WaitManaged(ctx)
			}
			return nil
		},
	}
}
