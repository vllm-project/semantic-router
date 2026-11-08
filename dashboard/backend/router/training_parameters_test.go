package router

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

func TestTrainingHTTPParameterNumbersSurviveRestart(t *testing.T) {
	authService, tokens := trainingTokens(t)
	root := t.TempDir()
	server := openTrainingHTTP(t, root, authService)
	t.Cleanup(func() { _ = server.store.Close() })
	token := tokens["alice"]
	body, err := os.ReadFile("../../../src/semantic-router/pkg/trainingcontract/testdata/selector.json")
	if err != nil {
		t.Fatal(err)
	}
	var f c.Fixture
	if operationErr := json.Unmarshal(body, &f); operationErr != nil {
		t.Fatal(operationErr)
	}
	upload := trainingDecode[c.Upload](t, trainingRequest(t, server.handler, token, "POST", "/uploads", []byte("dataset"), 201))
	asset := trainingDecode[c.DataAsset](t, trainingRequest(t, server.handler, token, "POST", "/data-assets", f.Asset.DataAssetSpec, 201))
	snapshotSpec := f.Snapshot.SnapshotSpec
	snapshotSpec.AssetID, snapshotSpec.UploadHandle = asset.ID, upload.ID
	snapshot := trainingDecode[c.DataSnapshot](t, trainingRequest(t, server.handler, token, "POST", "/data-snapshots", snapshotSpec, 201))
	experiment := trainingDecode[c.Experiment](t, trainingRequest(t, server.handler, token, "POST", "/experiments", f.Experiment.ExperimentSpec, 201))
	req := f.Submit
	req.Spec.SnapshotID, req.Spec.ExperimentID = snapshot.ID, experiment.ID
	// Raw JSON ensures the HTTP decoder, not the test client, chooses number types.
	req.Spec.Parameters = map[string]any{
		"seed":   json.RawMessage(`9007199254740993`),
		"nested": json.RawMessage(`[9007199254740993,{"epochs":100,"learning_rate":0.001,"weight":1.0,"scale":1e20}]`),
	}
	want := map[string]any{
		"seed":   int(9007199254740993),
		"nested": []any{int(9007199254740993), map[string]any{"epochs": 100, "learning_rate": 0.001, "weight": float64(1), "scale": 1e20}},
	}
	assertParameters := func(parameters map[string]any) {
		t.Helper()
		if !reflect.DeepEqual(parameters, want) {
			t.Fatalf("frozen parameters=%#v want=%#v", parameters, want)
		}
	}
	g := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "POST", "/runs", req, 202))
	assertParameters(g.Run.Spec.Parameters)
	req.Spec.Parameters["seed"] = json.RawMessage(`9007199254740992`)
	trainingRequest(t, server.handler, token, "POST", "/runs", req, 409)
	req.Spec.Parameters["seed"] = json.RawMessage(`9007199254740993`)
	claims, err := authService.ParseToken(token)
	if err != nil {
		t.Fatal(err)
	}
	work, err := server.service.StartAttempt(t.Context(), claims.UserID, g.Run.ID, g.Tasks[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	assertParameters(work.Parameters)
	if operationErr := server.store.Close(); operationErr != nil {
		t.Fatal(operationErr)
	}
	server = openTrainingHTTP(t, root, authService)
	replay := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "POST", "/runs", req, 202))
	if replay.Run.ID != g.Run.ID {
		t.Fatal("replay after restart created another run")
	}
	assertParameters(replay.Run.Spec.Parameters)
	saved := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "GET", "/runs/"+g.Run.ID, nil, 200))
	assertParameters(saved.Run.Spec.Parameters)
	runs := trainingDecode[[]c.TrainingRun](t, trainingRequest(t, server.handler, token, "GET", "/runs?experiment_id="+experiment.ID, nil, 200))
	if len(runs) != 1 {
		t.Fatalf("run listing=%+v", runs)
	}
	assertParameters(runs[0].Spec.Parameters)
	recovery, err := server.service.Recover(t.Context())
	if err != nil || len(recovery) != 1 {
		t.Fatalf("recovery=%+v err=%v", recovery, err)
	}
	assertParameters(recovery[0].Request.Parameters)
	if recovery[0].Request.AttemptID != work.AttemptID {
		t.Fatal("recovery changed attempt identity")
	}
	for _, value := range []string{"9223372036854775808", "-9223372036854775809", "1e309"} {
		req.Spec.Parameters["seed"] = json.RawMessage(value)
		trainingRequest(t, server.handler, token, "POST", "/runs", req, 400)
		trainingRequest(t, server.handler, token, "POST", "/runs/validate", c.ValidationRequest{SchemaVersion: c.Version, Spec: req.Spec}, 400)
	}
}
