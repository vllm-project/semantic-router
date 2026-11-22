package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// has_session=true keeps the records that belong to a session, so a sessions
// view's totals count what its list of runs holds; the plain API calls of the
// range stay out of the list and the aggregate alike.
func TestReplayHasSessionKeepsTheSessionsRecords(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(20, 0))
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	start := time.Date(2026, 11, 19, 9, 0, 0, 0, time.UTC)
	for index, session := range []string{"ses-a", "ses-a", "", "ses-b", ""} {
		_, err := recorder.AddRecord(routerreplay.RoutingRecord{
			ID: "record-" + string(rune('a'+index)), SessionID: session, Timestamp: start.Add(time.Duration(index) * time.Second),
			SelectedModel: "small", LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
		})
		require.NoError(t, err)
	}

	var list struct {
		Total int `json:"total"`
	}
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays", &list))
	require.Equal(t, 5, list.Total)
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays?has_session=true", &list))
	require.Equal(t, 3, list.Total)

	var aggregate struct {
		RecordCount int `json:"record_count"`
	}
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate?has_session=true", &aggregate))
	require.Equal(t, 3, aggregate.RecordCount)
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate?has_session=false", &aggregate))
	require.Equal(t, 5, aggregate.RecordCount)

	var invalid map[string]any
	require.Equal(t, 400, replayAPIJSON(t, router, "/api/v1/observability/replays?has_session=maybe", &invalid))
}
