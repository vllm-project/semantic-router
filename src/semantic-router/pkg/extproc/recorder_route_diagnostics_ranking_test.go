package extproc

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// Every trace field gets a distinct value, so a field the mapper forgets
// shows up as a JSON difference.
func TestReplayDecisionRankingCopiesEveryField(t *testing.T) {
	trace := &decision.RankingTrace{}
	value := reflect.ValueOf(trace).Elem()
	for i := 0; i < value.NumField(); i++ {
		switch field := value.Field(i); field.Kind() {
		case reflect.String:
			field.SetString(value.Type().Field(i).Name)
		case reflect.Int:
			field.SetInt(int64(i + 1))
		case reflect.Bool:
			field.SetBool(true)
		default:
			t.Fatalf("unhandled field kind %s; extend this test", field.Kind())
		}
	}
	want, _ := json.Marshal(trace)
	got, _ := json.Marshal(replayDecisionRanking(trace))
	if string(got) != string(want) {
		t.Fatalf("replay ranking = %s, want %s", got, want)
	}
}
