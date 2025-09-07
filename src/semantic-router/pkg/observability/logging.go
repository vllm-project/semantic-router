package observability

import (
	"encoding/json"
	"log"
	"time"
)

// LogEvent emits a structured JSON log line with a standard envelope
// Fields provided by callers take precedence and will not be overwritten.
func LogEvent(event string, fields map[string]interface{}) {
	if fields == nil {
		fields = map[string]interface{}{}
	}
	if _, ok := fields["event"]; !ok {
		fields["event"] = event
	}
	if _, ok := fields["ts"]; !ok {
		fields["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
	}
	b, err := json.Marshal(fields)
	if err != nil {
		// Fallback to regular log on marshal error
		log.Printf("event=%s marshal_error=%v fields_len=%d", event, err, len(fields))
		return
	}
	log.Println(string(b))
}
