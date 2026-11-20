package systemonecodec

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// checkState refuses a state no question can read, naming it: one of another
// JSON type than a text, an object of parts or (media placed in front of it)
// an array, and a state that gives no text at all ("", only white space, or an
// object without a part) in a request without an image, a video or an asset to
// read instead. A state left out or null is refused where it is required.
func checkState(request systemoneprotocol.Request, at string) error {
	state := request.State
	switch state.Kind() {
	case systemoneprotocol.KindNull:
		return nil
	case systemoneprotocol.KindString, systemoneprotocol.KindObject, systemoneprotocol.KindArray:
	default:
		return fmt.Errorf("%sstate must be a text or an object of text parts", at)
	}
	if len(request.Assets) > 0 {
		return nil
	}
	for _, member := range request.Extensions {
		if (member.Key == "images" || member.Key == "videos") && len(member.Value.Items()) > 0 {
			return nil
		}
	}
	text, isText := state.Text()
	if (isText && strings.TrimSpace(text) == "") || (state.Kind() == systemoneprotocol.KindObject && len(state.Members()) == 0) {
		return fmt.Errorf("%sthe state is empty: send the text the questions are about", at)
	}
	return nil
}
