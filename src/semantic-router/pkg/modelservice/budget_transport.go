package modelservice

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

// budgetTransport sits on physical clients, below the replica pool. Discovery
// does not consume inference calls; a bundle consumes one physical exchange.
type budgetTransport struct{ next api.HttpRequestDoer }

func (b budgetTransport) Do(request *http.Request) (*http.Response, error) {
	if request.Method == http.MethodPost {
		if err := budget.Consume(request.Context()); err != nil {
			if request.Body != nil {
				_ = request.Body.Close()
			}
			return nil, err
		}
	}
	return b.next.Do(request)
}
