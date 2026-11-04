package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// listingTimeout bounds one read of a remote models listing; it runs beside
// requests, never on their path.
const listingTimeout = 10 * time.Second

const listingLimit = 1 << 20

// Remote answers decision questions with deployments the model runtime does
// not serve: systemone deployments, which a remote System One API serves.
type Remote interface {
	Decider
	// Card is the model card questions to name compile against; ok is false
	// for a name Remote does not serve.
	Card(name string) (card ModelCard, ok bool)
}

// RemoteSystemOne asks the systemone deployments of one configuration their
// decision questions over their System One APIs. It is a signal provider
// beside the model runtime and never serves a request's model.
type RemoteSystemOne struct {
	deployments map[string]*systemOneDeployment
	ctx         context.Context
	cancel      context.CancelFunc
}

type systemOneDeployment struct {
	name     string
	url      string
	models   string
	model    string
	auth     *config.DeploymentAuth
	declared []string
	client   *http.Client
	started  atomic.Bool
	listed   atomic.Pointer[ModelCard]
	failures sync.Map
}

// NewRemoteSystemOne prepares a client for each systemone deployment of cfg.
// Nothing is called until a question or a card needs it.
func NewRemoteSystemOne(cfg *config.RouterConfig) (*RemoteSystemOne, error) {
	ctx, cancel := context.WithCancel(context.Background())
	remote := &RemoteSystemOne{deployments: map[string]*systemOneDeployment{}, ctx: ctx, cancel: cancel}
	if cfg == nil {
		return remote, nil
	}
	for name, deployment := range cfg.ModelDeployments {
		if !deployment.IsSystemOne() {
			continue
		}
		endpoint, client, err := newHTTPClient(deployment.Endpoint)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("systemone deployment %q: %w", name, err)
		}
		// A redirect would carry the credential header to another host.
		client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
		listing, err := url.Parse(endpoint)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("systemone deployment %q: endpoint: %w", name, err)
		}
		listing.Path = strings.TrimRight(listing.Path, "/") + "/models"
		listing.RawPath = ""
		remote.deployments[name] = &systemOneDeployment{
			name: name, url: endpoint, models: listing.String(), model: deployment.Model,
			auth: deployment.Auth, declared: append([]string(nil), deployment.QuestionTypes...), client: client,
		}
	}
	return remote, nil
}

// Card is the card name's questions compile against: the question types its
// API lists for the model in GET <endpoint>/models, else the declared
// question_types, else the System One types. The first call reads the listing
// in the background; until it answers, the declared or System One types apply.
func (r *RemoteSystemOne) Card(name string) (ModelCard, bool) {
	d := r.deployment(name)
	if d == nil {
		return ModelCard{}, false
	}
	if card := d.listed.Load(); card != nil {
		return *card, true
	}
	if d.started.CompareAndSwap(false, true) {
		go d.readListing(r.ctx)
	}
	return RemoteCard(d.model, d.declared, nil), true
}

// Decide asks name every question of request in one System One call. The API
// answers a request with a failed question as a whole, so the call's questions
// share one outcome.
func (r *RemoteSystemOne) Decide(ctx context.Context, name string, request Request) (Response, error) {
	d := r.deployment(name)
	if d == nil {
		requestsTotal.WithLabelValues(name, ErrorReason(ErrUnknownDeployment)).Inc()
		return Response{}, ErrUnknownDeployment
	}
	started := time.Now()
	response, err := d.decide(ctx, request)
	requestDuration.WithLabelValues(name, "decisions").Observe(time.Since(started).Seconds())
	requestsTotal.WithLabelValues(name, ErrorReason(err)).Inc()
	return response, err
}

// Close stops listing reads and drops idle connections.
func (r *RemoteSystemOne) Close() error {
	if r == nil {
		return nil
	}
	r.cancel()
	for _, d := range r.deployments {
		d.client.CloseIdleConnections()
	}
	return nil
}

func (r *RemoteSystemOne) deployment(name string) *systemOneDeployment {
	if r == nil {
		return nil
	}
	return r.deployments[name]
}

func (d *systemOneDeployment) decide(ctx context.Context, request Request) (Response, error) {
	request.Model = d.model
	body, err := EncodeSystemOneRequest(ctx, request)
	if err != nil {
		return Response{}, err
	}
	header, err := d.header("application/json")
	if err != nil {
		d.reportFailure(0, "credential", err)
		return Response{}, fmt.Errorf("%w: systemone deployment %q: %w", ErrFailed, d.name, err)
	}
	header.Set("Content-Type", "application/json")
	done := Await(ctx)
	status, data, err := d.exchange(ctx, http.MethodPost, d.url, body, header)
	done()
	if ctxErr := ctx.Err(); ctxErr != nil {
		return Response{}, ctxErr
	}
	if err != nil {
		d.reportFailure(0, "transport", err)
		return Response{}, fmt.Errorf("%w: systemone deployment %q: %w", ErrFailed, d.name, err)
	}
	response, err := DecodeSystemOneResponse(status, data, request.Questions)
	if err != nil {
		d.reportFailure(status, ErrorReason(err), nil)
	}
	return response, err
}

// header carries the credential, read when the call needs it.
func (d *systemOneDeployment) header(accept string) (http.Header, error) {
	header := http.Header{"Accept": {accept}}
	if d.auth == nil {
		return header, nil
	}
	key, err := d.auth.Key()
	if err != nil {
		return nil, err
	}
	header.Set(d.auth.HeaderName(), d.auth.HeaderValue(key))
	return header, nil
}

// exchange sends one request. A transport error drops the request URL, whose
// query may hold a gateway's parameters.
func (d *systemOneDeployment) exchange(ctx context.Context, method, target string, body []byte, header http.Header) (int, []byte, error) {
	request, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(body))
	if err != nil {
		return 0, nil, errors.New("invalid endpoint")
	}
	request.Header = header
	response, err := d.client.Do(request)
	if err != nil {
		var urlErr *url.Error
		if errors.As(err, &urlErr) {
			err = urlErr.Err
		}
		return 0, nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, SystemOneResponseLimit+1))
	if err != nil {
		return 0, nil, err
	}
	return response.StatusCode, data, nil
}

func (d *systemOneDeployment) readListing(parent context.Context) {
	ctx, cancel := context.WithTimeout(parent, listingTimeout)
	defer cancel()
	card, err := d.listing(ctx)
	if err != nil {
		logging.ComponentEvent("model_runtime", "systemone_deployment_unlisted", map[string]interface{}{
			"deployment": d.name, "reason": err.Error(), "question_types": RemoteCard(d.model, d.declared, nil).QuestionTypes,
		})
		return
	}
	d.listed.Store(&card)
}

func (d *systemOneDeployment) listing(ctx context.Context) (ModelCard, error) {
	header, err := d.header("application/json")
	if err != nil {
		return ModelCard{}, err
	}
	status, data, err := d.exchange(ctx, http.MethodGet, d.models, nil, header)
	if err != nil {
		return ModelCard{}, err
	}
	if status != http.StatusOK || len(data) > listingLimit {
		return ModelCard{}, fmt.Errorf("the models listing answered %d", status)
	}
	var models struct {
		Data []struct {
			ID            string   `json:"id"`
			QuestionTypes []string `json:"question_types"`
			Presets       []string `json:"presets"`
		} `json:"data"`
	}
	if json.Unmarshal(data, &models) != nil {
		return ModelCard{}, errors.New("the models listing is not a model list")
	}
	for _, entry := range models.Data {
		if entry.ID == d.model && len(entry.QuestionTypes) > 0 {
			return RemoteCard(d.model, entry.QuestionTypes, entry.Presets), nil
		}
	}
	return ModelCard{}, fmt.Errorf("the models listing gives no question types for %q", d.model)
}

// reportFailure logs each kind of failure once per deployment and generation;
// the signals it leaves unknown carry the reason on every request. Neither the
// credential nor the API's answer is logged.
func (d *systemOneDeployment) reportFailure(status int, reason string, err error) {
	if _, seen := d.failures.LoadOrStore(fmt.Sprintf("%d\x00%s", status, reason), struct{}{}); seen {
		return
	}
	fields := map[string]interface{}{"deployment": d.name, "status": status, "reason": reason}
	if err != nil {
		fields["error"] = err.Error()
	}
	logging.ComponentWarnEvent("model_runtime", "systemone_deployment_failed", fields)
}

// RemoteCard is the card of a remote decision model: it answers questionTypes
// on the decisions surface, the System One types (choice, noul and score)
// when there are none.
func RemoteCard(id string, questionTypes, presets []string) ModelCard {
	return ModelCard{
		ID: id, Surfaces: []string{"decisions"}, Ready: true, Status: "ready",
		QuestionTypes: append([]string(nil), questionTypes...), Presets: append([]string(nil), presets...),
	}
}

// EncodeSystemOneRequest is request as a System One API reads it: each
// question's ordered choices and levels as System One criteria (an object of
// option keys to descriptions, a list of level descriptions; labels already
// go as one), and the context's deadline as options.deadline_ms in whole
// milliseconds. A System One API reads one bounded input, so neither a scan
// budget nor truncation goes.
func EncodeSystemOneRequest(ctx context.Context, request Request) ([]byte, error) {
	request = boundedRead(request)
	body, err := encodeDecisionRequest(ctx, request)
	if err != nil {
		return nil, err
	}
	if deadline := body.Options.DeadlineMs; deadline != nil {
		whole := math.Max(1, math.Floor(*deadline))
		body.Options.DeadlineMs = &whole
	} else {
		body.Options = nil
	}
	for _, question := range request.Questions {
		body.Questions[question.ID] = systemOneQuestion(question)
	}
	return json.Marshal(body)
}

func systemOneQuestion(question Question) api.Question {
	encoded := encodeQuestion(question)
	if question.Preset != "" {
		return encoded
	}
	var criteria interface{}
	switch {
	case len(question.Choices) > 0:
		criteria = labelCriteria(question.Choices)
	case len(question.Levels) > 0:
		levels := make([]interface{}, len(question.Levels))
		for index, level := range question.Levels {
			levels[index] = level
		}
		criteria = levels
	default:
		return encoded
	}
	encoded.Choices, encoded.Levels, encoded.Criteria = nil, nil, &criteria
	return encoded
}

// DecodeSystemOneResponse reads a System One API's answer to questions as a
// decisions answer. A System One API answers a request with a failed question
// as a whole, with an error status, so the call's questions share its error.
func DecodeSystemOneResponse(status int, body []byte, questions []Question) (Response, error) {
	if err := systemOneStatusError(status); err != nil {
		return Response{}, err
	}
	var decoded api.DecisionResponse
	if len(body) > SystemOneResponseLimit || json.Unmarshal(body, &decoded) != nil || decoded.Answers == nil {
		return Response{}, fmt.Errorf("%w: invalid System One response", ErrFailed)
	}
	return decodeResponse(decoded, questions), nil
}
