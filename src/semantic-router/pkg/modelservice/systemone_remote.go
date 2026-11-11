package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// listingTimeout bounds one read of a remote models listing beside requests;
// preparationListingTimeout bounds the read a generation's preparation waits
// for, so an API that does not list never holds a generation back long.
const (
	listingTimeout            = 10 * time.Second
	preparationListingTimeout = 5 * time.Second
)

const listingLimit = 1 << 20

// systemOneRequestLimit bounds one System One request body unless the
// deployment's limits set another.
const systemOneRequestLimit = 16 << 20

// remoteCacheTTL is how long a remote answer serves a repeated call. The
// cache belongs to one configuration generation and is dropped with it.
const remoteCacheTTL = 10 * time.Minute

// Remote answers decision questions with deployments the model runtime does
// not serve: systemone deployments, which a remote System One API serves.
type Remote interface {
	Decider
	// Card is the model card questions to name compile against; ok is false
	// for a name Remote does not serve.
	Card(name string) (card ModelCard, ok bool)
	// PrepareCard is the card a generation's preparation checks name's
	// consumers against, its models listing read now if it can be.
	PrepareCard(ctx context.Context, name string) (ModelCard, error)
}

// RemoteSystemOne asks the systemone deployments of one configuration their
// decision questions over their System One APIs. It is a signal provider
// beside the model runtime and never serves a request's model.
//
// In a request stage, every question to one deployment travels in one call
// (see Bundle), and a call answers a repeated question from the generation's
// result cache.
type RemoteSystemOne struct {
	deployments map[string]*systemOneDeployment
	ctx         context.Context
	cancel      context.CancelFunc
}

type systemOneDeployment struct {
	name       string
	model      string
	declared   []string
	presets    []string
	modalities []string
	client     *connector.Client
	cache      *resultCache
	started    atomic.Bool
	prepared   sync.Once
	listed     atomic.Pointer[ModelCard]
	failures   sync.Map
	// codec and deviations speak the deployment's schema; a schema without
	// batches splits a stage's states into calls of their own, at most
	// maxConcurrency in flight. lister reads the listing when it is not
	// under the endpoint.
	codec          systemonecodec.SchemaCodec
	deviations     []systemonecodec.Deviation
	maxConcurrency int
	lister         *connector.Client
}

// native reports whether the deployment speaks vllm-sr/4.0, whose calls carry
// a fused task's internal request as it is.
func (d *systemOneDeployment) native() bool {
	return d.codec == nil || d.codec.ID() == systemonecodec.SchemaVLLMSR40
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
		// The connector refuses redirects, which would carry the credential
		// header to another host. A call's deadline is its caller's; the
		// attempt bound only caps it.
		transport := deployment
		transport.TimeoutMs = 0
		client, err := connector.ForDeployment(transport, connector.Options{
			AttemptTimeout: time.Duration(config.MaxSystemOneTimeoutMs) * time.Millisecond, MaxRetries: 0,
			MaxRequestBytes: systemOneRequestLimit, MaxResponseBytes: SystemOneResponseLimit, MaxErrorBytes: SystemOneResponseLimit,
		})
		if err != nil {
			cancel()
			return nil, fmt.Errorf("systemone deployment %q: %w", name, err)
		}
		// A delegated deployment answers each user with that user's
		// credential, so no answer is kept for another request.
		var cache *resultCache
		if !deployment.Auth.Delegated() {
			cache = newExpiringResultCache(ResultCacheEntries(), remoteCacheTTL)
		}
		codec, deviations, err := systemonecodec.Builtin().Resolve(systemonecodec.SchemaID(deployment.Schema), deployment.Deviations)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("systemone deployment %q: %w", name, err)
		}
		entry := &systemOneDeployment{
			name: name, model: deployment.Model,
			declared: deployment.DeclaredQuestionTypes(), presets: deployment.DeclaredPresets(),
			modalities: deployment.DeclaredModalities(), client: client, cache: cache,
			codec: codec, deviations: deviations, maxConcurrency: deployment.EffectiveMaxConcurrency(),
		}
		if path := codec.ListingPath(); !strings.HasPrefix(path, "/") {
			sibling := transport
			sibling.Endpoint = siblingEndpoint(deployment.Endpoint, path)
			if entry.lister, err = connector.ForDeployment(sibling, connector.Options{
				AttemptTimeout: listingTimeout, MaxRetries: 0, MaxRequestBytes: systemOneRequestLimit,
				MaxResponseBytes: SystemOneResponseLimit, MaxErrorBytes: SystemOneResponseLimit,
			}); err != nil {
				cancel()
				return nil, fmt.Errorf("systemone deployment %q: %w", name, err)
			}
		}
		remote.deployments[name] = entry
	}
	return remote, nil
}

// siblingEndpoint is the endpoint with its last path segment replaced, as a
// relative reference resolves: POST /v1/systemone lists at GET /v1/models.
func siblingEndpoint(endpoint, segment string) string {
	base, err := url.Parse(endpoint)
	if err != nil {
		return endpoint
	}
	return base.ResolveReference(&url.URL{Path: segment, RawQuery: base.RawQuery}).String()
}

// Card is the card name's questions compile against: the question types and
// presets its API lists for the model in GET <endpoint>/models, else the
// declared ones, else the System One types; and the modalities it lists, else
// the declared ones, else text. The first call reads the listing in the
// background unless a preparation has read it; until it answers, the declared
// or System One types apply.
func (r *RemoteSystemOne) Card(name string) (ModelCard, bool) {
	d := r.deployment(name)
	if d == nil {
		return ModelCard{}, false
	}
	if card := d.listed.Load(); card != nil {
		return *card, true
	}
	if d.started.CompareAndSwap(false, true) {
		go d.readListing(r.ctx, listingTimeout)
	}
	return d.declaredCard(), true
}

// ListedCard is the card name's API lists for its model, when the listing has
// answered; like Card, the first call reads the listing in the background.
func (r *RemoteSystemOne) ListedCard(name string) (ModelCard, bool) {
	d := r.deployment(name)
	if d == nil {
		return ModelCard{}, false
	}
	if card := d.listed.Load(); card != nil {
		return *card, true
	}
	if d.started.CompareAndSwap(false, true) {
		go d.readListing(r.ctx, listingTimeout)
	}
	return ModelCard{}, false
}

// PrepareCard reads name's listing once, for at most
// preparationListingTimeout, and returns its card as Card does. An API that
// does not list leaves the declared or System One types; it never fails the
// preparation, and Card still reads the listing once in the background.
func (r *RemoteSystemOne) PrepareCard(ctx context.Context, name string) (ModelCard, error) {
	d := r.deployment(name)
	if d == nil {
		return ModelCard{}, ErrUnknownDeployment
	}
	d.prepared.Do(func() {
		if d.listed.Load() == nil {
			d.readListing(ctx, preparationListingTimeout)
		}
	})
	if card := d.listed.Load(); card != nil {
		return *card, nil
	}
	return d.declaredCard(), nil
}

// Forward sends a native System One request a provider alias publishes to
// name, as the API's own model, and returns the API's answer: the native
// API's public alias forwards to the remote deployment it names. A target of
// another schema gets the request converted to its schema, and its answers
// and errors come back in the native form, a batch reassembled.
func (r *RemoteSystemOne) Forward(ctx context.Context, name string, body json.RawMessage) (SystemOneResult, error) {
	d := r.deployment(name)
	if d == nil {
		return SystemOneResult{}, ErrUnavailable
	}
	if !d.native() || (d.codec != nil && systemonecodec.HasRoles(body)) {
		return d.forwardCodec(ctx, body)
	}
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return SystemOneResult{}, fmt.Errorf("%w: native request: %w", ErrRejected, err)
	}
	model, _ := json.Marshal(d.model)
	request["model"] = model
	encoded, err := json.Marshal(request)
	if err != nil {
		return SystemOneResult{}, err
	}
	status, answer, err := d.exchange(ctx, http.MethodPost, "", encoded)
	if err != nil {
		return SystemOneResult{}, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return SystemOneResult{Status: status, Body: answer}, nil
}

// Serves reports whether name is one of the deployments r asks.
func (r *RemoteSystemOne) Serves(name string) bool {
	return r.deployment(name) != nil
}

// Empty reports whether the configuration declares no systemone deployment.
func (r *RemoteSystemOne) Empty() bool {
	return r == nil || len(r.deployments) == 0
}

// Decide asks name the questions of request. In a request stage the call
// parks in the stage's bundle and travels with the stage's other questions
// to name; alone it is one System One call. An answer the generation's cache
// holds is not asked again.
func (r *RemoteSystemOne) Decide(ctx context.Context, name string, request Request) (Response, error) {
	d := r.deployment(name)
	if d == nil {
		requestsTotal.WithLabelValues(name, ErrorReason(ErrUnknownDeployment)).Inc()
		return Response{}, ErrUnknownDeployment
	}
	request.Model = d.model
	request = boundedRead(request)
	body, err := systemOneBody(ctx, request)
	if err != nil {
		return Response{}, err
	}
	decision := &decisionCall{request: request, cache: d.cache.active(), deployment: name}
	if bundle := bundleFrom(ctx); bundle != nil {
		return bundle.decideRemote(ctx, d, decision, body)
	}
	call := &bundleCall{ctx: ctx, task: api.BundleTask{Decisions: &body}, done: make(chan struct{}), decision: decision}
	for _, task := range answerCached(fuseDecisions(1, []*bundleCall{call})) {
		d.send(ctx, task)
	}
	<-call.done
	return decision.response, call.err
}

// ResetCache drops every cached answer, for a generation that takes over
// this client from the one before it.
func (r *RemoteSystemOne) ResetCache() {
	if r == nil {
		return
	}
	for _, d := range r.deployments {
		d.cache.reset()
	}
}

// Close stops listing reads and drops idle connections.
func (r *RemoteSystemOne) Close() error {
	if r == nil {
		return nil
	}
	r.cancel()
	for _, d := range r.deployments {
		_ = d.client.Close()
	}
	return nil
}

func (r *RemoteSystemOne) deployment(name string) *systemOneDeployment {
	if r == nil {
		return nil
	}
	return r.deployments[name]
}

func (d *systemOneDeployment) declaredCard() ModelCard {
	card := SchemaCard(d.model, d.schemaID(), d.declared, d.presets)
	card.Modalities = slices.Clone(d.modalities)
	return card
}

func (d *systemOneDeployment) schemaID() systemonecodec.SchemaID {
	if d.codec == nil {
		return systemonecodec.SchemaVLLMSR50
	}
	return d.codec.ID()
}

// send asks a fused task's questions: as the internal request it is to a
// vllm-sr/4.0 deployment (sendBatch), else through the deployment's codec
// (sendCodec).
func (d *systemOneDeployment) send(ctx context.Context, task *bundleTask) {
	if !d.native() {
		d.sendCodec(ctx, task)
		return
	}
	d.sendBatch(ctx, task)
}

// sendBatch asks a fused task's questions in one System One call. A System
// One API fails a call with a failed question as a whole and names the
// question and, in a batch, its entry; sendBatch then asks the rest once more
// without it, within the same deadline, and leaves only that question unknown
// (about that entry, else about every state). An entry whose input the API
// refused (invalid_input) is left out as a whole. When the retry fails the
// same question about another entry, sendBatch asks once more without it
// about every state. Any other failure leaves every question of the task
// unknown.
func (d *systemOneDeployment) sendBatch(ctx context.Context, task *bundleTask) {
	plan := sentAs(task)
	status, data, err := d.post(ctx, *task.task.Decisions)
	var failures []questionFailure
	for err == nil {
		failed, ok := failedQuestion(status, data)
		if len(failures) > 0 {
			// At most one more retry, and only for the same question.
			last := failures[len(failures)-1]
			ok = ok && last.state != "" && failed.question == last.question
			failed.state = ""
		}
		if !ok || ctx.Err() != nil {
			break
		}
		retry, ok := plan.without(failed)
		if !ok {
			break
		}
		systemOneRetries.WithLabelValues(d.name).Inc()
		logging.ComponentDebugEvent("model_runtime", "systemone_question_retried", map[string]interface{}{
			"deployment": d.name, "question": failed.question, "state": failed.state, "question_error": failed.code,
		})
		failures, plan = append(failures, failed), retry
		status, data, err = d.post(ctx, plan.body)
	}
	var responses []Response
	if err == nil {
		responses, err = plan.decode(task, status, data)
		if err != nil {
			d.reportFailure(status, ErrorReason(err), nil)
		}
	}
	if err == nil {
		for _, failure := range failures {
			failure.answer(task, responses)
		}
	}
	if err == nil && task.decision.cache != nil {
		for _, call := range task.calls {
			cacheTotal.WithLabelValues(call.decision.deployment, "miss").Inc()
		}
		if len(failures) == 0 && completeAll(responses) {
			if len(responses) == 1 {
				task.decision.cache.put(task.decision.key, responses[0])
			} else {
				task.decision.cache.put(task.decision.key, responses)
			}
		}
	}
	task.deliver(responses, err)
}

// post makes one System One call with body, its deadline the time ctx has
// left, and returns the API's status and answer; err is set when there is
// none.
func (d *systemOneDeployment) post(ctx context.Context, body api.DecisionRequest) (int, []byte, error) {
	started := time.Now()
	status, data, err := d.exchangeDecisions(ctx, body)
	outcome := err
	if outcome == nil {
		outcome = systemOneStatusError(status)
	}
	requestDuration.WithLabelValues(d.name, "decisions").Observe(time.Since(started).Seconds())
	requestsTotal.WithLabelValues(d.name, ErrorReason(outcome)).Inc()
	return status, data, err
}

func (d *systemOneDeployment) exchangeDecisions(ctx context.Context, body api.DecisionRequest) (int, []byte, error) {
	if deadline, ok := ctx.Deadline(); ok {
		remaining := math.Floor(float64(time.Until(deadline)) / float64(time.Millisecond))
		if remaining < 1 {
			<-ctx.Done()
			return 0, nil, ctx.Err()
		}
		options := api.RequestOptions{}
		if body.Options != nil {
			options = *body.Options
		}
		options.DeadlineMs = &remaining
		body.Options = &options
	}
	encoded, err := marshalSystemOne(body)
	if err != nil {
		return 0, nil, err
	}
	status, data, err := d.exchange(ctx, http.MethodPost, "", encoded)
	if ctxErr := ctx.Err(); ctxErr != nil {
		return 0, nil, ctxErr
	}
	if err != nil {
		return 0, nil, fmt.Errorf("%w: systemone deployment %q: %w", ErrFailed, d.name, err)
	}
	return status, data, nil
}

// systemOneOperation is a call to a System One API: POST <endpoint> asks
// questions, GET <endpoint>/models lists the models.
var systemOneOperation = connector.Operation{Name: "systemone", RetrySafe: true}

// exchange sends one request and returns the API's status and answer, an
// error status's included: a System One API names a failed question in it.
// A transport failure is reported once and never carries the request URL,
// whose query may hold a gateway's parameters.
func (d *systemOneDeployment) exchange(ctx context.Context, method, path string, body []byte) (int, []byte, error) {
	operation := systemOneOperation
	operation.Method, operation.Path = method, path
	result, err := d.client.DoRequest(ctx, operation, connector.Request{Body: body})
	if err == nil {
		return result.StatusCode, result.Body, nil
	}
	var remote *connector.Error
	if errors.As(err, &remote) {
		switch remote.Kind {
		case connector.KindStatus:
			data, _ := remote.ResponseBody()
			return remote.StatusCode, data, nil
		case connector.KindAuthorization:
			d.reportFailure(0, "credential", remote.Cause)
			return 0, nil, remote.Cause
		}
		cause := remote.Cause
		if cause == nil {
			cause = remote
		}
		var urlErr *url.Error
		if errors.As(cause, &urlErr) {
			cause = urlErr.Err
		}
		d.reportFailure(remote.StatusCode, "transport", cause)
		return 0, nil, cause
	}
	return 0, nil, err
}

func (d *systemOneDeployment) readListing(parent context.Context, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()
	if !d.codec.Capabilities().Has(systemonecodec.CapabilityCard) {
		d.checkListed(ctx)
		return
	}
	card, err := d.listing(ctx)
	if errors.Is(err, principal.ErrRouterCall) {
		// The listing is the Router's own read; a delegated deployment
		// answers it for no one, so its declared card applies.
		logging.ComponentWarnEvent("model_runtime", "delegated_router_call_refused", map[string]interface{}{
			"call": "systemone_listing", "deployment": d.name, "question_types": d.declaredCard().QuestionTypes,
		})
		return
	}
	if err != nil {
		logging.ComponentEvent("model_runtime", "systemone_deployment_unlisted", map[string]interface{}{
			"deployment": d.name, "reason": err.Error(), "question_types": d.declaredCard().QuestionTypes,
		})
		return
	}
	d.listed.Store(&card)
}

func (d *systemOneDeployment) listing(ctx context.Context) (ModelCard, error) {
	if d.lister != nil {
		return d.listedCard(ctx)
	}
	status, data, err := d.exchange(ctx, http.MethodGet, "/models", nil)
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
			Modalities    []string `json:"modalities"`
		} `json:"data"`
	}
	if json.Unmarshal(data, &models) != nil {
		return ModelCard{}, errors.New("the models listing is not a model list")
	}
	for _, entry := range models.Data {
		if entry.ID != d.model {
			continue
		}
		if card, ok := d.listedEntryCard(entry.QuestionTypes, entry.Presets, entry.Modalities); ok {
			return card, nil
		}
	}
	return ModelCard{}, fmt.Errorf("the models listing gives no question types or modalities for %q", d.model)
}

// listedEntryCard is the card a listing entry for the deployment's model
// gives: the question types and presets it lists, else the declared ones, and
// the modalities it lists, else the declared ones. An entry that lists
// neither question types nor modalities gives none.
func (d *systemOneDeployment) listedEntryCard(questionTypes, presets, modalities []string) (ModelCard, bool) {
	if len(questionTypes) == 0 && modalities == nil {
		return ModelCard{}, false
	}
	card := d.declaredCard()
	if len(questionTypes) > 0 {
		card.QuestionTypes, card.Presets = questionTypes, presets
	}
	if modalities != nil {
		// A System One request carries no audio part.
		card.Modalities = slices.DeleteFunc(modalities, func(modality string) bool { return modality == config.SystemOneModalityAudio })
	}
	return card, true
}

// listedCard reads the card of the deployment's model from a listing beside
// its endpoint: vllm-sr/5.0's GET /v1/models, whose models carry cards.
func (d *systemOneDeployment) listedCard(ctx context.Context) (ModelCard, error) {
	result, err := d.lister.DoRequest(ctx, connector.Operation{Name: "systemone", Method: http.MethodGet, RetrySafe: true}, connector.Request{})
	status := result.StatusCode
	var remote *connector.Error
	if errors.As(err, &remote) && remote.Kind == connector.KindStatus {
		status, err = remote.StatusCode, nil
	}
	if err != nil {
		return ModelCard{}, err
	}
	listing, err := d.codec.DecodeListing(status, result.Body)
	if err != nil {
		return ModelCard{}, err
	}
	for _, model := range listing.Models {
		if model.Name != d.model {
			continue
		}
		data, ok := systemonecodec.ListedCardJSON(model)
		if !ok {
			break
		}
		var card struct {
			QuestionTypes []string `json:"question_types"`
			Presets       []string `json:"presets"`
			Modalities    []string `json:"modalities"`
		}
		if json.Unmarshal(data, &card) == nil {
			if listed, ok := d.listedEntryCard(card.QuestionTypes, card.Presets, card.Modalities); ok {
				listed.MaxInputTokens, listed.ValidatedInputTokens = listedInputLimits(data)
				return listed, nil
			}
		}
	}
	return ModelCard{}, fmt.Errorf("the models listing gives no question types or modalities for %q", d.model)
}

// listedInputLimits are the question token caps a listed card's limits give:
// max_input_tokens and validated_input_tokens, zero when absent or not a
// positive whole number.
func listedInputLimits(data json.RawMessage) (maxInput, validated int) {
	var card struct {
		Limits struct {
			MaxInputTokens       json.Number `json:"max_input_tokens"`
			ValidatedInputTokens json.Number `json:"validated_input_tokens"`
		} `json:"limits"`
	}
	if json.Unmarshal(data, &card) != nil {
		return 0, 0
	}
	whole := func(number json.Number) int {
		value, err := number.Int64()
		if err != nil || value < 1 || value > math.MaxInt32 {
			return 0
		}
		return int(value)
	}
	return whole(card.Limits.MaxInputTokens), whole(card.Limits.ValidatedInputTokens)
}

// checkListed reads the listing of a schema without model cards, whose
// declared capabilities always apply, and logs when it does not list the
// deployment's model.
func (d *systemOneDeployment) checkListed(ctx context.Context) {
	reason := ""
	result, err := d.lister.DoRequest(ctx, connector.Operation{Name: "systemone", Method: http.MethodGet, RetrySafe: true}, connector.Request{})
	status := result.StatusCode
	var remote *connector.Error
	if errors.As(err, &remote) && remote.Kind == connector.KindStatus {
		status, err = remote.StatusCode, nil
	}
	if err != nil {
		reason = "the models listing could not be read"
	} else if listing, decodeErr := d.codec.DecodeListing(status, result.Body); decodeErr != nil {
		reason = decodeErr.Error()
	} else if !slices.ContainsFunc(listing.Models, func(model systemoneprotocol.ListedModel) bool { return model.Name == d.model }) {
		reason = fmt.Sprintf("the models listing does not list %q", d.model)
	}
	if reason != "" {
		logging.ComponentEvent("model_runtime", "systemone_deployment_unlisted", map[string]interface{}{
			"deployment": d.name, "schema": string(d.schemaID()), "reason": reason, "question_types": d.declaredCard().QuestionTypes,
		})
	}
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

// RemoteDevice is the device and precision a remote decision model's card
// reports: the model runs behind its API, not on a device of this host.
const RemoteDevice = "remote"

// RemoteCard is the card of a remote decision model: it answers questionTypes
// on the decisions surface, the System One types (choice, noul and score)
// when there are none, and defines presets.
func RemoteCard(id string, questionTypes, presets []string) ModelCard {
	return ModelCard{
		ID: id, Surfaces: []string{"decisions"}, Ready: true, Status: "ready", Device: RemoteDevice, Dtype: RemoteDevice,
		QuestionTypes: slices.Clone(questionTypes), Presets: slices.Clone(presets),
	}
}

// SchemaCard is the declared card of a remote model whose API speaks schema.
// A schema that carries no presets or no proof of full input coverage gives
// its card none; one without set or span questions answers the standard
// types, listed so a generation checks its rules against them as it
// prepares.
func SchemaCard(id string, schema systemonecodec.SchemaID, questionTypes, presets []string) ModelCard {
	card := RemoteCard(id, questionTypes, presets)
	card.Schema = string(schema)
	codec, _, err := systemonecodec.Builtin().Resolve(schema, nil)
	if err != nil {
		return card
	}
	capabilities := codec.Capabilities()
	if !capabilities.Has(systemonecodec.CapabilityPresets) {
		card.Presets = nil
	}
	if !capabilities.Has(systemonecodec.CapabilitySpan) && len(card.QuestionTypes) == 0 {
		card.QuestionTypes = []string{systemoneprotocol.QuestionChoice, systemoneprotocol.QuestionNoul, systemoneprotocol.QuestionScore}
	}
	return card
}

// DeploymentCard is the declared card of a systemone deployment, with the
// input modalities it declares.
func DeploymentCard(deployment config.ModelDeployment) ModelCard {
	card := SchemaCard(deployment.Model, systemonecodec.SchemaID(deployment.EffectiveSchema()), deployment.DeclaredQuestionTypes(), deployment.DeclaredPresets())
	card.Modalities = deployment.DeclaredModalities()
	return card
}

// EncodeSystemOneRequest is request as a System One API reads it: each
// question's ordered choices and levels as System One criteria (an object of
// option keys to descriptions, a list of level descriptions; labels already
// go as one), and the context's deadline as options.deadline_ms in whole
// milliseconds. A System One API reads one bounded input, so neither a scan
// budget nor truncation goes.
func EncodeSystemOneRequest(ctx context.Context, request Request) ([]byte, error) {
	body, err := systemOneBody(ctx, boundedRead(request))
	if err != nil {
		return nil, err
	}
	return marshalSystemOne(body)
}

func systemOneBody(ctx context.Context, request Request) (api.DecisionRequest, error) {
	body, err := encodeDecisionRequest(ctx, request)
	if err != nil {
		return api.DecisionRequest{}, err
	}
	if deadline := body.Options.DeadlineMs; deadline != nil {
		whole := math.Max(1, math.Floor(*deadline))
		body.Options.DeadlineMs = &whole
	}
	for _, question := range request.Questions {
		body.Questions[question.ID] = systemOneQuestion(question)
	}
	return body, nil
}

// marshalSystemOne encodes body without empty options, which a fused call
// whose callers set no deadline carries.
func marshalSystemOne(body api.DecisionRequest) ([]byte, error) {
	if body.Options != nil && reflect.ValueOf(*body.Options).IsZero() {
		body.Options = nil
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
	decoded, err := decodeSystemOneBody(status, body, false)
	if err != nil {
		return Response{}, err
	}
	return decodeResponse(decoded, questions), nil
}

// decodeSystemOneBody reads a System One API's answer to a standard request,
// or with batch to a batch, whose answers are its entries'. The connector
// bounds body by the deployment's limits.max_response_bytes.
func decodeSystemOneBody(status int, body []byte, batch bool) (api.DecisionResponse, error) {
	if err := systemOneStatusError(status); err != nil {
		return api.DecisionResponse{}, err
	}
	var decoded api.DecisionResponse
	invalid := json.Unmarshal(body, &decoded) != nil
	if batch {
		invalid = invalid || decoded.States == nil
	} else {
		invalid = invalid || decoded.Answers == nil
	}
	if invalid {
		return api.DecisionResponse{}, fmt.Errorf("%w: invalid System One response", ErrFailed)
	}
	return decoded, nil
}
