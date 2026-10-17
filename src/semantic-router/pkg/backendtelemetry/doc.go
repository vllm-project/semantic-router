// Package backendtelemetry defines the engine-neutral contract for observing
// the serving replicas behind one provider model alias.
//
// An Adapter collects Observations from one serving engine (vLLM, SGLang,
// ATOM, TensorRT-LLM) off the request path. A Store keeps the latest
// Observation per replica and answers only with observations that are still
// fresh, so a consumer never mistakes old or missing evidence for current
// load.
//
// The contract has three rules:
//
//   - Absent is not zero. Every metric is a Value that is either observed or
//     absent; an adapter reports only what its engine exposed in that scrape.
//   - Freshness is explicit. An observation carries the time it was collected
//     and a TTL; once it expires it is never returned as fresh.
//   - Observations are values. They hold no pointers or maps, so a copy can be
//     stored, returned and mutated without aliasing the store.
//
// This package does not choose a replica. Replica selection belongs to the
// serving layer (the upstream load balancer), which may consult fresh
// observations and must fall back to its own policy when they are missing.
package backendtelemetry
