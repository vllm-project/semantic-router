// Package gateway is the Router's native HTTP frontend: it serves the
// OpenAI-compatible API itself, runs the routing core in process, and proxies
// to model backends through the upstream layer, doing what the local Envoy
// template and its ext_proc filter do in Envoy mode.
//
// A request is built the way Envoy's HTTP connection manager hands it to
// ext_proc, client API keys are checked the way the template's Lua filter
// checks them, and every request then goes through routing.Engine: the
// planned call goes upstream, the upstream response goes back through the
// engine, and the result streams to the client.
//
// This package imports no Envoy types and no pkg/extproc; the composition
// root binds it to an engine.
package gateway
