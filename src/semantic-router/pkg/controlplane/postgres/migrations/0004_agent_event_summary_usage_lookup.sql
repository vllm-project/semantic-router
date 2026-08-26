-- Model-step summaries are durable worker events. Replace the baseline checks
-- with explicitly named constraints so subsequent changes remain forward-only.
ALTER TABLE agent_events
  DROP CONSTRAINT agent_events_event_type_check,
  DROP CONSTRAINT agent_events_check1;

ALTER TABLE agent_events
  ADD CONSTRAINT agent_events_event_type_check CHECK (event_type IN (
    'user_input','assistant_delta','model_step_summary','tool_request','tool_result','progress',
    'context_checkpoint','approval_request','approval_result','cancellation','terminal'
  )) NOT VALID,
  ADD CONSTRAINT agent_events_worker_event_type_ck CHECK (
    origin <> 'worker' OR event_type IN (
      'assistant_delta','model_step_summary','tool_request','tool_result','progress',
      'context_checkpoint','approval_request','terminal'
    )
  ) NOT VALID;

ALTER TABLE agent_events
  VALIDATE CONSTRAINT agent_events_event_type_check;
ALTER TABLE agent_events
  VALIDATE CONSTRAINT agent_events_worker_event_type_ck;

-- Request-log details resolve an external request within one namespace.
CREATE INDEX usage_events_external_request_idx
  ON usage_events(namespace_id, external_request_id, occurred_at DESC, event_id)
  WHERE external_request_id IS NOT NULL;
