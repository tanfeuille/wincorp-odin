-- @spec specs/llm-factory.spec.md v1.3.3 §24.4
-- Phase 1.6b — Table llm_usage : persistance events TokenUsageEvent.
-- Schema pragmatique, colonnes alignees sur dataclass TokenUsageEvent.

CREATE TABLE IF NOT EXISTS llm_usage (
  id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  model_name TEXT NOT NULL,
  session_id TEXT,
  agent_name TEXT,
  client_id TEXT,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  total_tokens INTEGER NOT NULL,
  cost_eur NUMERIC(12, 6) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp ON llm_usage(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_model_name ON llm_usage(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_usage_client_id ON llm_usage(client_id) WHERE client_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_usage_session_id ON llm_usage(session_id) WHERE session_id IS NOT NULL;

-- Retention : optionnel, ajuster selon besoin
-- DELETE FROM llm_usage WHERE timestamp < NOW() - INTERVAL '90 days';
