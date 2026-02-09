/**
 * Shared fake container status data for demos
 */

import { State, CumulativeUsage } from "../../../types";

export const sampleUsage: CumulativeUsage = {
  start_time: "2024-01-15T10:00:00Z",
  messages: 1337,
  input_tokens: 25432,
  output_tokens: 18765,
  cache_read_input_tokens: 8234,
  cache_creation_input_tokens: 12354,
  total_cost_usd: 2.03,
  tool_uses: {
    bash: 45,
    patch: 23,
    think: 12,
    keyword_search: 6,
  },
};

export const sampleContainerState: State = {
  state_version: 1,
  message_count: 27,
  total_usage: sampleUsage,
  initial_commit: "decafbad42abc123",
  slug: "file-upload-component",
  branch_name: "sketch-wip",
  branch_prefix: "sketch",
  hostname: "example.hostname",
  working_dir: "/app",
  os: "linux",
  git_origin: "https://github.com/user/repo.git",
  outstanding_llm_calls: 0,
  outstanding_tool_calls: null,
  session_id: "session-abc123",
  ssh_available: true,
  in_container: true,
  first_message_index: 0,
  agent_state: "ready",
  outside_hostname: "host.example.com",
  inside_hostname: "container.local",
  outside_os: "macOS",
  inside_os: "linux",
  outside_working_dir: "/Users/dev/project",
  inside_working_dir: "/app",
  todo_content:
    "- Implement file upload component\n- Add drag and drop support\n- Write tests",
  skaband_addr: "localhost:8080",
  link_to_github: true,
  ssh_connection_string: "ssh user@example.com",
  diff_lines_added: 245,
  diff_lines_removed: 67,
  plan_mode: false,
};

export const lightUsageState: State = {
  ...sampleContainerState,
  message_count: 5,
  total_usage: {
    ...sampleUsage,
    messages: 5,
    input_tokens: 1234,
    output_tokens: 890,
    total_cost_usd: 0.15,
    tool_uses: {
      bash: 2,
      patch: 1,
    },
  },
  diff_lines_added: 45,
  diff_lines_removed: 12,
};

export const heavyUsageState: State = {
  ...sampleContainerState,
  message_count: 156,
  total_usage: {
    ...sampleUsage,
    messages: 156,
    input_tokens: 89234,
    output_tokens: 67890,
    cache_read_input_tokens: 23456,
    cache_creation_input_tokens: 45678,
    total_cost_usd: 12.45,
    tool_uses: {
      bash: 234,
      patch: 89,
      think: 67,
      keyword_search: 45,
      browser_navigate: 12,
      codereview: 8,
    },
  },
  diff_lines_added: 2847,
  diff_lines_removed: 1456,
};
