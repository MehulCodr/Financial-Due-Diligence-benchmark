# One Oxygen Sandbox

One Oxygen is being built in phases. The repository currently implements:

- Phase 1: a hardened local Docker sandbox runner.
- Phase 2: a provider-independent tool protocol for future LLM agent loops.

There are still no model SDKs, graders, RAG, database, web UI, Kubernetes support, financial
PDF libraries, spreadsheet libraries, or browser automation in this repo.

## Architecture

`SandboxSession` remains the Phase 1 execution boundary. It owns the run ID, temporary workspace,
Docker container, command execution, timeouts, artifact copying, run-record persistence, and
cleanup.

The Phase 2 tool protocol sits above `SandboxSession`:

- `ToolDefinition` exposes provider-neutral JSON schemas generated from typed Pydantic argument
  models.
- `ToolCall` is the normalized input a future model adapter can produce.
- `ToolDispatcher` validates calls, checks `ToolPolicy`, invokes a registered tool, normalizes
  failures, tracks submission state, and appends bounded trace events to `RunRecord`.
- `ToolResult` is the normalized model-facing response.
- `SecureWorkspace` is the only filesystem view given to file tools. It accepts relative
  `/workspace` paths, rejects traversal and symlinks, and never returns host paths.

Docker-specific behavior remains inside `docker_adapter.py`. Tools do not receive the Docker
client or arbitrary host filesystem access.

## Security Model

Phase 1 guarantees are preserved:

- one fresh temporary workspace per run;
- non-root numeric container user;
- disabled networking by default;
- read-only root filesystem;
- only `/workspace` mounted writable;
- in-memory hardened `/tmp`;
- all Linux capabilities dropped;
- `no-new-privileges`;
- CPU, memory, PID, command, and overall runtime limits;
- no Docker socket or repository mount;
- no provider API keys;
- bounded command output and artifact collection;
- cleanup on success, failure, interruption, and timeout.

Phase 2 adds:

- fail-closed tool policy defaults;
- stable machine-readable tool error codes;
- deterministic tool schemas;
- bounded tool arguments/results in trace records, plus SHA-256 hashes;
- protected workspace paths such as `.oneoxygen/`;
- atomic text writes and replacements;
- binary-file and symlink rejection for file tools;
- one successful `submit_result` per run;
- post-submission tool-call rejection.

Model-facing tool errors are sanitized. They do not expose host paths, stack traces, Docker
configuration, API keys, or temporary workspace locations.

## Available Tools

- `list_files`: list bounded workspace entries without following symlinks.
- `read_text_file`: read bounded UTF-8 text with optional line ranges.
- `write_text_file`: atomically write UTF-8 text.
- `replace_text`: atomically replace exact text only when the expected count matches.
- `execute_shell`: run a shell command inside the active sandbox container.
- `execute_python`: write trusted Python source to a protected runtime file, execute it inside
  the container, then remove it.
- `submit_result`: submit final summary, structured findings, and approved output artifacts.

`execute_shell` and `execute_python` are disabled unless the task policy explicitly enables them.

## Tool Policy

Existing Phase 1 task YAML files still work. `tool_policy` is optional and defaults to a safe
policy that allows file inspection/editing and submission, but not shell or Python execution.

Example:

```yaml
tool_policy:
  allowed_tool_names:
    - list_files
    - read_text_file
    - execute_python
    - write_text_file
    - submit_result
  max_total_tool_calls: 10
  per_tool_call_limits:
    execute_python: 1
    submit_result: 1
  max_read_size_bytes: 65536
  max_write_size_bytes: 65536
  max_file_list_entries: 100
  shell_timeout_seconds: 5
  python_timeout_seconds: 10
  max_tool_result_size_bytes: 32768
  shell_execution_allowed: false
  python_execution_allowed: true
  protected_workspace_paths:
    - .oneoxygen
    - .oneoxygen/tool-runtime
```

Policies are stored in `run.json` alongside the sandbox policy and tool-event traces.

## Normalized Messages

Example `ToolCall`:

```json
{
  "call_id": "call-3",
  "tool_name": "execute_python",
  "arguments": {
    "source_code": "print('hello from the sandbox')",
    "timeout_seconds": 5
  }
}
```

Example `ToolResult`:

```json
{
  "call_id": "call-3",
  "tool_name": "execute_python",
  "success": true,
  "content": {
    "stdout": "hello from the sandbox\n",
    "stderr": "",
    "exit_code": 0,
    "duration_seconds": 0.12,
    "timed_out": false,
    "output_truncated": false
  },
  "error": null,
  "truncated": false,
  "metadata": {
    "content_sha256": "..."
  }
}
```

Future model adapters should translate provider-specific tool-call formats into `ToolCall` and
translate `ToolResult` back to that provider. The core tool layer does not import or depend on any
provider SDK.

## CLI

Check Docker:

```text
python -m oneoxygen_sandbox doctor
```

Build and smoke-test the base image:

```text
python -m oneoxygen_sandbox build
```

Run the Phase 1 example:

```text
python -m oneoxygen_sandbox run examples/basic/task.yaml
```

List tools:

```text
python -m oneoxygen_sandbox tools list
```

List provider-neutral JSON definitions:

```text
python -m oneoxygen_sandbox tools list --json
```

Run the Phase 2 scripted tool demo:

```text
python -m oneoxygen_sandbox tool-demo examples/tool_demo/task.yaml
```

The demo uses `examples/tool_demo/scripted_calls.yaml` and a tiny synthetic CSV. It lists files,
reads `company_metrics.csv`, executes standard-library Python to calculate revenue growth and
gross margin, writes `output/findings.md`, reads it back, and submits it through `submit_result`.

## Setup

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m oneoxygen_sandbox doctor
python -m oneoxygen_sandbox build
```

Linux shell:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m oneoxygen_sandbox doctor
python -m oneoxygen_sandbox build
```

Docker Desktop on Windows must be running Linux containers. The application itself does not depend
on Bash scripts.

## Tests

Run fast tests:

```text
python -m pytest -m "not integration"
```

Run Docker integration tests:

```text
python -m pytest -m integration
```

Run everything:

```text
python -m pytest
```

Format and lint:

```text
python -m ruff format .
python -m ruff check .
```

## Run Records

Retained runs are written as:

```text
runs/<run-id>/
|-- run.json
`-- artifacts/
    `-- ...approved output files only...
```

`run.json` includes sandbox policy, resolved tool policy, command results, tool events,
submission metadata, final status, and copied artifact metadata. The full temporary workspace is
not retained.

## Limitations

- Linux containers only.
- Local single-host isolation, not a multi-tenant sandbox service.
- No network access policy beyond disabled networking.
- Tool execution requires a running `SandboxSession`.
- `execute_python` uses Python already available in the sandbox image and no third-party
  libraries.
- Artifact collection still copies all approved files from the configured output directory.
- Retained run records and artifacts are not encrypted or signed.
- Base image provenance is tag-based; teams that need bit-for-bit provenance should mirror and
  digest-pin the base image.

## Phase 3 Preview

Phase 3 can add provider adapters that convert model-specific tool calls into the normalized
`ToolCall` format, then feed `ToolResult` objects back to the provider. That future work may also
add benchmark orchestration and evaluation, but those are intentionally outside Phase 2.
