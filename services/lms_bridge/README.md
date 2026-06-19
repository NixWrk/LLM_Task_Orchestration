# lms bridge

Host-side HTTP bridge over the local `lms` (LM Studio) CLI, plus an in-container
`lms` shim. Together they let the **containerized** lifecycle service load LM
Studio models with the profile's context length, even though `lms` is a
host-bound binary that the Linux container cannot execute directly.

## Why

`lms` controls the local LM Studio app over a private channel and has no
`--host` option, so it cannot be run from inside the Docker container. Without
this bridge, lifecycle's `cli-if-available` strategy hits `FileNotFoundError`,
skips the load, and LM Studio JIT-loads the model at its default context (e.g.
4096), ignoring `lms_context_length` in the model profile.

## How it works

- `lms_bridge.py` runs on the **host** and executes whitelisted `lms`
  subcommands (`load`, `unload`, `ps`, ...) on request, returning
  `{returncode, stdout, stderr}`.
- `lms-shim` is mounted into the lifecycle container as `/usr/local/bin/lms`.
  The orchestrator's unchanged `subprocess.run(["lms", ...])` calls hit this
  shim, which forwards them to the host bridge and passes stdout/stderr/exit
  code straight through.

The orchestrator load logic is **not** modified: it still runs
`lms load --context-length <N> --parallel <M>`; the shim just makes that reach
the host LM Studio.

## Run the bridge (host)

```powershell
$env:LMS_BINARY="C:\Users\<you>\.lmstudio\bin\lms.exe"
python services\lms_bridge\lms_bridge.py --host 0.0.0.0 --port 4399
```

Keep it running (e.g. a Task Scheduler task at logon). Health check:
`GET http://127.0.0.1:4399/health`.

## Compose wiring

`docker-compose.yml` (lifecycle service):

- mounts `./services/lms_bridge/lms-shim` at `/usr/local/bin/lms`;
- sets `LMS_BRIDGE_URL=http://host.docker.internal:4399`;
- defaults `LIFECYCLE_DRY_RUN=false` so loads actually execute.

## Verify

A cold allocation should load with the profile context:

```
POST http://127.0.0.1:4300/allocations  {"model":"zotero-html-translate"}
lms ps --json   ->  contextLength: 32768, parallel: 2
```
