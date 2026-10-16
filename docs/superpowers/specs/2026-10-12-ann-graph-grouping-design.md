# ANN Graph Grouping (Postgres) — Design

## Goal

Scale similar-face grouping beyond the previous on-demand O(n²) `GET /v1/faces/groups` path (hard-capped at 5k faces) so republic-scale enrollments (~200k) can be clustered.

## Decisions

- **Async rebuild job** — `POST /v1/faces/groups/rebuild` starts work; clients poll status.
- **ANN graph** — for each face, take Qdrant/store top-K neighbors with score ≥ threshold, union-find connected components, drop groups smaller than `min_size`.
- **Postgres storage** — `group_runs`, `groups`, `group_members` on the **host OS Postgres** (not a Compose service). Qdrant remains vectors-only (no `group_id` payload). Compose API reaches it via `host.docker.internal`.
- **Read path** — `GET /v1/faces/groups` returns the latest **completed** run. No completed run → empty `groups`.

## API

| Method | Path | Notes |
|--------|------|-------|
| POST | `/v1/faces/groups/rebuild` | `202` + `run_id`; `409` if already running |
| GET | `/v1/faces/groups/rebuild` | list runs, newest `started_at` first |
| GET | `/v1/faces/groups/rebuild/{run_id}` | status / counts / error |
| GET | `/v1/faces/groups` | latest completed snapshot |

## Operational notes

- Single-flight lock is **in-process** (one API replica). Multi-replica needs an external lock later.
- On process start, `pending`/`running` rows are marked `failed` (`interrupted by process restart`).
- Older completed runs are pruned (`FACE_GROUPS_KEEP_RUNS`, default 2).

## Out of scope

- `regionId` partitioning
- Celery/Redis workers
- Incremental enroll-time cluster updates
