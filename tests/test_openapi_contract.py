import json
from pathlib import Path

BASELINE = Path(__file__).parent / "contract" / "openapi_baseline.json"


def _success_schemas(responses: dict) -> str:
    """Every 2xx response schema, keyed by status.

    Not just "200": enroll answers 201 and group rebuild answers 202, so
    hardcoding 200 would leave the response model of the only two endpoints
    that create things completely unconstrained.
    """
    return json.dumps(
        {
            status: response.get("content", {})
            .get("application/json", {})
            .get("schema", {})
            for status, response in responses.items()
            if status.startswith("2")
        },
        sort_keys=True,
    )


def _signature(spec: dict) -> dict:
    """Path + method + status codes + response schema refs, ignoring tags/summaries."""
    out: dict = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            key = f"{method.upper()} {path}"
            out[key] = {
                "statuses": sorted(op.get("responses", {}).keys()),
                "request": json.dumps(
                    op.get("requestBody", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {}),
                    sort_keys=True,
                ),
                "parameters": json.dumps(
                    sorted(
                        (p.get("in"), p.get("name"), p.get("required", False))
                        for p in op.get("parameters", [])
                    ),
                    sort_keys=True,
                ),
                "response": _success_schemas(op.get("responses", {})),
            }
    return out


def test_openapi_signature_matches_baseline(client):
    current = _signature(client.get("/openapi.json").json())
    expected = _signature(json.loads(BASELINE.read_text()))
    assert current == expected


def test_component_schemas_unchanged(client):
    current = client.get("/openapi.json").json()["components"]["schemas"]
    expected = json.loads(BASELINE.read_text())["components"]["schemas"]
    assert sorted(current) == sorted(expected)
    for name in expected:
        assert current[name] == expected[name], f"schema {name} drifted"
