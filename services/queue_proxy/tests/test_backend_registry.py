from queue_proxy.backend_registry import parse_registry_instances


def test_parse_registry_instances_filters_ready_http_backends() -> None:
    instances = parse_registry_instances(
        {
            "instances": [
                {
                    "instance_id": "a",
                    "model": "local-main",
                    "base_url": "http://backend-a:8000/v1",
                    "state": "ready",
                    "active_requests": 3,
                },
                {
                    "instance_id": "b",
                    "model": "local-main",
                    "base_url": "dry-run://local-main/b",
                    "state": "ready",
                    "active_requests": 0,
                },
            ]
        }
    )

    ready = [instance for instance in instances if instance.is_ready]

    assert len(ready) == 1
    assert ready[0].instance_id == "a"
