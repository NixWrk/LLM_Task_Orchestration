from gpu_inventory.nvidia_smi import parse_fake_inventory, parse_nvidia_smi_csv


def test_parse_nvidia_smi_csv() -> None:
    gpus = parse_nvidia_smi_csv(
        "0, NVIDIA RTX 4090, 24564, 1024, 12, 45\n"
        "1, NVIDIA RTX 3090, 24576, 4096, 88, 70\n"
    )

    assert len(gpus) == 2
    assert gpus[0].id == "gpu0"
    assert gpus[0].memory_free_mb == 23540
    assert gpus[1].utilization_gpu_percent == 88


def test_parse_fake_inventory() -> None:
    snapshot = parse_fake_inventory(
        '{"gpus":[{"id":"gpu0","index":0,"name":"fake","memory_total_mb":1000,'
        '"memory_used_mb":250}]}'
    )

    assert snapshot.source == "fake"
    assert snapshot.gpus[0].memory_free_mb == 750
