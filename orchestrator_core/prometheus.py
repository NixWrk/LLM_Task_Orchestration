from __future__ import annotations


def prom_labels(**labels: object) -> str:
    return ",".join(
        f'{name}="{prom_label_value(value)}"'
        for name, value in labels.items()
    )


def prom_label_value(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
