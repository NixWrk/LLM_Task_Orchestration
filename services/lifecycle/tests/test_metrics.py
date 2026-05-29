from lifecycle.main import prom_label_value


def test_prom_label_value_escapes_special_characters() -> None:
    assert prom_label_value('gpu "0"\\main\n') == 'gpu \\"0\\"\\\\main\\n'
