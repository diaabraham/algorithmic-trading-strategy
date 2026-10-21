from src.live_trading import open_position_side, resolve_order_symbol


def test_resolve_wti_to_uso():
    assert resolve_order_symbol("WTI") == "USO"
    assert resolve_order_symbol("spy") == "SPY"


def test_open_position_side_ignores_flat_book():
    assert open_position_side(1) == "long"
    assert open_position_side(-1) == "short"
    assert open_position_side(0) is None
