from src.live_trading import resolve_order_symbol


def test_resolve_wti_to_uso():
    assert resolve_order_symbol("WTI") == "USO"
    assert resolve_order_symbol("spy") == "SPY"
