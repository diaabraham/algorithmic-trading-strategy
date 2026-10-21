import asyncio
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

# eventkit (pulled in by ib_insync) expects a thread loop at import time on some Python versions.
try:
    asyncio.get_running_loop()
except RuntimeError:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

from src.broker_ibkr import IBKRBroker, OrderRequest


def test_build_contract_from_json():
    payload = {"symbol": "MES", "secType": "FUT", "exchange": "CME", "currency": "USD"}
    b = IBKRBroker("127.0.0.1", 7497, 1, contract_json=json.dumps(payload))
    req = OrderRequest(symbol="IGNORE", side="buy", quantity=1)
    c = b._build_contract(req)
    assert getattr(c, "symbol", None) == "MES"
    assert getattr(c, "secType", None) == "FUT"


@patch("ib_insync.MarketOrder")
@patch("ib_insync.Stock")
def test_place_order_qualifies_and_places(mock_stock, mock_mo):
    mock_stock.return_value = MagicMock()
    mock_mo.return_value = MagicMock()
    b = IBKRBroker("127.0.0.1", 7497, 1)
    mock_ib = MagicMock()
    mock_ib.isConnected.return_value = True
    qual = MagicMock()
    qual.symbol = "SPY"
    mock_ib.qualifyContracts.return_value = [qual]
    b.ib = mock_ib
    b.place_order(OrderRequest(symbol="SPY", side="buy", quantity=2), paper=True)
    mock_ib.qualifyContracts.assert_called_once()
    mock_ib.placeOrder.assert_called_once()


def test_paper_mode_rejects_live_port():
    broker = IBKRBroker("127.0.0.1", 7496, 1)
    try:
        broker.place_order(OrderRequest(symbol="SPY", side="buy", quantity=1), paper=True)
    except RuntimeError as exc:
        assert "live-trading port" in str(exc)
    else:
        raise AssertionError("live port was accepted in paper mode")


def test_readonly_session_rejects_orders():
    broker = IBKRBroker("127.0.0.1", 7497, 1, readonly=True)
    try:
        broker.place_order(OrderRequest(symbol="SPY", side="buy", quantity=1), paper=True)
    except RuntimeError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("read-only session placed an order")


def test_ensure_loop_allows_ib_insync_import():
    script = (
        "from src.broker_ibkr import _ensure_ib_event_loop\n"
        "_ensure_ib_event_loop()\n"
        "import ib_insync\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
