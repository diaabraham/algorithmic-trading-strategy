import asyncio
import json
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
