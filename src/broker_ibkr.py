import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# TWS live 7496, IB Gateway live 4001. Paper ports are 7497 and 4002.
_LIVE_TRADING_PORTS = {7496, 4001}


def _ensure_ib_event_loop() -> None:
    """eventkit calls get_event_loop() at import, which raises on Python 3.14+."""
    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: int
    order_type: str = "MKT"
    limit_price: Optional[float] = None


class IBKRBroker:
    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        timeout: int = 15,
        readonly: bool = False,
        contract_json: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self.contract_json = contract_json
        self.ib = None

    def connect(self) -> None:
        _ensure_ib_event_loop()
        from ib_insync import IB

        self.ib = IB()
        logger.info(
            "Connecting to IBKR at %s:%s clientId=%s timeout=%ss readonly=%s",
            self.host,
            self.port,
            self.client_id,
            self.timeout,
            self.readonly,
        )
        self.ib.connect(
            self.host,
            self.port,
            clientId=self.client_id,
            timeout=self.timeout,
            readonly=self.readonly,
        )
        if not self.ib.isConnected():
            raise RuntimeError(
                "IBKR connect() returned but session is not connected. "
                "Ensure TWS or IB Gateway is running and API is enabled on this port."
            )
        logger.info("IBKR session connected.")

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKR disconnected.")

    def _build_contract(self, request: OrderRequest) -> Any:
        _ensure_ib_event_loop()
        from ib_insync import Contract, Stock

        raw = (self.contract_json or "").strip()
        if raw:
            payload: dict[str, Any] = json.loads(raw)
            return Contract(**payload)

        return Stock(request.symbol.strip().upper(), "SMART", "USD")

    def place_order(self, request: OrderRequest, paper: bool = True):
        _ensure_ib_event_loop()
        if self.readonly:
            raise RuntimeError("IBKR session is read-only; refusing to place an order.")
        if paper and self.port in _LIVE_TRADING_PORTS:
            raise RuntimeError(
                f"Paper trading is enabled but IBKR port {self.port} is a live-trading port. "
                "Use 7497 (TWS paper) or 4002 (IB Gateway paper), "
                "or set PAPER_TRADING_ENABLED=false to allow live orders."
            )
        if self.ib is None or not self.ib.isConnected():
            raise RuntimeError("IBKR is not connected. Call connect() first.")

        from ib_insync import LimitOrder, MarketOrder

        contract = self._build_contract(request)
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(
                f"Could not qualify IBKR contract for symbol={request.symbol!r}. "
                "Check symbol/exchange or set IBKR_CONTRACT_JSON for futures/options."
            )
        resolved = qualified[0]

        action = "BUY" if request.side.lower() == "buy" else "SELL"
        if request.order_type.upper() == "LMT":
            if request.limit_price is None:
                raise ValueError("limit_price is required for LMT orders.")
            order = LimitOrder(action, request.quantity, request.limit_price)
        else:
            order = MarketOrder(action, request.quantity)

        trade = self.ib.placeOrder(resolved, order)
        return trade
