import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

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
        from ib_insync import Contract, Stock

        raw = (self.contract_json or "").strip()
        if raw:
            payload: dict[str, Any] = json.loads(raw)
            return Contract(**payload)

        return Stock(request.symbol.strip().upper(), "SMART", "USD")

    def place_order(self, request: OrderRequest, paper: bool = True):
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

        _ = paper
        trade = self.ib.placeOrder(resolved, order)
        return trade
