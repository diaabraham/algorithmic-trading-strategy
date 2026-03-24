from src.broker_ibkr import IBKRBroker, OrderRequest
from src.config import Settings

_STOCK_ROUTE_ALIASES = {"WTI": "USO", "CL": "USO", "OIL": "USO"}


def resolve_order_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return _STOCK_ROUTE_ALIASES.get(s, s)


class LiveTradingEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.broker = IBKRBroker(
            host=settings.ibkr_host,
            port=settings.ibkr_port,
            client_id=settings.ibkr_client_id,
            timeout=settings.ibkr_timeout,
            readonly=settings.ibkr_readonly,
            contract_json=settings.ibkr_contract_json,
        )

    def connect(self) -> None:
        self.broker.connect()

    def disconnect(self) -> None:
        self.broker.disconnect()

    def submit_signal(self, symbol: str, signal: str, quantity: int, last_price: float | None = None) -> None:
        if quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        if quantity > self.settings.max_order_qty:
            raise ValueError(
                f"Order quantity {quantity} exceeds MAX_ORDER_QTY={self.settings.max_order_qty}."
            )
        if last_price is not None:
            notional = quantity * last_price
            if notional > self.settings.max_notional_per_order:
                raise ValueError(
                    f"Order notional {notional:.2f} exceeds "
                    f"MAX_NOTIONAL_PER_ORDER={self.settings.max_notional_per_order:.2f}."
                )

        side = "buy" if signal == "long" else "sell"
        routed = resolve_order_symbol(symbol)
        request = OrderRequest(symbol=routed, side=side, quantity=quantity)
        self.broker.place_order(request, paper=self.settings.paper_trading_enabled)
