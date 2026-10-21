import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    polygon_api_key: str
    massive_api_key: str = ""
    databento_api_key: str = ""
    databento_dataset: str = "EQUS.SUMMARY"
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_timeout: int = 15
    ibkr_readonly: bool = False
    ibkr_contract_json: str | None = None
    live_trading_enabled: bool = False
    paper_trading_enabled: bool = True
    max_order_qty: int = 10
    max_notional_per_order: float = 25000.0
    cad_usd_fx: float = 0.74


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_settings() -> Settings:
    # Empty keys are allowed. Missing paid sources are skipped in the data loader.
    polygon_api_key = os.getenv("POLYGON_API_KEY", "").strip()
    massive_api_key = os.getenv("MASSIVE_API_KEY", "").strip()
    databento_api_key = os.getenv("DATABENTO_API_KEY", "").strip()
    databento_dataset = os.getenv("DATABENTO_DATASET", "EQUS.SUMMARY").strip() or "EQUS.SUMMARY"

    contract_json = os.getenv("IBKR_CONTRACT_JSON", "").strip() or None

    return Settings(
        polygon_api_key=polygon_api_key,
        massive_api_key=massive_api_key,
        databento_api_key=databento_api_key,
        databento_dataset=databento_dataset,
        ibkr_host=os.getenv("IBKR_HOST", "127.0.0.1").strip(),
        ibkr_port=int(os.getenv("IBKR_PORT", "7497").strip()),
        ibkr_client_id=int(os.getenv("IBKR_CLIENT_ID", "1").strip()),
        ibkr_timeout=int(os.getenv("IBKR_TIMEOUT", "15").strip()),
        ibkr_readonly=_as_bool(os.getenv("IBKR_READONLY", "false")),
        ibkr_contract_json=contract_json,
        live_trading_enabled=_as_bool(os.getenv("LIVE_TRADING_ENABLED", "false")),
        paper_trading_enabled=_as_bool(os.getenv("PAPER_TRADING_ENABLED", "true")),
        max_order_qty=int(os.getenv("MAX_ORDER_QTY", "10").strip()),
        max_notional_per_order=float(os.getenv("MAX_NOTIONAL_PER_ORDER", "25000").strip()),
        cad_usd_fx=float(os.getenv("CAD_USD_FX", "0.74").strip()),
    )
