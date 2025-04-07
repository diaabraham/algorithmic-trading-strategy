from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position: int  # 1 for long, -1 for short
    pnl: float
    holding_period: int 