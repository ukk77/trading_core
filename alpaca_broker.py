import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load .env from the trading root (parent of this file's directory)
_TRADING_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_TRADING_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)

class AlpacaBroker:
    """Wrapper around Alpaca Trading API for execution and portfolio tracking."""
    
    def __init__(self, paper: bool = True):
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key or "your_" in secret_key:
            raise ValueError("Alpaca API keys are missing or invalid in .env")

        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_account_info(self) -> Dict[str, float]:
        """Fetch current account balances."""
        account = self.client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
        }

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all open positions."""
        positions = self.client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "shares": float(pos.qty),
                "market_value": float(pos.market_value),
                "avg_entry_price": float(pos.avg_entry_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "side": pos.side.name
            }
        return result

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific open position."""
        return self.get_positions().get(symbol.upper())

    def submit_market_order(self, symbol: str, qty: float, side: str) -> Optional[Any]:
        """Execute a market order. 'side' should be 'buy' or 'sell'."""
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=symbol.upper(),
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            order = self.client.submit_order(order_data=req)
            logger.info(f"Submitted {side} order for {qty} shares of {symbol}. Order ID: {order.id}")
            return order
        except Exception as e:
            logger.error(f"Failed to submit {side} order for {symbol}: {e}")
            return None

    def get_order(self, order_id: str) -> Optional[Any]:
        """Fetch a specific order by ID."""
        try:
            return self.client.get_order_by_id(order_id)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
            
    def get_open_orders(self) -> list:
        """Fetch all open orders."""
        from alpaca.trading.requests import GetOrdersRequest
        try:
            req = GetOrdersRequest(status="open")
            return self.client.get_orders(req)
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def close_position(self, symbol: str) -> Optional[Any]:
        """Liquidate the entire position for a given symbol."""
        try:
            order = self.client.close_position(symbol_or_asset_id=symbol.upper())
            logger.info(f"Closed position for {symbol}")
            return order
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return None
