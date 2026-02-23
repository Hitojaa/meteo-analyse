"""Polymarket order placement via the CLOB API.

Uses py-clob-client to sign and submit limit buy orders.
Configuration is loaded from environment variables (or .env file):
  - POLYMARKET_PRIVATE_KEY  : Wallet private key (required)
  - POLYMARKET_FUNDER       : Funder address (only for proxy/email wallets)
  - POLYMARKET_SIG_TYPE     : Signature type: 0=EOA, 1=email, 2=browser (default: 0)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


@dataclass
class OrderResult:
    """Result of a single order placement."""
    label: str
    success: bool
    order_id: str | None = None
    error: str | None = None
    price: float = 0.0
    size: float = 0.0


def _load_env() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _get_client():
    """Create and authenticate a ClobClient from env vars.

    Raises RuntimeError if dependencies are missing or config is invalid.
    """
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        raise RuntimeError(
            "py-clob-client is not installed.\n"
            "Run: pip install py-clob-client python-dotenv"
        )

    _load_env()

    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
    if not private_key:
        raise RuntimeError(
            "POLYMARKET_PRIVATE_KEY is not set.\n"
            "Create a .env file with:\n"
            "  POLYMARKET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY\n\n"
            "To export your key from Polymarket:\n"
            "  1. Go to polymarket.com → Cash (wallet icon)\n"
            "  2. Click ⋮ → Export Private Key\n"
            "  3. Copy the key and paste it in .env"
        )

    sig_type = int(os.environ.get("POLYMARKET_SIG_TYPE", "0"))
    funder = os.environ.get("POLYMARKET_FUNDER", "").strip() or None

    client = ClobClient(
        CLOB_HOST,
        key=private_key,
        chain_id=CHAIN_ID,
        signature_type=sig_type,
        funder=funder,
    )

    # Derive or retrieve API credentials
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


def place_orders(
    orders: list[dict],
) -> list[OrderResult]:
    """Place limit buy orders on Polymarket.

    Each order dict must have:
      - token_id: str   (CLOB token ID for the "Yes" outcome)
      - label: str      (human-readable label, e.g. "13°C")
      - price: float    (0.0–1.0, e.g. 0.38 for 38¢)
      - size: float     (number of shares)

    Returns a list of OrderResult for each order.
    """
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    client = _get_client()
    results: list[OrderResult] = []

    for order in orders:
        label = order["label"]
        token_id = order["token_id"]
        price = order["price"]
        size = order["size"]

        try:
            resp = client.create_and_post_order(
                OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=size,
                    side=BUY,
                ),
                OrderType.GTC,  # Good-Til-Cancelled limit order
            )

            success = resp.get("success", False)
            order_id = resp.get("orderID")
            error = resp.get("errorMsg") if not success else None

            results.append(OrderResult(
                label=label,
                success=success,
                order_id=order_id,
                error=error,
                price=price,
                size=size,
            ))

            if success:
                log.info("Order placed: %s — %s shares @ %.0f¢ (ID: %s)",
                         label, size, price * 100, order_id)
            else:
                log.warning("Order failed: %s — %s", label, error)

        except Exception as exc:
            results.append(OrderResult(
                label=label,
                success=False,
                error=str(exc),
                price=price,
                size=size,
            ))
            log.warning("Order error: %s — %s", label, exc)

    return results
