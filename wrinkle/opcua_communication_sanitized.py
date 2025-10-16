"""
opcua_communication.py — sanitized for external sharing.

Notes:
- This file avoids embedding any OPC-UA node strings or institute-specific info.
- The endpoint URL defaults to "opc.tcp://127.0.0.1:4840" but can be overridden with
  the environment variable OPCUA_ENDPOINT, e.g.:
      set OPCUA_ENDPOINT=opc.tcp://127.0.0.1:4840
- Node strings are NOT hard-coded; they are read from globals.nodes (already sanitized).
- If 'opcua' package is unavailable, calls will no-op so the GUI can still run for algorithm work.
"""

import os
import logging

try:
    from opcua import Client
except Exception:
    Client = None  # Library not available; run in no-op mode

import globals as vars

DEFAULT_ENDPOINT = os.environ.get("OPCUA_ENDPOINT", "opc.tcp://127.0.0.1:4840")


class OpcuaInterface:
    def __init__(self, endpoint: str | None = None, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.client = None
        self._connected = False

    # Keep an explicit connect() (don’t auto-connect on import/construct)
    def connect(self) -> bool:
        if Client is None:
            self.logger.warning("python-opcua not installed; OpcuaInterface running in no-op mode.")
            return False
        try:
            self.client = Client(self.endpoint)
            self.client.connect()
            self._connected = True
            self.logger.info("OPC-UA connected: %s", self.endpoint)
            return True
        except Exception as e:
            self.logger.error("Failed to connect OPC-UA: %s", e)
            self.client = None
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
                self.logger.info("OPC-UA disconnected.")
            except Exception as e:
                self.logger.warning("Error on OPC-UA disconnect: %s", e)
        self.client = None
        self._connected = False

    def get_single_values(self, nodeset: list[str]) -> dict:
        """
        Reads the nodes in 'nodeset'. Node strings are looked up in globals.nodes.
        Returns a dict {entry_name: float_value} without raising if any read fails.
        """
        results = {}
        if not nodeset:
            return results

        # No-op mode (no opcua lib or not connected): return empty dictionary
        if Client is None or not self._connected or self.client is None:
            self.logger.debug("OPC-UA not connected; get_single_values() returning empty results.")
            return results

        for entry in nodeset:
            node_id = vars.nodes.get(entry)
            if not node_id:
                self.logger.debug("Skipping unknown node key: %s", entry)
                continue
            try:
                val = self.client.get_node(node_id).get_value()
                try:
                    val = float(val)
                except Exception:
                    # leave as is if not float-convertible
                    pass
                results[entry] = val
            except Exception as e:
                self.logger.debug("Failed to read %s (%s): %s", entry, node_id, e)

        # Update global cache if present/desired
        try:
            vars.single_values_opcua.update(results)
        except Exception:
            pass

        return results
