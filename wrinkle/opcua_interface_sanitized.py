"""
opcua_interface.py — sanitized for safe external sharing.

- No hard-coded OPC-UA node strings (read from globals.nodes instead).
- Default endpoint is localhost; can override with OPCUA_ENDPOINT env var.
- Graceful behavior if python-opcua is not installed (no-op).
- Same public API class & method names so other code keeps working.
"""

import os
import sys
import logging

try:
    from opcua import Client
    import opcua.ua as ua
except Exception:
    Client = None
    ua = None

import globals as vars

# Endpoint: localhost by default; can override via environment
URL = os.environ.get("OPCUA_ENDPOINT", "opc.tcp://127.0.0.1:4840")

# Limits (safe constants, non-sensitive)
MIN_NUM_SHEETS = 1
MAX_NUM_SHEETS = 10
MIN_GAPSIZE = 100.0
MAX_GAPSIZE = 1000.0
MIN_SPEED = 0.5
MAX_SPEED = 4.0

# Map method-intents to keys in globals.nodes (values are sanitized there)
NODE_KEYS = {
    "SET_NUM_SHEETS": "SET_NUM_SHEETS",
    "SET_MANUAL_PRODUCTION": "SET_MANUAL_PRODUCTION",
    "SET_SETUP_MODULES_START": "SET_SETUP_MODULES_START",
    "SET_SETUP_MODULES_EXECUTE": "SET_SETUP_MODULES_EXECUTE",
    "GET_SETUP_MODULES_IDLE": "GET_SETUP_MODULES_IDLE",
    "SET_DEMONSTRATION_MODE": "SET_DEMONSTRATION_MODE",
    "SET_CALANDERING_START": "SET_CALANDERING_START",
    "SET_CALANDERING_EXECUTE": "SET_CALANDERING_EXECUTE",
    "GET_CALANDERING_IDLE": "GET_CALANDERING_IDLE",
    "SET_GAPSIZE": "SET_GAPSIZE",
    "SET_SPEED": "SET_SPEED",
}


class OpcuaInterface:
    def __init__(self, endpoint: str | None = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.endpoint = endpoint or URL
        self.client = None
        self._connected = False

        if Client is None:
            self.logger.warning("python-opcua not installed; running in no-op mode.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('client', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-create client lazily in connect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    def connect(self) -> bool:
        if Client is None:
            return False
        try:
            self.client = Client(self.endpoint)
            self.client.connect()
            self._connected = True
            self.logger.info("Connected to OPC UA server: %s", self.endpoint)
            return True
        except Exception as err:
            self.logger.error("OPC UA connect failed: %s", err)
            self.client = None
            self._connected = False
            return False

    def disconnect(self):
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception as err:
                self.logger.debug("OPC UA disconnect warning: %s", err)
        self.client = None
        self._connected = False
        self.logger.info("Disconnected from OPC UA server.")

    # --- internal helpers ---
    def _node(self, key: str):
        node_id = vars.nodes.get(key)
        if not node_id:
            self.logger.debug("Missing node key in globals.nodes: %s", key)
            return None
        if not self._connected or self.client is None:
            return None
        try:
            return self.client.get_node(node_id)
        except Exception as e:
            self.logger.debug("get_node failed for %s: %s", key, e)
            return None

    def _set_bool(self, key: str, value: bool):
        if Client is None or ua is None:
            return
        n = self._node(key)
        if n is None:
            return
        try:
            n.set_value(ua.DataValue(ua.Variant(bool(value), ua.VariantType.Boolean)))
        except Exception as e:
            self.logger.debug("set bool failed for %s: %s", key, e)

    def _set_int16(self, key: str, value: int):
        if Client is None or ua is None:
            return
        n = self._node(key)
        if n is None:
            return
        try:
            n.set_value(ua.DataValue(ua.Variant(int(value), ua.VariantType.Int16)))
        except Exception as e:
            self.logger.debug("set int16 failed for %s: %s", key, e)

    def _set_float(self, key: str, value: float):
        if Client is None or ua is None:
            return
        n = self._node(key)
        if n is None:
            return
        try:
            n.set_value(ua.DataValue(ua.Variant(float(value), ua.VariantType.Float)))
        except Exception as e:
            self.logger.debug("set float failed for %s: %s", key, e)

    def _get_value(self, key: str):
        if Client is None:
            return None
        n = self._node(key)
        if n is None:
            return None
        try:
            return n.get_value()
        except Exception as e:
            self.logger.debug("get_value failed for %s: %s", key, e)
            return None

    # --- public API (same method names) ---
    def set_demonstration_mode(self, value: bool):
        self._set_bool(NODE_KEYS["SET_DEMONSTRATION_MODE"], value)

    def set_manual_production(self, value: bool):
        self._set_bool(NODE_KEYS["SET_MANUAL_PRODUCTION"], value)

    def set_num_sheet(self, value: int):
        v = max(MIN_NUM_SHEETS, min(int(value), MAX_NUM_SHEETS))
        self._set_int16(NODE_KEYS["SET_NUM_SHEETS"], v)

    def set_setup_modules_start(self, value: bool):
        self._set_bool(NODE_KEYS["SET_SETUP_MODULES_START"], value)

    def set_setup_modules_execute(self, value: bool):
        self._set_bool(NODE_KEYS["SET_SETUP_MODULES_EXECUTE"], value)

    def get_setup_modules_idle(self):
        return self._get_value(NODE_KEYS["GET_SETUP_MODULES_IDLE"])

    def set_calandering_start(self, value: bool):
        self._set_bool(NODE_KEYS["SET_CALANDERING_START"], value)

    def set_calandering_execute(self, value: bool):
        self._set_bool(NODE_KEYS["SET_CALANDERING_EXECUTE"], value)

    def get_calandering_idle(self):
        return self._get_value(NODE_KEYS["GET_CALANDERING_IDLE"])

    def set_gapsize(self, value: float):
        v = max(MIN_GAPSIZE, min(float(value), MAX_GAPSIZE))
        self._set_float(NODE_KEYS["SET_GAPSIZE"], v)

    def set_speed(self, value: float):
        v = max(MIN_SPEED, min(float(value), MAX_SPEED))
        self._set_float(NODE_KEYS["SET_SPEED"], v)


# --- optional quick test ---
def test_opcua_interface():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s [%(levelname)s]: %(message)s")
    with OpcuaInterface() as interface:
        interface.set_num_sheet(1)
        # Other calls will no-op safely if not connected

if __name__ == "__main__":
    test_opcua_interface()
