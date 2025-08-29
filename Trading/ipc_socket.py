# ipc_socket.py
import json, socket, threading, time, struct, os

ENC = 'utf-8'
HDR = struct.Struct('!I')   # 4-byte big-endian length header

class IPCSocket:
    """
    One class does both ends of the conversation.

    >>> # Program A (server / producer)
    >>> link = IPCSocket('demo')
    >>> link.send({'msg': 'hello', 'value': 42})

    >>> # Program B (client / consumer)
    >>> link = IPCSocket('demo')
    >>> print(link.recv())   # {'msg': 'hello', 'value': 42}
    """
    _lock = threading.Lock()

    def __init__(self, channel: str, host: str = '127.0.0.1', port: int = None):
        """
        channel – arbitrary ASCII name used to pick an *unused* port
        host    – loopback by default (same machine)
        port    – supply your own if you already know it
        """
        self.host = host
        if port is None:
            # hash channel name into the "ephemeral" range 49152–65535
            port = 49152 + (hash(channel) & 0x3FFF)
        self.port = port
        self.sock = None
        self._ensure_connected()

    # ---------------- public API ----------------
    def send(self, obj) -> None:
        """Serialize and push one Python object."""
        self._ensure_connected()
        blob = json.dumps(obj, ensure_ascii=False).encode(ENC)
        self.sock.sendall(HDR.pack(len(blob)) + blob)

    def recv(self):
        """Block until exactly one Python object is read."""
        self._ensure_connected()
        raw_len = self._recv_exact(HDR.size)
        (msg_len,) = HDR.unpack(raw_len)
        blob = self._recv_exact(msg_len)
        return json.loads(blob.decode(ENC))

    # ---------------- internals ----------------
    def _ensure_connected(self):
        if self.sock is None:
            with IPCSocket._lock:
                if self.sock is None:
                    try:
                        # try client first (connect to existing server)
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect((self.host, self.port))
                        self.sock = sock
                    except ConnectionRefusedError:
                        # no listener yet → become server
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        sock.bind((self.host, self.port))
                        sock.listen(1)
                        conn, _ = sock.accept()
                        sock.close()        # listening socket no longer needed
                        self.sock = conn

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionResetError("peer closed")
            buf.extend(chunk)
        return bytes(buf)

    # optional: tidy-up
    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None