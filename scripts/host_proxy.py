#!/usr/bin/env python3
"""Expose the loopback-bound server on the docker bridge for in-container agents.

Terminal-Bench / Harbor run the agent inside a container, so it cannot reach
127.0.0.1:30000 on the host. This forwards <bridge-ip>:PORT to 127.0.0.1:TARGET
without touching the server's own bind address (which stays loopback-only).

    nohup python3 scripts/host_proxy.py > results/proxy.log 2>&1 &
"""
from __future__ import annotations

import os
import socket
import socketserver
import threading

BIND = os.environ.get("PROXY_BIND", "172.17.0.1")
PORT = int(os.environ.get("PROXY_PORT", "30001"))
TARGET = ("127.0.0.1", int(os.environ.get("TARGET_PORT", "30000")))


def pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            buf = src.recv(65536)
            if not buf:
                break
            dst.sendall(buf)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            up = socket.create_connection(TARGET, timeout=30)
        except OSError as exc:
            print(f"upstream connect failed: {exc}", flush=True)
            return
        up.settimeout(None)
        self.request.settimeout(None)
        t = threading.Thread(target=pump, args=(self.request, up), daemon=True)
        t.start()
        pump(up, self.request)
        t.join(timeout=5)
        up.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print(f"forwarding {BIND}:{PORT} -> {TARGET[0]}:{TARGET[1]}", flush=True)
    Server((BIND, PORT), Handler).serve_forever()
