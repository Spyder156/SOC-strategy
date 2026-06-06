"""Tiny websocket broadcast hub.

One simulation feeds all connected browser clients. A late-joining client gets a short
backlog of recent view-frames plus the latest metric snapshot so its charts aren't empty.
The hub is the ONLY coupling between the model backend and the UI.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Optional, Set


class Hub:
    def __init__(self, backlog: int = 1500):
        self.clients: Set = set()
        self.recent = deque(maxlen=backlog)   # recent view frames for late joiners
        self.last_metric: Optional[dict] = None
        self.last_status: Optional[dict] = None
        self.config: Optional[dict] = None

    async def register(self, ws) -> None:
        self.clients.add(ws)
        if self.config:
            await ws.send(json.dumps(self.config))
        if self.last_status:
            await ws.send(json.dumps(self.last_status))
        for frame in list(self.recent):
            await ws.send(json.dumps(frame))
        if self.last_metric:
            await ws.send(json.dumps(self.last_metric))

    def unregister(self, ws) -> None:
        self.clients.discard(ws)

    async def broadcast(self, msg: dict) -> None:
        if msg.get("type") in ("view", "uni", "frag"):
            self.recent.append(msg)
        elif msg.get("type") == "metric":
            self.last_metric = msg
        elif msg.get("type") == "status":
            self.last_status = msg
        elif msg.get("type") == "config":
            self.config = msg

        if not self.clients:
            return
        payload = json.dumps(msg)
        dead = []
        for ws in self.clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)
