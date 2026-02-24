"""throttle.py

Simple token bucket limiter to reduce Kite 429s.
"""

from __future__ import annotations
import time
import threading

class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int = 5):
        self.rate = max(0.1, float(rate_per_sec))
        self.capacity = max(1.0, float(burst))
        self.tokens = self.capacity
        self.updated = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0):
        tokens = max(0.0, float(tokens))
        with self.lock:
            while True:
                now = time.time()
                elapsed = now - self.updated
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                need = (tokens - self.tokens) / self.rate
                if need < 0:
                    need = 0.0
                time.sleep(min(1.0, need))
