import itertools
import time
from collections import Iterator, Iterable
import datetime
from math import ceil
from multiprocessing import Lock


class CoolDownIterator(Iterator):
    """Thread-safe iterator that with a blocking cool down after each iteration."""

    def __init__(self, iterable: Iterable, cool_down: datetime.timedelta = datetime.timedelta(seconds=0)):
        self._lock = Lock()
        self._cool_down = ceil(abs(cool_down).total_seconds())
        self._next_yield_time = float('-inf')
        self._iterable = iter(iterable) if iterable else itertools.count()

    def __next__(self):
        with self._lock:
            if time.perf_counter() < self._next_yield_time:
                time.sleep(self._next_yield_time - time.perf_counter())

            self._next_yield_time = time.perf_counter() + self._cool_down

            return self._iterable.__next__()

    def __iter__(self):
        return self
