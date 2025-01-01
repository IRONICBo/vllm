from contextlib import contextmanager
import time

@contextmanager
def measure_latency():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        cost_time = end_time - start_time
        print(f"Cost time: {cost_time:.6f} seconds")
        return cost_time

class LatencyContext:
    def __init__(self):
        self.start_time = None
        self.cost_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.cost_time = None
        print(111)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(111)
        end_time = time.perf_counter()
        self.cost_time = end_time - self.start_time

with LatencyContext() as ctx:
    time.sleep(1)

print(f"Cost time: {ctx.cost_time:.6f} seconds")

with measure_latency() as cost_time:
    time.sleep(1)

print(f"Cost time: {cost_time} seconds")