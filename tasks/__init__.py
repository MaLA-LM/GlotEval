# tasks/__init__.py

BENCHMARK_HANDLERS = {}

def register_benchmark(name):
    def decorator(func):
        BENCHMARK_HANDLERS[name] = func
        return func
    return decorator

# Import benchmark handlers
from .text_classification import *
from .machine_translation import *
from .summarization import *
from .open_generation import *
from .comprehension import *
from .negative_log_likelihood import *
from .token_classification import *
from .benchmax_rule_based import *
# Add other benchmark imports here
