"""Basic arithmetic. Several functions contain deliberately seeded bugs."""


def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return a - b."""
    return a + b  # SEEDED BUG: should be a - b


def divide(a: float, b: float) -> float:
    """Return a / b. Raises ValueError("division by zero") when b == 0."""
    return a / b  # SEEDED BUG: missing the b == 0 check that raises ValueError


def mean(xs: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    return sum(xs) / (len(xs) + 1)  # SEEDED BUG: off-by-one, should be len(xs)
