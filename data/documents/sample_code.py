"""
Sample Python code for RAG Agent testing.

This module demonstrates various Python constructs that the RAG Agent
can process and understand when answering questions about code.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class Person:
    """A simple person data class."""

    name: str
    age: int
    email: Optional[str] = None

    def greet(self) -> str:
        """Generate a greeting message."""
        return f"Hello, my name is {self.name} and I am {self.age} years old."


def calculate_factorial(n: int) -> int:
    """Calculate the factorial of a number using recursion.

    Args:
        n: A non-negative integer

    Returns:
        The factorial of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * calculate_factorial(n - 1)


def fibonacci_sequence(n: int) -> List[int]:
    """Generate the first n Fibonacci numbers.

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of Fibonacci numbers
    """
    if n <= 0:
        return []

    sequence = [0, 1]
    while len(sequence) < n:
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)

    return sequence[:n]


def process_data(
    data: List[Union[int, float, str]],
) -> Dict[str, Union[int, float, List]]:
    """Process a list of mixed data types.

    Args:
        data: List containing integers, floats, and strings

    Returns:
        Dictionary with statistics about the data
    """
    numbers = [x for x in data if isinstance(x, (int, float))]
    strings = [x for x in data if isinstance(x, str)]

    result = {
        "total_items": len(data),
        "numeric_count": len(numbers),
        "string_count": len(strings),
        "numbers": numbers,
        "strings": strings,
    }

    if numbers:
        result["sum"] = sum(numbers)
        result["average"] = sum(numbers) / len(numbers)
        result["maximum"] = max(numbers)
        result["minimum"] = min(numbers)

    return result


class Calculator:
    """A simple calculator class with basic operations."""

    def __init__(self):
        """Initialize the calculator."""
        self.history: List[str] = []

    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} × {b} = {result}")
        return result

    def divide(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} ÷ {b} = {result}")
        return result

    def get_history(self) -> List[str]:
        """Get calculation history."""
        return self.history.copy()


# Example usage
if __name__ == "__main__":
    # Test the calculator
    calc = Calculator()
    print("Calculator operations:")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 × 7 = {calc.multiply(6, 7)}")
    print(f"15 ÷ 3 = {calc.divide(15, 3)}")

    print("\nCalculation history:")
    for operation in calc.get_history():
        print(f"  {operation}")

    # Test other functions
    print(f"\nFactorial of 5: {calculate_factorial(5)}")

    print(f"\nFirst 10 Fibonacci numbers: {fibonacci_sequence(10)}")

    # Test data processing
    mixed_data = [1, 2.5, "hello", 3, "world", 4.2]
    processed = process_data(mixed_data)
    print(f"\nData processing result: {processed}")

    # Test Person class
    person = Person("Alice", 30, "alice@example.com")
    print(f"\nPerson greeting: {person.greet()}")
