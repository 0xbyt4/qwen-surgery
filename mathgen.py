"""Random math questions of the same kinds as math20.py, as a held-out set.

Neurons chosen with math20.py are tested on these questions, so that a
result does not come only from the 20 questions used to choose them.
"""

import random

NAMES = ["Mia", "Leo", "Ava", "Sam", "Zoe", "Max", "Lily", "Ben"]
THINGS = ["apples", "pens", "cards", "books", "stones", "cups"]


def questions(n: int, seed: int) -> list[tuple[str, int]]:
    r = random.Random(seed)
    out = []
    kinds = ["add", "sub", "mul", "div", "mix", "word_add", "word_mul", "linear"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        if kind == "add":
            a, b = r.randint(100, 9999), r.randint(100, 9999)
            out.append((f"What is {a} + {b}?", a + b))
        elif kind == "sub":
            a, b = r.randint(50, 999), r.randint(10, 49)
            out.append((f"What is {a} - {b}?", a - b))
        elif kind == "mul":
            a, b = r.randint(11, 49), r.randint(11, 49)
            out.append((f"What is {a} * {b}?", a * b))
        elif kind == "div":
            b, c = r.randint(2, 15), r.randint(2, 15)
            out.append((f"What is {b * c} divided by {b}?", c))
        elif kind == "mix":
            a, b, c = r.randint(2, 12), r.randint(2, 12), r.randint(1, 30)
            out.append((f"What is {a} * {b} + {c}?", a * b + c))
        elif kind == "word_add":
            name, thing = r.choice(NAMES), r.choice(THINGS)
            a, b, c = r.randint(3, 40), r.randint(2, 30), r.randint(1, 20)
            out.append(
                (
                    f"{name} has {a} {thing}. {name} gets {b} more and then gives "
                    f"away {c}. How many {thing} does {name} have now?",
                    a + b - c,
                )
            )
        elif kind == "word_mul":
            name, thing = r.choice(NAMES), r.choice(THINGS)
            a, b = r.randint(2, 9), r.randint(2, 12)
            out.append(
                (
                    f"{name} has {a} boxes with {b} {thing} in each box. "
                    f"How many {thing} does {name} have?",
                    a * b,
                )
            )
        else:
            x, a, b = r.randint(1, 20), r.randint(2, 9), r.randint(1, 30)
            out.append((f"Solve for x: {a}x + {b} = {a * x + b}. What is x?", x))
    return out
