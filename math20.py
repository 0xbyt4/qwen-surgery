"""A fixed 20-question math test with integer answers.

Each answer is a Python expression, so that the test does not depend on a
hand-computed number.
"""

QUESTIONS: list[tuple[str, str]] = [
    ("What is 47 + 38?", "47 + 38"),
    ("What is 125 - 68?", "125 - 68"),
    ("What is 12 * 13?", "12 * 13"),
    ("What is 144 divided by 12?", "144 // 12"),
    ("What is 7 * 8 + 6?", "7 * 8 + 6"),
    ("What is 3 to the power of 4?", "3**4"),
    ("What is 15% of 200?", "200 * 15 // 100"),
    ("What is 23 * 47?", "23 * 47"),
    ("What is 1234 + 5678?", "1234 + 5678"),
    ("What is (18 + 24) / 6?", "(18 + 24) // 6"),
    (
        "Tom has 5 apples. He buys 3 bags with 4 apples in each bag. "
        "How many apples does Tom have now?",
        "5 + 3 * 4",
    ),
    (
        "A train travels at 60 km per hour for 3 hours. How many km does it travel?",
        "60 * 3",
    ),
    (
        "Sara had 50 dollars. She spent 18 dollars on a book and 7 dollars on "
        "lunch. How many dollars does she have left?",
        "50 - 18 - 7",
    ),
    (
        "There are 4 boxes with 6 pens in each box. Ali gives away 5 pens. "
        "How many pens are left?",
        "4 * 6 - 5",
    ),
    ("Solve for x: 3x + 5 = 20. What is x?", "(20 - 5) // 3"),
    ("Solve for x: 2x - 7 = 11. What is x?", "(11 + 7) // 2"),
    ("What is the sum of the whole numbers from 1 to 10?", "sum(range(1, 11))"),
    ("A rectangle is 7 cm wide and 9 cm long. What is its area in square cm?", "7 * 9"),
    ("What is the next number in the sequence 2, 6, 18, 54?", "54 * 3"),
    (
        "A shop sells 3 pencils for 2 dollars. How many dollars do 12 pencils cost?",
        "12 // 3 * 2",
    ),
]

ANSWERS: list[int] = [eval(expr) for _, expr in QUESTIONS]  # noqa: S307
