"""
Harder item set for the reasoning-effort sweep.

The first run used arithmetic like 17*23 and produced a flat line: median 26
reasoning tokens at 'low' against 28 at 'max', with 100% accuracy everywhere.
Which item you asked mattered 26x more than which bucket you set. That is a
broken instrument, not a finding about the model.

These items all require multi-step computation an LLM cannot shortcut by
recall. Every answer was computed with Python and verified, not remembered.
They remain auto-gradable: one integer, no units.

Drop this in beside run_effort.py and import ITEMS from here.
"""

# (id, prompt, verified answer)
ITEMS = [
    ("h01", "How many integers from 1 to 10000 inclusive are divisible by 3 or by 5, "
            "but not by 7?", 4001),
    ("h02", "In how many ways can you make 137 cents using coins of 1, 5, 10 and 25 "
            "cents, if the order of coins does not matter?", 517),
    ("h03", "What is the sum of all prime numbers below 2000?", 277050),
    ("h04", "What is 7 raised to the power 1000, modulo 1001?", 672),
    ("h05", "How many 6-digit numbers (no leading zero) have digits summing to "
            "exactly 20?", 29496),
    ("h06", "How many trailing zeros does 250 factorial have?", 62),
    ("h07", "What is the smallest positive integer n such that n factorial has more "
            "than 100 decimal digits?", 70),
    ("h08", "In the Josephus problem with 41 people standing in a circle and every "
            "3rd person eliminated, what is the position number of the survivor? "
            "Positions are numbered 1 to 41.", 31),
    ("h09", "Which starting number under 10000 produces the longest Collatz "
            "sequence?", 6171),
    ("h10", "What is the sum of the decimal digits of 2 raised to the power 500?", 679),
    ("h11", "How many integer solutions are there to a + b + c = 30 where "
            "0 <= a <= 10, 0 <= b <= 15 and 0 <= c <= 20?", 121),
    ("h12", "How many derangements are there of 10 distinct objects, that is, "
            "permutations with no element in its original position?", 1334961),
    ("h13", "In how many ways can a 2-by-20 rectangle be tiled with 2-by-1 "
            "dominoes?", 10946),
    ("h14", "How many lattice paths go from (0,0) to (8,8) using only unit steps "
            "right and up, without ever rising above the line y = x?", 1430),
    ("h15", "How many 5-digit prime numbers are palindromes?", 93),
    ("h16", "What is 3 raised to the power 1000, modulo 1000000?", 220001),
    ("h17", "What is the sum of the proper divisors of 8128, excluding 8128 "
            "itself?", 8128),
    ("h18", "What is Euler's totient function of 123456?", 41088),
    ("h19", "How many decimal digits does 100 factorial have?", 158),
    ("h20", "How many integers from 1 to 1000 inclusive are coprime to 1000?", 400),
]
