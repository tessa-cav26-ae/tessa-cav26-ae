import argparse


P_LIST = [
    0.05, 0.07, 0.09, 0.11, 0.13,
    0.15, 0.17, 0.19, 0.21, 0.23,
    0.25, 0.27, 0.29, 0.31, 0.33,
    0.35, 0.37, 0.39, 0.41, 0.43,
    0.45, 0.47, 0.49, 0.51, 0.53,
    0.55, 0.57, 0.59, 0.61, 0.63,
]
Q_LIST = [
    0.54, 0.56, 0.58, 0.60, 0.62,
    0.64, 0.66, 0.68, 0.70, 0.72,
    0.74, 0.76, 0.78, 0.80, 0.82,
    0.84, 0.86, 0.88, 0.90, 0.92,
    0.94, 0.95, 0.96, 0.97, 0.98,
    0.99, 0.93, 0.91, 0.89, 0.87,
]


def generate_text(n: int):
    """Print a PRISM meeting model with ``n`` participants."""
    if n > len(P_LIST) or n > len(Q_LIST):
        raise ValueError(f"--n must be <= {min(len(P_LIST), len(Q_LIST))}")

    print("dtmc")
    print()

    for i in range(1, n + 1):
        print(f"const double p{i}={P_LIST[i - 1]:.2f};")
        print(f"const double q{i}={Q_LIST[i - 1]:.2f};")
    print()

    print("module P1")
    print("    s1 : [0.. 2] init 0;")
    print("    [step] s1 = 0 -> (1 - p1) : (s1' = 0) + p1 : (s1' = 1);")
    print("    [step] s1 = 1 -> (1 - q1) : (s1' = 0) + q1 : (s1' = 2);")
    print("    [step] s1 = 2 -> 1.0 : (s1' = 2);")
    print("endmodule")
    print()

    for i in range(2, n + 1):
        print(f"module P{i} = P1[s1=s{i},p1=p{i},q1=q{i}] endmodule")

    if n > 1:
        print()

    goal_terms = " & ".join(f"(s{i} = 2)" for i in range(1, n + 1))
    print(f'label "goal" = {goal_terms};')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="number of participants", type=int, required=True)
    args = parser.parse_args()

    if args.n <= 1 or args.n >= 30:
        raise ValueError("--n must be in [2, 30]")

    generate_text(args.n)
