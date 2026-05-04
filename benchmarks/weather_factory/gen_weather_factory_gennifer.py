import argparse

# https://github.com/sjunges/rubicon/blob/master/examples/weatherfactory/weatherfactory19.prism
# module weathermodule
#     sun : bool init true;
#     [act]  sun -> 0.7: (sun'=sun) + 0.3: (sun'=!sun);
#     [act] !sun -> 0.4: (sun'=sun) + 0.6: (sun'=!sun);
# endmodule

# module factory1
#     state1 : bool init false;
#     [act] state1 & sun  -> 0.3 * p1: (state1'=true) + 1-(0.3 * p1): (state1'=false);
#     [act] !state1 & sun -> 0.7 * q1: (state1'=true) + 1-(0.7 * q1): (state1'=false);
#     [act] state1 & !sun -> 0.6 * p1: (state1'=true) + 1-(0.6 * p1): (state1'=false);
#     [act] !state1 & !sun -> 0.4 * q1: (state1'=true) + 1-(0.4 * q1): (state1'=false);
# endmodule

rubicon_p = [0.1,0.2,0.41,0.94,0.434,0.4341,0.4345,0.4344,0.4499,0.438,0.4384,0.43813,0.43822,0.4381,0.4382,0.4181,0.4082,0.4034,0.4482]
rubicon_q = [0.2,0.3,0.45,0.243,0.293,0.2934,0.2939,0.2924,0.2933,0.233,0.2334,0.23394,0.23313,0.23381,0.2333,0.22381,0.2533,0.2133,0.2033]

assert len(rubicon_p) == 19
assert len(rubicon_q) == 19

sun_p = [0.6, 0.7]

state_p = [
    [0.4 * q, 0.6 * p, 0.7 * q, 0.3 * p]
    for p, q in zip(rubicon_p, rubicon_q)
]

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--h", type=int, required=True)
parser.add_argument("--mode", type=str, required=True, choices=["monolithic", "sequential"])

args = parser.parse_args()

assert 0 < args.n and args.n <= 19

def generate_text(n, h):
    print("let sun_0 = true in")
    if args.mode == "sequential":
        print(f"let strike_0_0 = true in")

    # Initial state declarations
    for i in range(1, n + 1):
        print(f"let state_{i}_0 = false in")
        if args.mode == "sequential":
            print(f"let strike_{i}_0 = (strike_{i-1}_0 && state_{i}_0) in")
                
    # Generate allStrike condition
    all_strike= " && ".join(f"state_{i}_0" for i in range(1, n + 1)) if args.mode == "monolithic" else f"strike_{n}_0"
    print(f"if {all_strike} then 1 else")
    
    # Generate steps
    for step in range(1, h + 1):
        ident = step  * 4
        print(f"let sun_{step} = if sun_{step-1} then sample(Bernoulli({sun_p[1]})) else sample(Bernoulli({sun_p[0]})) in")
        if args.mode == "sequential":
            print(f"let strike_0_{step} = true in")
        for i in range(1, n + 1):
            print(f"let state_{i}_{step} = if sun_{step-1} then if state_{i}_{step-1} then sample(Bernoulli({state_p[i-1][3]})) else sample(Bernoulli({state_p[i-1][2]})) else if state_{i}_{step-1} then sample(Bernoulli({state_p[i-1][1]})) else sample(Bernoulli({state_p[i-1][0]})) in")
            if args.mode == "sequential":
                print(f"let strike_{i}_{step} = (strike_{i-1}_{step} && state_{i}_{step}) in")

        # Generate allStrike for the current step
        all_strike_step = " && ".join(f"state_{i}_{step}" for i in range(1, n + 1)) if args.mode == "monolithic" else f"strike_{n}_{step}"

        if step < h:
            print(f"if {all_strike_step} then 1 else")
        else:
            print(f"({all_strike_step})")
    
generate_text(args.n, args.h)