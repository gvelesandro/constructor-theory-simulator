#!/usr/bin/env python3
import time
from ct_framework import (
    Substrate,
    Attribute,
    Task,
    Constructor,
    NullConstructor,
    TimerSubstrate,
    TimerConstructor,
    ClockConstructor,
    SwapConstructor,
    ascii_branch,
)


def demo_excitation():
    print("=== Excitation Demo ===")
    g = Attribute("ground")
    e = Attribute("excited")
    excite = Task("excite", g, [(e, -1, 0)], duration=0.0, clock_inc=0)
    cons = Constructor([excite])
    s = Substrate("S1", g, energy=1)
    print(f" Before: {s}")
    out = cons.perform(s)[0]
    print(f" After:  {out}\n")


def demo_many_worlds():
    print("=== Many-Worlds Demo ===")
    ground = Attribute("ground")
    excited = Attribute("excited")
    alternate = Attribute("alternate")
    decay = Task(
        name="decay",
        input_attr=excited,
        outputs=[(ground, 0, 0), (alternate, 0, 0)],
        duration=0.0,
        clock_inc=0,
    )
    cons = Constructor([decay])
    particle = Substrate("P", excited, energy=1)
    worlds = cons.perform(particle)
    print("Branches after decay:")
    for w in worlds:
        print("  ", w)
    print("\nASCII branching:")
    print(ascii_branch(worlds))
    print()


def demo_entangle_decohere():
    print("=== Entangle / Decoherence Demo ===")
    up, down, ent = Attribute("up"), Attribute("down"), Attribute("entangled")
    ent_task = Task("ent", up, [(ent, 0, 0)], quantum=True)
    deco = Task("deco", ent, [(up, 0, 0), (down, 0, 0)], irreversible=True)
    cons = Constructor([ent_task, deco])

    a = Substrate("Q", up, energy=1)
    a.entangled_with = a
    entangled = cons.perform(a)[0]
    deco_branches = cons.perform(entangled)
    for b in deco_branches:
        print(f" Branch: {b}")
    print()


def demo_timer():
    print("=== Timer Demo ===")
    t = TimerSubstrate("T", 1.0)
    T = TimerConstructor()
    while t.attr.label != "halted":
        T.perform(t)
        print(f" Timer state: {t.attr.label}")
        time.sleep(0.3)
    print()


def demo_clock():
    print("=== Clock Tick Demo ===")
    clk = Substrate("C", Attribute("clock"), energy=1)
    ticker = ClockConstructor(0.0)
    for _ in range(3):
        ticker.perform(clk)
        print(f" Clock: {clk.clock}")
    print()


def demo_swap():
    print("=== Swap Fungible Demo ===")
    data = Attribute("data")
    a = Substrate("A", data, energy=1, fungible_id="X")
    b = Substrate("B", data, energy=1, fungible_id="X")
    print(f" Before swap: A.name={a.name}, B.name={b.name}")
    SwapConstructor.swap(a, b)
    print(f" After swap:  A.name={a.name}, B.name={b.name}")
    print()


def demo_relativity():
    print("=== Relativity Demo ===")
    c = 299_792_458
    ground = Attribute("ground")
    s1 = Substrate("low-speed", ground, energy=0)
    s2 = Substrate("high-speed", ground, energy=0, velocity=0.8 * c)
    print(f" Low-speed duration(1.0): {s1.adjusted_duration(1.0)}")
    print(f" High-speed duration(1.0): {s2.adjusted_duration(1.0)}\n")


def demo_ascii():
    print("=== ASCII Branch Demo ===")
    D = Attribute("D")
    D1 = Attribute("D1")
    D2 = Attribute("D2")
    decay = Task("dec", D, [(D1, 0, 0), (D2, 0, 0)])
    cons = Constructor([decay])
    worlds = cons.perform(Substrate("X", D, energy=1))
    print(ascii_branch(worlds))
    print()


if __name__ == "__main__":
    demo_excitation()
    demo_many_worlds()
    demo_entangle_decohere()
    demo_timer()
    demo_clock()
    demo_swap()
    demo_relativity()
    demo_ascii()
