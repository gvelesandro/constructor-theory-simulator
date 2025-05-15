#!/usr/bin/env python3
import time
import math
import matplotlib.pyplot as plt

from ct_framework import (
    Substrate,
    Attribute,
    Task,
    Constructor,
    ActionConstructor,
    MultiSubstrateTask,
    MultiConstructor,
    TimerSubstrate,
    TimerConstructor,
    ClockConstructor,
    SwapConstructor,
    ascii_branch,
    plot_phase_space,
    ContinuousSubstrate,
    DynamicsTask,
    RK4Task,
    SymplecticEulerTask,
    grav_coupling_fn,
    # 2D classes
    ContinuousSubstrate2D,
    Dynamics2DTask,
    RK42DTask,
    SymplecticEuler2DTask,
)


def demo_task_constructor():
    print("=== Task & Constructor Demo ===")
    g, e = Attribute("ground"), Attribute("excited")
    decay = Task("decay", e, [(g, 0, 0)])
    cons = Constructor([decay])
    p = Substrate("P", e, energy=1)
    print(ascii_branch(cons.perform(p)))
    print()


def demo_action_constructor():
    print("=== ActionConstructor (Least Action) Demo ===")
    inp = Attribute("in")
    A1, B1 = Attribute("A"), Attribute("B")
    low = Task("low", inp, [(A1, 0, 0)], action_cost=1.0)
    high = Task("high", inp, [(B1, 0, 0)], action_cost=2.0)
    ac = ActionConstructor([low, high])
    res = ac.perform(Substrate("X", inp, energy=1))[0]
    print("Picked branch:", res)
    print()


def demo_multi_constructor():
    print("=== MultiConstructor (Gravitational Coupling) Demo ===")
    cs1 = ContinuousSubstrate(
        "m1", 0.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1
    )
    cs2 = ContinuousSubstrate(
        "m2", 1.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1
    )
    grav_task = MultiSubstrateTask("grav", [cs1.attr, cs2.attr], grav_coupling_fn)
    mc = MultiConstructor([grav_task])
    for s1, s2 in mc.perform([cs1, cs2]):
        print(s1, s2)
    print()


def demo_timer_constructor():
    print("=== TimerConstructor Demo ===")
    t = TimerSubstrate("T", 0.5)
    T = TimerConstructor()
    while t.attr.label != "halted":
        T.perform(t)
        print("Timer state:", t.attr.label)
        time.sleep(0.1)
    print()


def demo_clock_constructor():
    print("=== ClockConstructor Demo ===")
    clk = Substrate("C", Attribute("clock"), energy=1)
    ticker = ClockConstructor(0.0)
    for _ in range(3):
        ticker.perform(clk)
        print("Clock value:", clk.clock)
    print()


def demo_swap_constructor():
    print("=== SwapConstructor Demo ===")
    data = Attribute("data")
    a = Substrate("A", data, energy=1, fungible_id="X")
    b = Substrate("B", data, energy=1, fungible_id="X")
    print("Before swap:", a.name, b.name)
    SwapConstructor.swap(a, b)
    print(" After swap:", a.name, b.name)
    print()


def demo_phase_space():
    print("=== Phase-Space Visualiser Demo ===")
    cs = ContinuousSubstrate(
        "traj", 0.0, 1.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1
    )
    traj = [cs]
    for _ in range(5):
        cs = DynamicsTask().apply(cs)[0]
        traj.append(cs)
    plot_phase_space({"simple": traj})
    print()


def demo_integrators():
    print("=== Integrators Demo ===")
    pot = lambda x: 0.5 * x * x
    cs = ContinuousSubstrate("int", 1.0, 0.0, mass=1.0, potential_fn=pot, dt=0.1)
    print(" DynamicsTask:   ", DynamicsTask().apply(cs.clone())[0])
    print(" RK4Task:        ", RK4Task().apply(cs.clone())[0])
    print(" SympEulerTask:  ", SymplecticEulerTask().apply(cs.clone())[0])
    print()


def demo_two_body_GR():
    print("=== Two-Body GR Demo (Newton + weak-field redshift) ===")
    G = 6.67430e-11
    dt = 0.1
    STEPS = 200
    s1 = ContinuousSubstrate(
        "Body1", -1.0, 0.5, mass=5.0, potential_fn=lambda x: 0.0, dt=dt
    )
    s2 = ContinuousSubstrate(
        "Body2", 1.0, -0.3, mass=3.0, potential_fn=lambda x: 0.0, dt=dt
    )
    grav_task = MultiSubstrateTask("grav", [s1.attr, s2.attr], grav_coupling_fn)
    mc = MultiConstructor([grav_task])
    tau1 = tau2 = 0.0
    traj1, traj2, proper1, proper2 = [s1.clone()], [s2.clone()], [], []
    for i in range(1, STEPS + 1):
        s1, s2 = mc.perform([s1, s2])[0]
        s1 = DynamicsTask().apply(s1)[0]
        s2 = DynamicsTask().apply(s2)[0]
        r = abs(s2.x - s1.x) or 1e-6
        s1.grav = -G * s2.mass / r
        s2.grav = -G * s1.mass / r
        tau1 += s1.adjusted_duration(dt)
        tau2 += s2.adjusted_duration(dt)
        proper1.append(tau1)
        proper2.append(tau2)
        traj1.append(s1.clone())
        traj2.append(s2.clone())
        if i % 50 == 0:
            print(
                f"Step {i:3d}: x1={s1.x:+.3f}, x2={s2.x:+.3f}, τ1={tau1:.5f}, τ2={tau2:.5f}"
            )
    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    ax1.plot([p.x for p in traj1], label="B1")
    ax1.plot([p.x for p in traj2], label="B2")
    ax1.set(title="Two-Body Orbit (1D)", xlabel="Step", ylabel="x")
    ax1.legend()
    ax2.plot(range(1, STEPS + 1), [p1 - p2 for p1, p2 in zip(proper1, proper2)])
    ax2.set(title="Δτ", xlabel="Step", ylabel="τ1−τ2")
    plt.tight_layout()
    plt.show()
    print()


def demo_orbit_conditions():
    print("=== Circular‐Orbit Conditions Demo ===")
    G = 6.67430e-11
    m1, m2 = 5.0, 3.0
    r = 1.0
    μ = m1 * m2 / (m1 + m2)
    v_rel = math.sqrt(G * m1 * m2 / (μ * r))
    print(f"r={r}, μ={μ:.3f}, |v1−v2|={v_rel:.5e}")
    p_mag = μ * v_rel
    s1 = ContinuousSubstrate(
        "B1", -r / 2, +p_mag, mass=m1, potential_fn=lambda x: 0.0, dt=0.01
    )
    s2 = ContinuousSubstrate(
        "B2", +r / 2, -p_mag, mass=m2, potential_fn=lambda x: 0.0, dt=0.01
    )
    traj1, traj2 = [s1.clone()], [s2.clone()]
    for i in range(100):
        s1 = DynamicsTask().apply(s1)[0]
        s2 = DynamicsTask().apply(s2)[0]
        traj1.append(s1.clone())
        traj2.append(s2.clone())
    print(f"Final x: B1={traj1[-1].x:.6f}, B2={traj2[-1].x:.6f}\n")
    plt.figure()
    plt.plot([p.x for p in traj1], label="B1")
    plt.plot([p.x for p in traj2], label="B2")
    plt.title("Circular‐Orbit Demo (1D)")
    plt.xlabel("Step")
    plt.ylabel("x")
    plt.legend()
    plt.show()


def demo_2d_orbit(integrator, steps=300):
    print(f"=== 2D Orbit with {integrator.__class__.__name__} ({steps} steps) ===")
    G, M = 1.0, 1.0
    potential2d = lambda x, y: -G * M / math.hypot(x, y)
    m_cent, m_sat, r0 = 100.0, 1.0, 1.0
    v0 = math.sqrt(G * m_cent / r0)
    sat = ContinuousSubstrate2D(
        "Sat", r0, 0.0, 0.0, m_sat * v0, mass=m_sat, potential_fn=potential2d, dt=0.01
    )
    cent = ContinuousSubstrate2D(
        "Cent", 0.0, 0.0, 0.0, 0.0, mass=m_cent, potential_fn=lambda x, y: 0.0, dt=0.01
    )

    def grav2d(subs):
        s1, s2 = subs
        dx, dy = s2.x - s1.x, s2.y - s1.y
        r2 = dx * dx + dy * dy
        if r2 == 0:
            fx = fy = 0.0
        else:
            F = G * s1.mass * s2.mass / r2
            r = math.sqrt(r2)
            fx, fy = F * dx / r, F * dy / r
        dt = s1.dt
        n1, n2 = s1.clone(), s2.clone()
        n1.px += fx * dt
        n1.py += fy * dt
        n2.px -= fx * dt
        n2.py -= fy * dt
        n1.clock += 1
        n2.clock += 1
        return [[n1, n2]]

    mc = MultiConstructor([MultiSubstrateTask("grav2d", [sat.attr, cent.attr], grav2d)])
    xs, ys = [], []
    for _ in range(steps):
        sat, cent = mc.perform([sat, cent])[0]
        sat = integrator.apply(sat)[0]
        xs.append(sat.x)
        ys.append(sat.y)
    plt.figure(figsize=(5, 5))
    plt.plot(xs, ys, "-", lw=1)
    plt.scatter([0], [0], color="red", s=40, label="Central")
    plt.axis("equal")
    plt.title(f"2D Orbit ({integrator.__class__.__name__})")
    plt.legend()
    plt.show()
    print()


if __name__ == "__main__":
    demo_task_constructor()
    demo_action_constructor()
    demo_multi_constructor()
    demo_timer_constructor()
    demo_clock_constructor()
    demo_swap_constructor()
    demo_phase_space()
    demo_integrators()
    demo_two_body_GR()
    demo_orbit_conditions()
    demo_2d_orbit(Dynamics2DTask(), steps=300)
    demo_2d_orbit(RK42DTask(), steps=300)
    demo_2d_orbit(SymplecticEuler2DTask(), steps=300)
