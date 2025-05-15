import time
import math
import matplotlib.pyplot as plt

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
    ContinuousSubstrate2D,
    Dynamics2DTask,
    RK42DTask,
    SymplecticEuler2DTask,
    # Electromagnetism imports
    PHOTON,
    PhotonEmissionTask,
    PhotonAbsorptionTask,
    EMConstructor,
    coulomb_coupling_fn,
    # Lorentz-force imports
    FieldSubstrate,
    lorentz_coupling_fn,
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
    print("=== Two-Body GR Demo (1D) ===")
    G = 6.67430e-11
    dt = 0.1
    STEPS = 50
    s1 = ContinuousSubstrate(
        "Body1", -1.0, 0.5, mass=5.0, potential_fn=lambda x: 0.0, dt=dt
    )
    s2 = ContinuousSubstrate(
        "Body2", 1.0, -0.3, mass=3.0, potential_fn=lambda x: 0.0, dt=dt
    )
    grav_task = MultiSubstrateTask("grav", [s1.attr, s2.attr], grav_coupling_fn)
    mc = MultiConstructor([grav_task])
    for i in range(1, STEPS + 1):
        s1, s2 = mc.perform([s1, s2])[0]
        s1 = DynamicsTask().apply(s1)[0]
        s2 = DynamicsTask().apply(s2)[0]
    print(f"Final positions: {s1.x:.3f}, {s2.x:.3f}")
    print()


def demo_orbit_conditions():
    print("=== Circular‐Orbit Conditions Demo ===")
    G, m1, m2, r = 1.0, 5.0, 3.0, 1.0
    μ = m1 * m2 / (m1 + m2)
    v_rel = math.sqrt(G * m1 * m2 / (μ * r))
    print(f"r={r}, μ={μ:.3f}, v_rel={v_rel:.3f}")
    print()


def demo_2d_orbit(integrator, steps=100):
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
        F = G * s1.mass * s2.mass / (r2 if r2 else 1e-6)
        dirx, diry = dx / math.sqrt(r2), dy / math.sqrt(r2) if r2 else (1.0, 0.0)
        s1n, s2n = s1.clone(), s2.clone()
        s1n.px += F * dirx * s1.dt
        s1n.py += F * diry * s1.dt
        s2n.px -= F * dirx * s1.dt
        s2n.py -= F * diry * s1.dt
        return [[s1n, s2n]]

    mc = MultiConstructor([MultiSubstrateTask("grav2d", [sat.attr, cent.attr], grav2d)])
    xs, ys = [], []
    for _ in range(steps):
        sat, cent = mc.perform([sat, cent])[0]
        sat = integrator.apply(sat)[0]
        xs.append(sat.x)
        ys.append(sat.y)
    plt.figure(figsize=(5, 5))
    plt.plot(xs, ys, "-", lw=1)
    plt.scatter([0], [0], color="red", s=30, label="Center")
    plt.axis("equal")
    plt.title("2D Orbit")
    plt.legend()
    plt.show()
    print()


# Electromagnetism demos


def demo_photon_emission_absorption():
    print("=== Photon Emission/Absorption Demo ===")
    ELEC = Attribute("charge_site")
    EM = EMConstructor(ELEC, ΔE=5.0)
    s = Substrate("Atom", ELEC, energy=20.0)
    branches = EM.perform(s)
    for w in branches:
        print(w)
    photon = next(w for w in branches if w.attr == PHOTON)
    print("Absorbing photon...")
    back = EM.perform(photon)[0]
    print("After absorption:", back)
    print()


def demo_coulomb_coupling():
    print("=== Coulomb Coupling Demo (1D) ===")
    cs1 = ContinuousSubstrate(
        "p1", -1.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1, charge=+1
    )
    cs2 = ContinuousSubstrate(
        "p2", +1.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1, charge=+1
    )
    task = MultiSubstrateTask("coulomb", [cs1.attr, cs2.attr], coulomb_coupling_fn)
    mc = MultiConstructor([task])
    out1, out2 = mc.perform([cs1, cs2])[0]
    print("Before coupling:", cs1.p, cs2.p)
    print("After coupling: ", out1.p, out2.p)
    print()


def demo_lorentz_force():
    print("=== Lorentz‐Force Coupling Demo (2D) ===")
    cs = ContinuousSubstrate2D(
        "e−",
        x=0.0,
        y=1.0,
        px=2.0,
        py=3.0,
        mass=1.0,
        potential_fn=lambda x, y: 0.0,
        dt=0.1,
        charge=+1,
    )
    B = FieldSubstrate("B", Bz=0.5)
    task = MultiSubstrateTask("lorentz", [cs.attr, B.attr], lorentz_coupling_fn)
    mc = MultiConstructor([task])

    before = (cs.px, cs.py)
    out_cs, out_B = mc.perform([cs, B])[0]
    after = (out_cs.px, out_cs.py)

    print(f" Before momentum: px={before[0]:.3f}, py={before[1]:.3f}")
    print(f" After  momentum: px={after[0]:.3f}, py={after[1]:.3f}")
    print(f"  B-field (unchanged): Bz={out_B.Bz}")
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
    demo_2d_orbit(Dynamics2DTask(), steps=100)
    demo_2d_orbit(RK42DTask(), steps=100)
    demo_2d_orbit(SymplecticEuler2DTask(), steps=100)
    demo_photon_emission_absorption()
    demo_coulomb_coupling()
    demo_lorentz_force()
