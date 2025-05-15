# ct_framework.py  ·  Constructor-Theory mini-framework
# Includes: core constructors, continuous dynamics, quantum gravity,
# electromagnetism (photons & Coulomb), Lorentz‐force, and more.

import math
import time
from typing import List, Tuple, Dict, Optional, Callable

# ── 1. Core datatypes ───────────────────────────────────────────────────


class Attribute:
    def __init__(self, label: str):
        self.label = label

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return isinstance(other, Attribute) and other.label == self.label

    def __repr__(self):
        return f"〈{self.label}〉"


class Substrate:
    def __init__(
        self,
        name: str,
        attr: Attribute,
        energy: float,
        charge: int = 0,
        clock: int = 0,
        velocity: float = 0.0,
        grav: float = 0.0,
        fungible_id: Optional[str] = None,
        entangled_with: Optional["Substrate"] = None,
    ):
        self.name = name
        self.attr = attr
        self.energy, self.charge, self.clock = energy, charge, clock
        self.velocity, self.grav = velocity, grav
        self.fungible_id = fungible_id or attr.label
        self.entangled_with = entangled_with
        self._locked = False

    def adjusted_duration(self, τ: float) -> float:
        c = 299_792_458
        if self.velocity:
            β = self.velocity / c
            γ = 1 / math.sqrt(1 - β * β)
            return τ / γ
        if self.grav:
            return τ * (1 + self.grav / c**2)
        return τ

    def is_fungible_with(self, other: "Substrate") -> bool:
        return self.attr == other.attr and self.fungible_id == other.fungible_id

    def clone(self) -> "Substrate":
        w = Substrate(
            self.name,
            self.attr,
            self.energy,
            self.charge,
            self.clock,
            self.velocity,
            self.grav,
            self.fungible_id,
            self.entangled_with,
        )
        w._locked = self._locked
        return w

    def evolve_to(self, attr: Attribute):
        self.attr = attr

    def __repr__(self):
        return (
            f"{self.name}:{self.attr.label}"
            f"(E={self.energy},Q={self.charge},t={self.clock},"
            f"F={self.fungible_id})"
        )


# ── 2. Task & Constructor ────────────────────────────────────────────────


class Task:
    def __init__(
        self,
        name: str,
        input_attr: Attribute,
        outputs: List[Tuple[Attribute, float, int]],
        duration: float = 0.0,
        quantum: bool = False,
        irreversible: bool = False,
        clock_inc: int = 0,
        action_cost: float = 0.0,
    ):
        self.name = name
        self.input_attr = input_attr
        self.outputs = outputs
        self.duration = duration
        self.quantum = quantum
        self.irreversible = irreversible
        self.clock_inc = clock_inc
        self.action_cost = action_cost

    def possible(self, s: Substrate) -> bool:
        return (
            not getattr(s, "_locked", False)
            and s.attr == self.input_attr
            and all(s.charge + dq == s.charge for _, _, dq in self.outputs)
        )

    def apply(self, s: Substrate) -> List[Substrate]:
        if not self.possible(s):
            return []
        # simulate time delay
        time.sleep(min(s.adjusted_duration(self.duration), 0.004))
        orig = s.clone()
        # classical irreversible: mutate real substrate
        if self.irreversible and not self.quantum:
            out_attr, dE, dQ = self.outputs[0]
            s.evolve_to(out_attr)
            s.energy += dE
            s.charge += dQ
            s.clock += self.clock_inc
            s._locked = True
        worlds: List[Substrate] = []
        for attr, dE, dQ in self.outputs:
            if orig.energy + dE < 0:
                continue
            w = orig.clone()
            w.attr = attr
            # special‐case energies
            if attr is GRAVITON:
                w.energy = 0.0
            elif attr is PHOTON:
                # photon carries residual emitter energy
                # emission_energy is outputs[0][1]
                dE_emit = self.outputs[0][1]
                w.energy = orig.energy + dE_emit
            else:
                w.energy = orig.energy + dE
            w.charge = orig.charge + dQ
            w.clock = orig.clock + self.clock_inc
            if self.irreversible and not self.quantum:
                w._locked = True
            if self.quantum:
                w.entangled_with = orig.entangled_with
            worlds.append(w)
        return worlds


class Constructor:
    def __init__(self, tasks: List[Task]):
        self.tasks_by_input: Dict[str, List[Task]] = {}
        for t in tasks:
            self.tasks_by_input.setdefault(t.input_attr.label, []).append(t)

    def perform(self, s: Substrate) -> List[Substrate]:
        if getattr(s, "_locked", False):
            return []
        candidates = self.tasks_by_input.get(s.attr.label, [])
        if not candidates:
            return [s]
        worlds: List[Substrate] = []
        for t in candidates:
            worlds.extend(t.apply(s))
        return worlds


class ActionConstructor(Constructor):
    def __init__(self, tasks: List[Task]):
        super().__init__(tasks)
        # keep only the least-action task per input
        self.tasks_by_input = {
            label: [min(ts, key=lambda t: t.action_cost)]
            for label, ts in self.tasks_by_input.items()
        }


class NullConstructor(Constructor):
    def __init__(self):
        super().__init__([])

    def perform(self, s: Substrate) -> List[Substrate]:
        return [s]


# ── 3. Timer & Clock ────────────────────────────────────────────────────


class TimerSubstrate(Substrate):
    def __init__(self, name: str, period: float):
        super().__init__(name, Attribute("start"), energy=0.0)
        self.period = period
        self._t0 = time.time()


class TimerConstructor(Constructor):
    START, RUN, HALT = Attribute("start"), Attribute("running"), Attribute("halted")

    def __init__(self):
        super().__init__([])

    def perform(self, t: TimerSubstrate) -> List[Substrate]:
        dt = time.time() - t._t0
        if t.attr == self.START:
            t.attr = self.RUN
        if t.attr == self.RUN and dt >= t.period:
            t.attr = self.HALT
        return [t]


class ClockConstructor:
    def __init__(self, tick: float):
        self.tick = tick

    def perform(self, s: Substrate) -> List[Substrate]:
        time.sleep(min(s.adjusted_duration(self.tick), 0.004))
        s.clock += 1
        return [s]


# ── 4. Fungible swap & ASCII ─────────────────────────────────────────────


class SwapConstructor:
    @staticmethod
    def swap(a: Substrate, b: Substrate):
        if not a.is_fungible_with(b):
            raise ValueError("Substrates not fungible")
        a.name, b.name = b.name, a.name
        return a, b


def ascii_branch(worlds: List[Substrate]) -> str:
    return "\n".join(f"* {w.attr.label} ({w.name})" for w in worlds)


# ── 5. Phase-space visualiser ───────────────────────────────────────────


def plot_phase_space(trajectories: Dict[str, List["ContinuousSubstrate"]]):
    try:
        import matplotlib.pyplot as _plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return
    for label, traj in trajectories.items():
        xs = [s.x for s in traj]
        ps = [s.p for s in traj]
        _plt.figure()
        _plt.plot(xs, ps)
        _plt.title(f"Phase space: {label}")
        _plt.xlabel("x")
        _plt.ylabel("p")
    _plt.show()


# ── 6. 1D Continuous dynamics & integrators ─────────────────────────────


def finite_diff(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


class ContinuousSubstrate(Substrate):
    def __init__(
        self,
        name: str,
        x: float,
        p: float,
        mass: float,
        potential_fn: Callable[[float], float],
        dt: float,
        **kwargs,
    ):
        super().__init__(name, Attribute("dynamic"), energy=0.0, **kwargs)
        self.x, self.p, self.mass = x, p, mass
        self.potential_fn, self.dt = potential_fn, dt

    def clone(self) -> "ContinuousSubstrate":
        w = ContinuousSubstrate(
            self.name,
            self.x,
            self.p,
            self.mass,
            self.potential_fn,
            self.dt,
            fungible_id=self.fungible_id,
            entangled_with=self.entangled_with,
        )
        w.energy, w.charge, w.clock = self.energy, self.charge, self.clock
        w.velocity, w.grav = self.velocity, self.grav
        w._locked = self._locked
        return w


class DynamicsTask(Task):
    def __init__(self):
        super().__init__(
            "dynamics",
            Attribute("dynamic"),
            [(Attribute("dynamic"), 0, 0)],
            clock_inc=1,
        )

    def apply(self, s: ContinuousSubstrate) -> List[Substrate]:
        if not self.possible(s):
            return []
        force = -finite_diff(s.potential_fn, s.x, s.dt)
        p_new = s.p + force * s.dt
        x_new = s.x + (p_new / s.mass) * s.dt
        w = s.clone()
        w.p, w.x, w.clock = p_new, x_new, w.clock + 1
        return [w]


class RK4Task(Task):
    def __init__(self):
        super().__init__(
            "rk4", Attribute("dynamic"), [(Attribute("dynamic"), 0, 0)], clock_inc=1
        )

    def apply(self, s: ContinuousSubstrate) -> List[Substrate]:
        if not self.possible(s):
            return []
        dt, f = s.dt, s.potential_fn

        def deriv(x, p):
            return p / s.mass, -finite_diff(f, x, dt)

        k1x, k1p = deriv(s.x, s.p)
        k2x, k2p = deriv(s.x + k1x * dt / 2, s.p + k1p * dt / 2)
        k3x, k3p = deriv(s.x + k2x * dt / 2, s.p + k2p * dt / 2)
        k4x, k4p = deriv(s.x + k3x * dt, s.p + k3p * dt)
        x_new = s.x + (k1x + 2 * k2x + 2 * k3x + k4x) * dt / 6
        p_new = s.p + (k1p + 2 * k2p + 2 * k3p + k4p) * dt / 6
        w = s.clone()
        w.x, w.p, w.clock = x_new, p_new, w.clock + 1
        return [w]


class SymplecticEulerTask(Task):
    def __init__(self):
        super().__init__(
            "symp_euler",
            Attribute("dynamic"),
            [(Attribute("dynamic"), 0, 0)],
            clock_inc=1,
        )

    def apply(self, s: ContinuousSubstrate) -> List[Substrate]:
        if not self.possible(s):
            return []
        dt = s.dt
        force = -finite_diff(s.potential_fn, s.x, s.dt)
        p_new = s.p + force * dt
        x_new = s.x + (p_new / s.mass) * dt
        w = s.clone()
        w.p, w.x, w.clock = p_new, x_new, w.clock + 1
        return [w]


# ── 7. Multi‐substrate Tasks & 1D coupling ──────────────────────────────


class MultiSubstrateTask:
    def __init__(
        self,
        name: str,
        input_attrs: List[Attribute],
        apply_fn: Callable[[List[Substrate]], List[List[Substrate]]],
        duration: float = 0.0,
    ):
        self.name = name
        self.input_attrs = input_attrs
        self.apply_fn = apply_fn
        self.duration = duration

    def possible(self, substrates: List[Substrate]) -> bool:
        return all(sub.attr == inp for sub, inp in zip(substrates, self.input_attrs))

    def apply(self, substrates: List[Substrate]) -> List[List[Substrate]]:
        time.sleep(min(substrates[0].adjusted_duration(self.duration), 0.004))
        return self.apply_fn(substrates)


class MultiConstructor:
    def __init__(self, tasks: List[MultiSubstrateTask]):
        self.tasks = tasks

    def perform(self, substrates: List[Substrate]) -> List[List[Substrate]]:
        results: List[List[Substrate]] = []
        for t in self.tasks:
            if t.possible(substrates):
                results.extend(t.apply(substrates))
        return results


def grav_coupling_fn(subs: List[Substrate]) -> List[List[Substrate]]:
    s1, s2 = subs
    G = 6.67430e-11
    r = abs(s2.x - s1.x)
    F = G * s1.mass * s2.mass / (r * r if r else 1.0)
    dir12 = (s2.x - s1.x) / r if r else 1.0
    s1n, s2n = s1.clone(), s2.clone()
    dt = s1.dt
    s1n.p += F * dir12 * dt
    s2n.p -= F * dir12 * dt
    s1n.clock += 1
    s2n.clock += 1
    return [[s1n, s2n]]


# ── 8. 2D Continuous & integrators ─────────────────────────────────────


def finite_diff_x(
    f: Callable[[float, float], float], x: float, y: float, h: float
) -> float:
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


def finite_diff_y(
    f: Callable[[float, float], float], x: float, y: float, h: float
) -> float:
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


class ContinuousSubstrate2D(Substrate):
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        px: float,
        py: float,
        mass: float,
        potential_fn: Callable[[float, float], float],
        dt: float,
        **kwargs,
    ):
        super().__init__(name, Attribute("dynamic2d"), energy=0.0, **kwargs)
        self.x, self.y, self.px, self.py, self.mass = x, y, px, py, mass
        self.potential2d, self.dt = potential_fn, dt

    def clone(self) -> "ContinuousSubstrate2D":
        w = ContinuousSubstrate2D(
            self.name,
            self.x,
            self.y,
            self.px,
            self.py,
            self.mass,
            self.potential2d,
            self.dt,
            fungible_id=self.fungible_id,
            entangled_with=self.entangled_with,
        )
        w.energy, w.charge, w.clock = self.energy, self.charge, self.clock
        w.velocity, w.grav = self.velocity, self.grav
        w._locked = self._locked
        return w


class Dynamics2DTask(Task):
    def __init__(self):
        super().__init__(
            "dynamics2d",
            Attribute("dynamic2d"),
            [(Attribute("dynamic2d"), 0, 0)],
            clock_inc=1,
        )

    def apply(self, s: ContinuousSubstrate2D) -> List[Substrate]:
        if not self.possible(s):
            return []
        fx = -finite_diff_x(s.potential2d, s.x, s.y, s.dt)
        fy = -finite_diff_y(s.potential2d, s.x, s.y, s.dt)
        px_new = s.px + fx * s.dt
        py_new = s.py + fy * s.dt
        x_new = s.x + (px_new / s.mass) * s.dt
        y_new = s.y + (py_new / s.mass) * s.dt
        w = s.clone()
        w.px, w.py, w.x, w.y = px_new, py_new, x_new, y_new
        w.clock += 1
        return [w]


class RK42DTask(Task):
    def __init__(self):
        super().__init__(
            "rk42d",
            Attribute("dynamic2d"),
            [(Attribute("dynamic2d"), 0, 0)],
            clock_inc=1,
        )

    def apply(self, s: ContinuousSubstrate2D) -> List[Substrate]:
        if not self.possible(s):
            return []
        dt, f = s.dt, s.potential2d

        def deriv(x, y, px, py):
            fx = -finite_diff_x(f, x, y, dt)
            fy = -finite_diff_y(f, x, y, dt)
            return px / s.mass, py / s.mass, fx, fy

        k1x, k1y, k1px, k1py = deriv(s.x, s.y, s.px, s.py)
        k2x, k2y, k2px, k2py = deriv(
            s.x + k1x * dt / 2,
            s.y + k1y * dt / 2,
            s.px + k1px * dt / 2,
            s.py + k1py * dt / 2,
        )
        k3x, k3y, k3px, k3py = deriv(
            s.x + k2x * dt / 2,
            s.y + k2y * dt / 2,
            s.px + k2px * dt / 2,
            s.py + k2py * dt / 2,
        )
        k4x, k4y, k4px, k4py = deriv(
            s.x + k3x * dt, s.y + k3y * dt, s.px + k3px * dt, s.py + k3py * dt
        )
        x_new = s.x + (k1x + 2 * k2x + 2 * k3x + k4x) * dt / 6
        y_new = s.y + (k1y + 2 * k2y + 2 * k3y + k4y) * dt / 6
        px_new = s.px + (k1px + 2 * k2px + 2 * k3px + k4px) * dt / 6
        py_new = s.py + (k1py + 2 * k2py + 2 * k3py + k4py) * dt / 6
        w = s.clone()
        w.x, w.y, w.px, w.py = x_new, y_new, px_new, py_new
        w.clock += 1
        return [w]


class SymplecticEuler2DTask(Task):
    def __init__(self):
        super().__init__(
            "symp_euler2d",
            Attribute("dynamic2d"),
            [(Attribute("dynamic2d"), 0, 0)],
            clock_inc=1,
        )

    def apply(self, s: ContinuousSubstrate2D) -> List[Substrate]:
        if not self.possible(s):
            return []
        dt = s.dt
        fx = -finite_diff_x(s.potential2d, s.x, s.y, dt)
        fy = -finite_diff_y(s.potential2d, s.x, s.y, dt)
        px_new = s.px + fx * dt
        py_new = s.py + fy * dt
        x_new = s.x + (px_new / s.mass) * dt
        y_new = s.y + (py_new / s.mass) * dt
        w = s.clone()
        w.px, w.py, w.x, w.y = px_new, py_new, x_new, y_new
        w.clock += 1
        return [w]


# ── 9. Quantum Gravity ──────────────────────────────────────────────────

GRAVITON = Attribute("graviton")


class GravitonEmissionTask(Task):
    def __init__(self, mass_attr: Attribute, emission_energy: float = 1.0):
        super().__init__(
            "emit_graviton",
            mass_attr,
            [(mass_attr, -emission_energy, 0), (GRAVITON, 0, 0)],
            quantum=True,
            irreversible=True,
            clock_inc=1,
            action_cost=emission_energy,
        )


class GravitonAbsorptionTask(Task):
    def __init__(self, mass_attr: Attribute, absorption_energy: float = 1.0):
        super().__init__(
            "absorb_graviton",
            GRAVITON,
            [(mass_attr, absorption_energy, 0)],
            quantum=True,
            irreversible=False,
            clock_inc=1,
            action_cost=absorption_energy,
        )


class QuantumGravityConstructor(Constructor):
    def __init__(self, mass_attr: Attribute, ΔE: float = 1.0):
        emit = GravitonEmissionTask(mass_attr, emission_energy=ΔE)
        absorb = GravitonAbsorptionTask(mass_attr, absorption_energy=ΔE)
        super().__init__([emit, absorb])


# ── 10. Electromagnetism ─────────────────────────────────────────────────

PHOTON = Attribute("photon")


class PhotonEmissionTask(Task):
    def __init__(self, source_attr: Attribute, emission_energy: float = 1.0):
        super().__init__(
            "emit_photon",
            source_attr,
            [(source_attr, -emission_energy, 0), (PHOTON, 0, 0)],
            quantum=True,
            irreversible=True,
            clock_inc=1,
            action_cost=emission_energy,
        )


class PhotonAbsorptionTask(Task):
    def __init__(self, target_attr: Attribute, absorption_energy: float = 1.0):
        super().__init__(
            "absorb_photon",
            PHOTON,
            [(target_attr, absorption_energy, 0)],
            quantum=True,
            irreversible=False,
            clock_inc=1,
            action_cost=absorption_energy,
        )


class EMConstructor(Constructor):
    def __init__(self, attr: Attribute, ΔE: float = 1.0):
        emit = PhotonEmissionTask(attr, emission_energy=ΔE)
        absorb = PhotonAbsorptionTask(attr, absorption_energy=ΔE)
        super().__init__([emit, absorb])


def coulomb_coupling_fn(subs: List[Substrate]) -> List[List[Substrate]]:
    s1, s2 = subs
    k = 8.9875517923e9
    dx = s2.x - s1.x
    r2 = dx * dx or 1e-12
    F = k * s1.charge * s2.charge / r2
    dir12 = 1 if dx >= 0 else -1
    s1n, s2n = s1.clone(), s2.clone()
    dt = s1.dt
    s1n.p += F * dir12 * dt
    s2n.p -= F * dir12 * dt
    s1n.clock += 1
    s2n.clock += 1
    return [[s1n, s2n]]


# ── 11. Lorentz‐Force (2D) ───────────────────────────────────────────────


class FieldSubstrate(Substrate):
    def __init__(self, name: str, Bz: float):
        super().__init__(name, Attribute("B_field"), energy=0.0)
        self.Bz = Bz

    def clone(self) -> "FieldSubstrate":
        w = FieldSubstrate(self.name, self.Bz)
        w.energy, w.charge, w.clock = self.energy, self.charge, self.clock
        w.velocity, w.grav = self.velocity, self.grav
        w.fungible_id, w.entangled_with = self.fungible_id, self.entangled_with
        w._locked = self._locked
        return w


def lorentz_coupling_fn(subs: List[Substrate]) -> List[List[Substrate]]:
    particle, field = subs
    Bz = field.Bz
    dt = particle.dt
    Fx = particle.charge * particle.py * Bz
    Fy = -particle.charge * particle.px * Bz
    p_new = particle.clone()
    f_new = field.clone()
    p_new.px += Fx * dt
    p_new.py += Fy * dt
    p_new.clock += 1
    f_new.clock += 1
    return [[p_new, f_new]]
