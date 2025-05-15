"""
ct_framework.py  ·  Constructor-Theory mini-framework
Includes local‐dynamics (Euler–Lagrange step) with no runtime loops
May 2025
"""

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
        """Proper‐time: special relativity & weak‐field red‐shift."""
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


# ── 2. Task & Constructors ───────────────────────────────────────────────


class Task:
    """
    A Task represents a transformation:
      input_attr → outputs (list of (Attribute, ΔE, ΔQ)).
    Supports duration, quantum flag, irreversible flag, clock_inc, action_cost.
    """

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
        self.name, self.input_attr, self.outputs = name, input_attr, outputs
        self.duration, self.quantum = duration, quantum
        self.irreversible, self.clock_inc = irreversible, clock_inc
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
        time.sleep(min(s.adjusted_duration(self.duration), 0.004))

        # Irreversible tasks mutate & lock the original substrate
        if self.irreversible:
            out_attr, dE, dQ = self.outputs[0]
            s.evolve_to(out_attr)
            s.energy += dE
            s.charge += dQ
            s.clock += self.clock_inc
            s._locked = True

        worlds: List[Substrate] = []
        for attr, dE, dQ in self.outputs:
            if s.energy + dE < 0:
                continue
            w = s.clone()
            w.attr, w.energy, w.charge = attr, w.energy + dE, w.charge + dQ
            w.clock += self.clock_inc
            if self.irreversible:
                w._locked = True
            if self.quantum:
                w.entangled_with = s.entangled_with
            worlds.append(w)
        return worlds


class Constructor:
    """
    Default constructor: applies *all* possible tasks to a substrate,
    mutating it for irreversible ones and collecting all branches.
    """

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


class ActionConstructor:
    """
    Principle of Least Action without runtime loops:
    pick the task with minimal action_cost per input attribute at init.
    """

    def __init__(self, tasks: List[Task]):
        best: Dict[str, Task] = {}
        for t in tasks:
            key = t.input_attr.label
            prev = best.get(key)
            if prev is None or t.action_cost < prev.action_cost:
                best[key] = t
        self._best = best

    def perform(self, s: Substrate) -> List[Substrate]:
        if getattr(s, "_locked", False):
            return []
        t = self._best.get(s.attr.label)
        if t and t.possible(s):
            return t.apply(s)
        return [s]


class NullConstructor(Constructor):
    """A no-op constructor."""

    def __init__(self):
        super().__init__([])

    def perform(self, s: Substrate) -> List[Substrate]:
        return [s]


# ── 3. Timer & Clock Constructors ───────────────────────────────────────


class TimerSubstrate(Substrate):
    def __init__(self, name: str, period: float):
        super().__init__(name, Attribute("start"), 0.0)
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
    """Clock that increments the real substrate.clock each tick."""

    def __init__(self, tick: float):
        self.tick = tick

    def perform(self, s: Substrate) -> List[Substrate]:
        time.sleep(min(s.adjusted_duration(self.tick), 0.004))
        s.clock += 1
        return [s]


# ── 4. Fungibility swap & ASCII visualiser ─────────────────────────────


class SwapConstructor:
    @staticmethod
    def swap(a: Substrate, b: Substrate):
        if not a.is_fungible_with(b):
            raise ValueError("Substrates not fungible")
        a.name, b.name = b.name, a.name
        return a, b


def ascii_branch(worlds: List[Substrate]) -> str:
    return "\n".join(f"* {w.attr.label} ({w.name})" for w in worlds)


# ── 5. Local Dynamics (Euler–Lagrange step) ────────────────────────────


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

    def clone(self):
        w = super().clone()
        w.x, w.p, w.mass = self.x, self.p, self.mass
        w.potential_fn, w.dt = self.potential_fn, self.dt
        return w


class DynamicsTask(Task):
    """
    Local Hamiltonian dynamics:
    one step of Euler-Lagrange / Hamilton's equations.
    """

    def __init__(self):
        super().__init__(
            name="dynamics",
            input_attr=Attribute("dynamic"),
            outputs=[(Attribute("dynamic"), 0.0, 0)],
            duration=0.0,
            clock_inc=1,
        )

    def apply(self, s: ContinuousSubstrate) -> List[Substrate]:
        if not self.possible(s):
            return []
        # compute force = -∂V/∂x
        force = -finite_diff(s.potential_fn, s.x, s.dt)
        p_new = s.p + force * s.dt
        x_new = s.x + (p_new / s.mass) * s.dt

        w = s.clone()
        w.p = p_new
        w.x = x_new
        w.clock += 1
        return [w]
