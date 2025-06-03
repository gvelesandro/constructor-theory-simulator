# constructor-theory-simulator

A **Python implementation** of David Deutsch’s Constructor Theory framework, exposing key concepts—from simple Tasks and branching substrates to quantum-gravity and electromagnetism—entirely in code. Includes a “universal constructor” that can bootstrap itself from a list of Tasks, demonstrating self-replication and the power of Constructor Theory.

> *“A demonstration of how constructor theory **could** be explored in code, not a high-precision physics engine. For the formal definitions, see David Deutsch and Chiara Marletto’s recent paper “[Constructor Theory of Time](https://arxiv.org/abs/2505.08692)” (May 13, 2025).*

---

## 🚀 Features

* **Core framework**: Attributes, Substrates, Tasks, Constructors
* **Irreversible & quantum tasks**: Many-worlds branching, decoherence guards
* **Timers & Clocks**: Simulate proper-time, special/general relativity corrections
* **Fungibility & SwapConstructor**: Free exchange of identical substrates
* **ASCII visualizer**: `ascii_branch()` for quick text-based branch inspection
* **Continuous dynamics**: 1D & 2D substrates, `DynamicsTask`, RK4 & symplectic integrators
* **Coupling tasks**:

  * Gravitational two-body (1D)
  * Coulomb coupling (1D)
  * Lorentz-force (2D) for charged particles in a magnetic field
* **Quantum-Gravity & Electromagnetism**: Graviton & Photon emission/absorption Tasks
* **UniversalConstructor**: Bootstraps any list of Tasks into a working Constructor
* **Pluggable Backend Architecture**: Modular task ontologies for different physics domains
* **Hydrogen atom constructors**: Excitation, deexcitation and two-atom collisions
* **Demo scripts**:

  * `demo.py` – shows every constructor in action
  * `bootstrap_demo.py` – elegant self-replication via the UniversalConstructor
  * `backend_demo.py` – demonstrates the pluggable backend system

---

## 🛠 Getting Started

### Prerequisites

* Python 3.8+
* (Optional) `matplotlib` for phase-space plots

### Installation

```bash
git clone https://github.com/gvelesandro/constructor-theory-simulator.git
cd constructor-theory-simulator
```

### Run Unit Tests

```bash
python -m unittest ct_tests.py
```

### Run Demos

```bash
python demo.py
python bootstrap_demo.py
python backend_demo.py
```

> **Note:** If you don’t have `matplotlib`, the demos will still run; plots will simply be skipped with a warning.

---

## 🎯 Usage Example

```python
from ct_framework import (
    Attribute, Substrate,
    PhotonEmissionTask, PhotonAbsorptionTask,
    UniversalConstructor, ascii_branch
)

# 1) Define your “program” of photon Tasks
ELEC = Attribute("charge_site")
prog = [
    PhotonEmissionTask(ELEC, emission_energy=5.0, carry_residual=False),
    PhotonAbsorptionTask(ELEC, absorption_energy=5.0)
]

# 2) Build a Constructor at runtime
uc      = UniversalConstructor()
em_cons = uc.build(prog)

# 3) Emit a photon
atom = Substrate("A", ELEC, energy=20.0)
branches = em_cons.perform(atom)
print(ascii_branch(branches))
# => * charge_site (A)
#    * photon      (A)

# 4) Absorb it back
photon   = next(w for w in branches if w.attr.label=="photon")
restored = em_cons.perform(photon)[0]
print(restored)
# => A:charge_site(E=20.0,Q=0,t=2,F=charge_site)
```

## 🔌 Pluggable Backend System

The framework now supports a modular backend architecture for different task ontologies:

```python
from ct_framework import (
    UniversalConstructor, Substrate, Attribute,
    ElectromagnetismBackend, QuantumGravityBackend
)

# 1) Use individual physics backends
uc = UniversalConstructor()

# Pure electromagnetism
em_constructor = uc.build_from_backends(["electromagnetism"])

# Pure quantum gravity  
qg_constructor = uc.build_from_backends(["quantum_gravity"])

# 2) Combine multiple backends
unified_constructor = uc.build_from_backends([
    "electromagnetism", 
    "quantum_gravity", 
    "hydrogen_atoms"
])

# 3) Create custom backends
class CustomPhysicsBackend:
    def get_tasks(self):
        return [/* your custom tasks */]
    
    def get_name(self):
        return "custom_physics"

# 4) Available built-in backends:
# - "electromagnetism": photon emission/absorption
# - "quantum_gravity": graviton emission/absorption  
# - "hydrogen_atoms": H excitation/deexcitation
# - "continuous_dynamics": integrator tasks
```

Run `python backend_demo.py` to see the full backend system in action!

---

## 🤝 Contributing

This is intended as an **educational resource** and proof-of-concept. Contributions are very welcome! Please:

* File issues for missing tasks or physics modules
* Submit pull requests for new constructors (e.g. chemical reactions, friction)
* Improve documentation or add more demos

---

## 📜 License

Released under the **MIT License**. 

---

## 🙏 Acknowledgments

* Inspired by David Deutsch and Chiara Marletto’s work in **Constructor Theory** and their May 13, 2025 paper “[Constructor Theory of Time](https://arxiv.org/abs/2505.08692).”
* Thanks to the quantum-foundations community for feedback and discussion.
