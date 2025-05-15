from ct_framework import (
    UniversalConstructor,
    PhotonEmissionTask,
    PhotonAbsorptionTask,
    Substrate,
    Attribute,
)

# 1) Prepare a “program” of photon tasks
ELEC = Attribute("charge_site")
ΔE = 5.0
program = [
    PhotonEmissionTask(ELEC, emission_energy=ΔE),
    PhotonAbsorptionTask(ELEC, absorption_energy=ΔE),
]

# 2) Have the UC build us a fresh EM‐constructor
uc = UniversalConstructor()
em_cons = uc.build(program)

# 3) Use it exactly like EMConstructor would
s0 = Substrate("A", ELEC, energy=20.0)
branches = em_cons.perform(s0)
print(branches)  # → one charge_site‐branch at 15.0J, one photon‐branch

# 4) Absorb back:
photon = next(w for w in branches if w.attr.label == "photon")
restored = em_cons.perform(photon)[0]
print(restored.energy)  # → back to 20.0J
