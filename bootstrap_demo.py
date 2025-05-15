#!/usr/bin/env python3
"""
bootstrap_demo.py

A “universal constructor” that at runtime assembles
both photon‐ and graviton‐tasks into one mega‐constructor,
and then uses it to process different substrates:
• mass → graviton emission/absorption
• charge_site → photon emission/absorption
"""

import time
from ct_framework import (
    UniversalConstructor,
    PhotonEmissionTask,
    PhotonAbsorptionTask,
    GravitonEmissionTask,
    GravitonAbsorptionTask,
    Substrate,
    Attribute,
    ascii_branch,
)


def main():
    # 1) Pick our physical “program”: EM + QG tasks
    ELEC = Attribute("charge_site")
    MASS = Attribute("mass")
    ΔE_ph = 5.0  # photon energy
    ΔE_gr = 3.0  # graviton energy

    program = [
        PhotonEmissionTask(ELEC, emission_energy=ΔE_ph),
        PhotonAbsorptionTask(ELEC, absorption_energy=ΔE_ph),
        GravitonEmissionTask(MASS, emission_energy=ΔE_gr),
        GravitonAbsorptionTask(MASS, absorption_energy=ΔE_gr),
    ]

    # 2) Bootstrap it!
    uc = UniversalConstructor()
    mega = uc.build(program)

    # 3) Show it in action on a “mass” substrate:
    print("=== Graviton demo via universal constructor ===")
    m0 = Substrate("m0", MASS, energy=10.0)
    branches = mega.perform(m0)
    print("Branches after emission:")
    print(ascii_branch(branches))
    # pick out the graviton branch, then absorb
    graviton = next(w for w in branches if w.attr.label == "graviton")
    restored = mega.perform(graviton)[0]
    print("After absorption: ", restored)
    print()

    time.sleep(0.5)

    # 4) And now electromagnetism on a “charge_site”:
    print("=== Photon demo via universal constructor ===")
    e0 = Substrate("e0", ELEC, energy=20.0)
    branches = mega.perform(e0)
    print("Branches after emission:")
    print(ascii_branch(branches))
    photon = next(w for w in branches if w.attr.label == "photon")
    restored = mega.perform(photon)[0]
    print("After absorption: ", restored)
    print()

    time.sleep(0.5)

    # 5) Finally, show that you can interleave them at will:
    print("=== Mixed sequence: emit photon, then graviton ===")
    # start with mass, emit graviton
    m1 = Substrate("m1", MASS, energy=8.0)
    gm_br = mega.perform(m1)
    # take the non-graviton branch (the residual-mass branch)
    m_res = next(w for w in gm_br if w.attr == MASS)
    # now treat that mass as an EM charge site as well:
    m_res.attr = ELEC
    print("Now EM‐branch on same residual:")
    em_br = mega.perform(m_res)
    print(ascii_branch(em_br))
    print("\nDone.")


if __name__ == "__main__":
    main()
