#!/usr/bin/env python3
"""
backend_demo.py

Demonstrates the pluggable backend architecture for task ontologies.
Shows how different physics domains can be loaded separately or combined.
"""

import time
from ct_framework import (
    # Backend system
    UniversalConstructor,
    BackendRegistry,
    ElectromagnetismBackend,
    QuantumGravityBackend,
    HydrogenBackend,
    ContinuousDynamicsBackend,
    get_global_registry,
    
    # Core types
    Substrate,
    Attribute,
    ascii_branch,
)


def demo_individual_backends():
    """Demonstrate using individual physics backends."""
    print("=== Individual Backend Demonstrations ===")
    
    uc = UniversalConstructor()
    
    # 1. Electromagnetism only
    print("\n--- Electromagnetism Backend ---")
    em_constructor = uc.build_from_backends(["electromagnetism"])
    
    charge_sub = Substrate("electron", Attribute("charge_site"), energy=20.0)
    em_worlds = em_constructor.perform(charge_sub)
    print("EM emission results:")
    for w in em_worlds:
        print(f"  {w}")
    
    # Absorb the photon back
    photon = next(w for w in em_worlds if w.attr.label == "photon")
    absorbed = em_constructor.perform(photon)
    print(f"After absorption: {absorbed[0]}")
    
    # 2. Quantum Gravity only
    print("\n--- Quantum Gravity Backend ---")
    qg_constructor = uc.build_from_backends(["quantum_gravity"])
    
    mass_sub = Substrate("particle", Attribute("mass"), energy=15.0)
    qg_worlds = qg_constructor.perform(mass_sub)
    print("QG emission results:")
    for w in qg_worlds:
        print(f"  {w}")
    
    # 3. Hydrogen atoms only
    print("\n--- Hydrogen Atom Backend ---")
    h_constructor = uc.build_from_backends(["hydrogen_atoms"])
    
    h_ground = Substrate("H1", Attribute("H_ground"), energy=0.0)
    h_excited = h_constructor.perform(h_ground)[0]
    print(f"Excited hydrogen: {h_excited}")
    
    h_decay = h_constructor.perform(h_excited)
    print("Decay products:")
    for w in h_decay:
        print(f"  {w}")


def demo_combined_backends():
    """Demonstrate combining multiple backends."""
    print("\n=== Combined Backend Demonstration ===")
    
    uc = UniversalConstructor()
    
    # Create a universal physics constructor
    physics_constructor = uc.build_from_backends([
        "electromagnetism",
        "quantum_gravity", 
        "hydrogen_atoms"
    ])
    
    print("\nUniversal physics constructor can handle:")
    
    # Test with different substrate types
    substrates = [
        Substrate("e-", Attribute("charge_site"), energy=25.0),
        Substrate("mass", Attribute("mass"), energy=18.0),
        Substrate("H", Attribute("H_ground"), energy=5.0),
    ]
    
    for sub in substrates:
        worlds = physics_constructor.perform(sub)
        print(f"\n{sub.name} ({sub.attr.label}) → {len(worlds)} outcomes:")
        for w in worlds:
            print(f"  {w}")


def demo_custom_backend():
    """Demonstrate creating and using a custom backend."""
    print("\n=== Custom Backend Demonstration ===")
    
    # Create custom attributes for a fictional particle physics scenario
    QUARK = Attribute("quark")
    GLUON = Attribute("gluon")
    
    # Define a custom backend for strong force interactions
    class StrongForceBackend:
        def get_tasks(self):
            from ct_framework import Task
            return [
                Task(
                    "quark_gluon_emission",
                    QUARK,
                    [(QUARK, -2.0, 0), (GLUON, 0, 0)],
                    quantum=True,
                    irreversible=True,
                    clock_inc=1
                ),
                Task(
                    "gluon_absorption",
                    GLUON,
                    [(QUARK, 2.0, 0)],
                    quantum=True,
                    clock_inc=1
                )
            ]
        
        def get_name(self):
            return "strong_force"
        
        def get_description(self):
            return "Strong force (QCD) interactions with quarks and gluons"
    
    # Create and register the backend
    strong_backend = StrongForceBackend()
    registry = BackendRegistry()
    registry.register(strong_backend)
    
    # Build constructor with the custom backend
    uc = UniversalConstructor(backend_registry=registry)
    strong_constructor = uc.build_with_backend(strong_backend)
    
    # Test the custom physics
    quark = Substrate("up_quark", QUARK, energy=10.0)
    worlds = strong_constructor.perform(quark)
    
    print(f"Custom strong force backend:")
    print(f"  Backend: {strong_backend.get_name()}")
    print(f"  Description: {strong_backend.get_description()}")
    print(f"  Tasks: {len(strong_backend.get_tasks())}")
    
    print(f"\nQuark interaction results:")
    for w in worlds:
        print(f"  {w}")
    
    # Test gluon absorption
    gluon = next(w for w in worlds if w.attr == GLUON)
    absorbed = strong_constructor.perform(gluon)
    print(f"After gluon absorption: {absorbed[0]}")


def demo_backend_swapping():
    """Demonstrate swapping between different backend configurations."""
    print("\n=== Backend Swapping Demonstration ===")
    
    uc = UniversalConstructor()
    
    # Same substrate, different physics backends
    test_sub = Substrate("particle", Attribute("charge_site"), energy=20.0)
    
    scenarios = [
        (["electromagnetism"], "Pure EM"),
        (["electromagnetism", "quantum_gravity"], "EM + Quantum Gravity"),
        (["electromagnetism", "hydrogen_atoms"], "EM + Hydrogen"),
        (["electromagnetism", "quantum_gravity", "hydrogen_atoms"], "Full Physics")
    ]
    
    for backend_names, description in scenarios:
        constructor = uc.build_from_backends(backend_names)
        worlds = constructor.perform(test_sub.clone())
        
        print(f"\n{description}:")
        print(f"  Backends: {', '.join(backend_names)}")
        print(f"  Outcomes: {len(worlds)}")
        for w in worlds:
            print(f"    {w}")


def main():
    """Run all backend demonstrations."""
    print("🔬 Constructor Theory: Pluggable Backend Architecture Demo")
    print("=" * 60)
    
    # Show available backends
    registry = get_global_registry()
    print(f"\nAvailable backends: {', '.join(registry.list_backends())}")
    
    demo_individual_backends()
    demo_combined_backends()
    demo_custom_backend()
    demo_backend_swapping()
    
    print("\n" + "=" * 60)
    print("✅ Backend architecture demonstration complete!")
    print("\nKey Benefits:")
    print("• Modular physics: Each domain can be used independently")
    print("• Composable: Mix and match different physics backends") 
    print("• Extensible: Easy to add new physics domains")
    print("• Testable: Isolate and test specific physics behaviors")


if __name__ == "__main__":
    main()