import time
import math
import unittest
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:  # pragma: no cover - optional dependency
    plt = None
    HAS_PLT = False
import ct_framework as ctf

A = ctf.Attribute  # shorthand


class TestCTFramework(unittest.TestCase):
    def setUp(self):
        # basic attributes
        self.g, self.e = A("g"), A("e")
        self.alt = A("alt")
        self.up, self.down = A("up"), A("down")
        self.entangled = A("ent")

        excite = ctf.Task(
            "excite", self.g, [(self.e, -1, 0)], duration=0.001, clock_inc=1
        )
        decay = ctf.Task(
            "decay",
            self.e,
            [(self.g, +1, 0), (self.alt, +1, 0)],
            duration=0.001,
            clock_inc=1,
        )
        ent = ctf.Task("ent", self.up, [(self.entangled, 0, 0)], quantum=True)
        deco = ctf.Task(
            "deco",
            self.entangled,
            [(self.up, 0, 0), (self.down, 0, 0)],
            irreversible=True,
        )

        self.cons = ctf.Constructor([excite, decay, ent, deco])

    # 1. excite
    def test_excite(self):
        s = ctf.Substrate("e-", self.g, 1)
        out = self.cons.perform(s)[0]
        self.assertEqual(out.attr, self.e)

    # 2. many-worlds decay
    def test_many_worlds_decay(self):
        worlds = self.cons.perform(ctf.Substrate("e-", self.e, 2))
        self.assertEqual({w.attr for w in worlds}, {self.g, self.alt})

    # 3. irreversible guard
    def test_irreversible(self):
        s = ctf.Substrate("x", self.entangled, 1)
        self.cons.perform(s)
        self.assertEqual(self.cons.perform(s), [])

    # 4. decoherence branches
    def test_decoherence(self):
        a = ctf.Substrate("A", self.up, 1)
        a.entangled_with = a
        first = self.cons.perform(a)[0]
        branches = self.cons.perform(first)
        self.assertEqual({b.attr for b in branches}, {self.up, self.down})

    # 5. null constructor identity
    def test_null_constructor(self):
        n = ctf.NullConstructor()
        s = ctf.Substrate("idle", self.g, 0)
        self.assertIs(n.perform(s)[0], s)

    # 6. timer synchrony
    def test_timer(self):
        t1 = ctf.TimerSubstrate("T1", 0.03)
        t2 = ctf.TimerSubstrate("T2", 0.03)
        T = ctf.TimerConstructor()
        while t1.attr.label != "halted" or t2.attr.label != "halted":
            T.perform(t1)
            T.perform(t2)
            time.sleep(0.005)
        self.assertEqual(t1.attr, t2.attr)

    # 7. composition principle
    def test_composition(self):
        s = ctf.Substrate("p", self.g, 2)
        mid = self.cons.perform(s)[0]
        outs = self.cons.perform(mid)
        self.assertTrue(outs)

    # 8. charge conservation
    def test_charge_conservation(self):
        bad = ctf.Task("bad", self.g, [(self.e, -1, +1)])
        self.assertFalse(bad.possible(ctf.Substrate("ion", self.g, 1, charge=0)))

    # 9. clock increment
    def test_clock_increment(self):
        out = self.cons.perform(ctf.Substrate("clk", self.g, 1))[0]
        self.assertEqual(out.clock, 1)

    # 10. special relativity
    def test_special_relativity(self):
        c, β = 299_792_458, 0.6
        γ = 1 / math.sqrt(1 - β**2)
        fast = ctf.Substrate("μ", self.g, 1, velocity=β * c)
        self.assertAlmostEqual(fast.adjusted_duration(0.02), 0.02 / γ, places=7)

    # 11. general relativity
    def test_gravity_redshift(self):
        c, g = 299_792_458, 9.8
        deep = ctf.Substrate("clock", self.g, 1, grav=g)
        self.assertAlmostEqual(
            deep.adjusted_duration(0.02), 0.02 * (1 + g / c**2), places=10
        )

    # 12. fungibility
    def test_fungibility(self):
        a = ctf.Substrate("a", self.g, 1, fungible_id="q")
        b = ctf.Substrate("b", self.g, 1, fungible_id="q")
        self.assertTrue(a.is_fungible_with(b))

    # 13. swap fungibles
    def test_swap(self):
        a = ctf.Substrate("a", self.g, 1, fungible_id="q")
        b = ctf.Substrate("b", self.g, 1, fungible_id="q")
        ctf.SwapConstructor.swap(a, b)
        self.assertEqual({a.name, b.name}, {"a", "b"})

    # 14. ascii branch
    def test_ascii_branch(self):
        art = ctf.ascii_branch(self.cons.perform(ctf.Substrate("y", self.e, 2)))
        self.assertIn("* g", art)

    # 15. clock constructor tick
    def test_clock_constructor(self):
        clk = ctf.Substrate("tick", A("clock"), 1)
        ticker = ctf.ClockConstructor(0.001)
        ticker.perform(clk)
        self.assertEqual(clk.clock, 1)

    # 16. Mach–Zehnder interference
    def test_mach_zehnder(self):
        bs = ctf.Task("BS", A("src"), [(A("p1"), 0, 0), (A("p2"), 0, 0)], quantum=True)
        rc = ctf.Task("RC", A("p1"), [(A("int"), 0, 0)], quantum=True)
        mz = ctf.Constructor([bs, rc])
        ph = ctf.Substrate("γ", A("src"), 1)
        paths = mz.perform(ph)
        out = mz.perform(paths[0])[0]
        self.assertEqual(out.attr.label, "int")

    # 17. irreversible guard on output
    def test_irrev_on_output(self):
        s = ctf.Substrate("z", self.entangled, 1)
        branch = self.cons.perform(s)[0]
        self.assertEqual(self.cons.perform(branch), [])

    # 18. null identity
    def test_null_identity(self):
        s = ctf.Substrate("id", self.g, 0)
        self.assertEqual(ctf.NullConstructor().perform(s)[0].attr, self.g)

    # 19. duration identity
    def test_duration_identity(self):
        s = ctf.Substrate("rest", self.g, 0)
        self.assertEqual(s.adjusted_duration(0.01), 0.01)

    # 20. ActionConstructor single
    def test_action_single(self):
        inp, out = A("in"), A("out")
        t = ctf.Task("only", inp, [(out, 0, 0)], action_cost=0.5)
        ac = ctf.ActionConstructor([t])
        res = ac.perform(ctf.Substrate("X", inp, 1))
        self.assertEqual(res[0].attr, out)

    # 21. least-action selection
    def test_least_action(self):
        inp = A("in")
        low = ctf.Task("low", inp, [(A("A"), 0, 0)], action_cost=1.0)
        high = ctf.Task("high", inp, [(A("B"), 0, 0)], action_cost=2.0)
        ac = ctf.ActionConstructor([low, high])
        res = ac.perform(ctf.Substrate("X", inp, 1))
        self.assertEqual(res[0].attr.label, "A")

    # 22. gravitational coupling 1D
    def test_grav_coupling_1d(self):
        cs1 = ctf.ContinuousSubstrate(
            "m1", 0.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1
        )
        cs2 = ctf.ContinuousSubstrate(
            "m2", 1.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1
        )
        task = ctf.MultiSubstrateTask(
            "grav", [cs1.attr, cs2.attr], ctf.grav_coupling_fn
        )
        mc = ctf.MultiConstructor([task])
        m1n, m2n = mc.perform([cs1, cs2])[0]
        self.assertNotEqual(m1n.p, cs1.p)
        self.assertNotEqual(m2n.p, cs2.p)

    # 23. phase-space visualiser
    @unittest.skipUnless(HAS_PLT, "matplotlib not installed")
    def test_phase_space(self):
        orig = plt.show
        plt.show = lambda *a, **k: None
        cs = ctf.ContinuousSubstrate(
            "t", 0.0, 1.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1
        )
        cs2 = cs.clone()
        cs2.x += cs2.p / cs2.mass * cs2.dt
        cs2.clock += 1
        ctf.plot_phase_space({"traj": [cs, cs2]})
        plt.show = orig

    # 24. DynamicsTask (1D)
    def test_dynamics_task_1d(self):
        pot = lambda x: 0.5 * x * x
        cs = ctf.ContinuousSubstrate("cs", 1.0, 0.0, mass=1.0, potential_fn=pot, dt=0.1)
        w = ctf.DynamicsTask().apply(cs)[0]
        self.assertAlmostEqual(w.p, -0.1, places=5)
        self.assertAlmostEqual(w.x, 0.99, places=5)

    # 25. RK4Task (1D)
    def test_rk4_task_1d(self):
        pot = lambda x: 0.5 * x * x
        cs = ctf.ContinuousSubstrate("rk", 1.0, 0.0, mass=1.0, potential_fn=pot, dt=0.1)
        w = ctf.RK4Task().apply(cs)[0]
        self.assertAlmostEqual(w.p, -math.sin(cs.dt), places=5)
        self.assertAlmostEqual(w.x, math.cos(cs.dt), places=5)

    # 26. SymplecticEulerTask (1D)
    def test_symp_euler_task_1d(self):
        pot = lambda x: 0.5 * x * x
        cs = ctf.ContinuousSubstrate("se", 1.0, 0.0, mass=1.0, potential_fn=pot, dt=0.1)
        w = ctf.SymplecticEulerTask().apply(cs)[0]
        self.assertAlmostEqual(w.p, -0.1, places=5)
        self.assertAlmostEqual(w.x, 0.99, places=5)

    # 27. ContinuousSubstrate2D.clone
    def test_continuous_substrate2d_clone(self):
        fn = lambda x, y: x * y
        cs2 = ctf.ContinuousSubstrate2D(
            "2d", x=1.0, y=2.0, px=3.0, py=4.0, mass=5.0, potential_fn=fn, dt=0.1
        )
        clone = cs2.clone()
        self.assertEqual(clone.x, cs2.x)
        self.assertEqual(clone.y, cs2.y)
        self.assertEqual(clone.px, cs2.px)
        self.assertEqual(clone.py, cs2.py)
        self.assertIs(clone.potential2d, fn)

    # 28. Dynamics2DTask
    def test_dynamics2d_task(self):
        fn = lambda x, y: x + y
        cs2 = ctf.ContinuousSubstrate2D(
            "d2", x=1.0, y=2.0, px=0.0, py=0.0, mass=1.0, potential_fn=fn, dt=1.0
        )
        w = ctf.Dynamics2DTask().apply(cs2)[0]
        self.assertAlmostEqual(w.px, -1.0, places=5)
        self.assertAlmostEqual(w.py, -1.0, places=5)
        self.assertAlmostEqual(w.x, 0.0, places=5)
        self.assertAlmostEqual(w.y, 1.0, places=5)

    # 29. RK42DTask constant potential
    def test_rk42d_task_constant(self):
        fn = lambda x, y: 0.0
        cs2 = ctf.ContinuousSubstrate2D(
            "rk2d", x=0.0, y=0.0, px=1.0, py=2.0, mass=1.0, potential_fn=fn, dt=0.1
        )
        w = ctf.RK42DTask().apply(cs2)[0]
        self.assertAlmostEqual(w.px, 1.0, places=5)
        self.assertAlmostEqual(w.py, 2.0, places=5)
        self.assertAlmostEqual(w.x, 0.1, places=5)
        self.assertAlmostEqual(w.y, 0.2, places=5)

    # 30. SymplecticEuler2DTask constant potential
    def test_sympeuler2d_task_constant(self):
        fn = lambda x, y: 0.0
        cs2 = ctf.ContinuousSubstrate2D(
            "se2d", x=0.0, y=0.0, px=1.0, py=2.0, mass=1.0, potential_fn=fn, dt=0.1
        )
        w = ctf.SymplecticEuler2DTask().apply(cs2)[0]
        self.assertAlmostEqual(w.px, 1.0, places=5)
        self.assertAlmostEqual(w.py, 2.0, places=5)
        self.assertAlmostEqual(w.x, 0.1, places=5)
        self.assertAlmostEqual(w.y, 0.2, places=5)

    # 31. graviton emission
    def test_graviton_emission(self):
        MASS = A("mass")
        ΔE = 3.0
        emit = ctf.GravitonEmissionTask(MASS, emission_energy=ΔE)
        cons = ctf.Constructor([emit])
        m0 = ctf.Substrate("m", MASS, energy=10.0)
        worlds = cons.perform(m0)
        mass_br = [b for b in worlds if b.attr == MASS]
        grav_br = [b for b in worlds if b.attr == ctf.GRAVITON]
        self.assertEqual(len(mass_br), 1)
        self.assertEqual(len(grav_br), 1)
        self.assertAlmostEqual(mass_br[0].energy, 10.0 - ΔE, places=5)

    # 32. graviton absorption
    def test_graviton_absorption(self):
        MASS = A("mass")
        ΔE = 2.5
        absorb = ctf.GravitonAbsorptionTask(MASS, absorption_energy=ΔE)
        cons = ctf.Constructor([absorb])
        g_sub = ctf.Substrate("g", ctf.GRAVITON, energy=0.0)
        branches = cons.perform(g_sub)
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0].attr, MASS)
        self.assertAlmostEqual(branches[0].energy, ΔE, places=5)

    # 33. QuantumGravityConstructor end-to-end
    def test_quantum_gravity_constructor(self):
        MASS = A("mass")
        ΔE = 4.0
        qg = ctf.QuantumGravityConstructor(MASS, ΔE)
        m0 = ctf.Substrate("m", MASS, energy=12.0)
        worlds = qg.perform(m0)
        mass_w = [w for w in worlds if w.attr == MASS]
        grav_w = [w for w in worlds if w.attr == ctf.GRAVITON]
        self.assertEqual(len(mass_w), 1)
        self.assertEqual(len(grav_w), 1)
        self.assertAlmostEqual(mass_w[0].energy, 12.0 - ΔE, places=5)

        # now absorb
        absorb_worlds = qg.perform(grav_w[0])
        self.assertEqual(absorb_worlds[0].attr, MASS)
        self.assertAlmostEqual(absorb_worlds[0].energy, ΔE, places=5)

    # 34. photon emission
    def test_photon_emission(self):
        ELEC = A("charge_site")
        ΔE = 5.0
        emit = ctf.PhotonEmissionTask(ELEC, emission_energy=ΔE)
        cons = ctf.Constructor([emit])
        s0 = ctf.Substrate("S", ELEC, energy=20.0)
        worlds = cons.perform(s0)
        # one world with reduced energy, one photon
        self.assertEqual(len(worlds), 2)
        sch, ph = sorted(worlds, key=lambda w: w.attr.label)
        self.assertEqual(ph.attr, ctf.PHOTON)
        self.assertAlmostEqual(sch.energy, 20.0 - ΔE, places=6)

    # 35. photon absorption
    def test_photon_absorption(self):
        ELEC = A("charge_site")
        ΔE = 2.5
        absrt = ctf.PhotonAbsorptionTask(ELEC, absorption_energy=ΔE)
        cons = ctf.Constructor([absrt])
        ph = ctf.Substrate("γ", ctf.PHOTON, energy=0.0)
        worlds = cons.perform(ph)
        self.assertEqual(len(worlds), 1)
        self.assertEqual(worlds[0].attr, ELEC)
        self.assertAlmostEqual(worlds[0].energy, ΔE, places=6)

    # 36. Coulomb coupling 1D
    def test_coulomb_coupling_1d(self):
        # two positive charges repel
        cs1 = ctf.ContinuousSubstrate(
            "c1", -1.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1, charge=+1
        )
        cs2 = ctf.ContinuousSubstrate(
            "c2", +1.0, 0.0, mass=1.0, potential_fn=lambda x: 0.0, dt=0.1, charge=+1
        )
        task = ctf.MultiSubstrateTask(
            "coulomb", [cs1.attr, cs2.attr], ctf.coulomb_coupling_fn
        )
        mc = ctf.MultiConstructor([task])
        out1, out2 = mc.perform([cs1, cs2])[0]
        # they should have opposite momentum changes
        self.assertNotEqual(out1.p, cs1.p)
        self.assertAlmostEqual(out1.p, -out2.p, places=6)

    # 37. Lorentz force coupling (2D)
    def test_lorentz_force_coupling_2d(self):
        # set up a charged 2D substrate
        cs = ctf.ContinuousSubstrate2D(
            "e-",
            x=0.0,
            y=1.0,
            px=2.0,
            py=3.0,
            mass=1.0,
            potential_fn=lambda x, y: 0.0,
            dt=0.1,
            charge=+1,
        )
        # uniform Bz = 0.5 outward
        B = ctf.FieldSubstrate("B", Bz=0.5)
        # wire up the multi‐substrate lorentz task
        task = ctf.MultiSubstrateTask(
            "lorentz", [cs.attr, B.attr], ctf.lorentz_coupling_fn
        )
        mc = ctf.MultiConstructor([task])

        out_cs, out_B = mc.perform([cs, B])[0]

        # expected impulses
        expected_Fx = cs.charge * cs.py * B.Bz
        expected_Fy = -cs.charge * cs.px * B.Bz

        self.assertAlmostEqual(out_cs.px, cs.px + expected_Fx * cs.dt, places=6)
        self.assertAlmostEqual(out_cs.py, cs.py + expected_Fy * cs.dt, places=6)
        # B‐field should be unchanged
        self.assertEqual(out_B.Bz, B.Bz)


    # 38. Hydrogen atom excitation and deexcitation
    def test_hydrogen_excitation_cycle(self):
        gap = 10.2
        cons = ctf.HydrogenAtomConstructor(gap)
        h = ctf.Substrate("H", ctf.HYDROGEN_GROUND, energy=0.0)
        excited = cons.perform(h)[0]
        self.assertEqual(excited.attr, ctf.HYDROGEN_EXCITED)
        self.assertAlmostEqual(excited.energy, gap, places=6)
        worlds = cons.perform(excited)
        attrs = {w.attr for w in worlds}
        self.assertIn(ctf.HYDROGEN_GROUND, attrs)
        self.assertIn(ctf.PHOTON, attrs)

    # 39. Hydrogen collision leading to molecule
    def test_hydrogen_collision_bond(self):
        hi = ctf.HydrogenInteractionConstructor(bond_energy=4.5)
        h1 = ctf.Substrate("h1", ctf.HYDROGEN_GROUND, energy=3.0)
        h2 = ctf.Substrate("h2", ctf.HYDROGEN_GROUND, energy=3.0)
        result = hi.perform([h1, h2])[0]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].attr, ctf.HYDROGEN_MOLECULE)

    # 40. Hydrogen collision insufficient energy
    def test_hydrogen_collision_no_bond(self):
        hi = ctf.HydrogenInteractionConstructor(bond_energy=4.5)
        h1 = ctf.Substrate("ha", ctf.HYDROGEN_GROUND, energy=1.0)
        h2 = ctf.Substrate("hb", ctf.HYDROGEN_GROUND, energy=1.0)
        result = hi.perform([h1, h2])[0]
        self.assertEqual(len(result), 2)
        self.assertEqual({s.attr for s in result}, {ctf.HYDROGEN_GROUND})

    # 41. Backend registry functionality
    def test_backend_registry(self):
        # Create a new registry for testing
        registry = ctf.BackendRegistry()
        
        # Create a simple test backend
        test_backend = ctf.ElectromagnetismBackend(A("test_charge"), 2.0)
        
        # Register the backend
        registry.register(test_backend)
        
        # Test retrieval
        retrieved = registry.get_backend("electromagnetism")
        self.assertEqual(retrieved.get_name(), "electromagnetism")
        
        # Test listing
        backends = registry.list_backends()
        self.assertIn("electromagnetism", backends)
        
        # Test getting all tasks
        tasks = registry.get_all_tasks(["electromagnetism"])
        self.assertEqual(len(tasks), 2)  # emission + absorption
        self.assertTrue(all(isinstance(t, ctf.Task) for t in tasks))

    # 42. Global registry default backends
    def test_global_registry_defaults(self):
        registry = ctf.get_global_registry()
        expected_backends = [
            "electromagnetism", 
            "quantum_gravity", 
            "hydrogen_atoms", 
            "continuous_dynamics"
        ]
        
        for backend_name in expected_backends:
            self.assertIn(backend_name, registry.list_backends())
            backend = registry.get_backend(backend_name)
            self.assertIsInstance(backend, ctf.TaskOntologyBackend)
            tasks = backend.get_tasks()
            self.assertTrue(len(tasks) > 0)

    # 43. UniversalConstructor with backends
    def test_universal_constructor_with_backends(self):
        uc = ctf.UniversalConstructor()
        
        # Test building from specific backends
        em_constructor = uc.build_from_backends(["electromagnetism"])
        self.assertIsInstance(em_constructor, ctf.Constructor)
        
        # Test with single backend
        qg_backend = ctf.QuantumGravityBackend(A("mass"), 5.0)
        qg_constructor = uc.build_with_backend(qg_backend)
        self.assertIsInstance(qg_constructor, ctf.Constructor)
        
        # Test that tasks work
        mass_sub = ctf.Substrate("m", A("mass"), energy=10.0)
        worlds = qg_constructor.perform(mass_sub)
        self.assertEqual(len(worlds), 2)  # mass + graviton

    # 44. Backend task execution
    def test_backend_task_execution(self):
        # Test electromagnetism backend
        em_backend = ctf.ElectromagnetismBackend(A("charge"), 3.0)
        uc = ctf.UniversalConstructor()
        em_constructor = uc.build_with_backend(em_backend)
        
        charge_sub = ctf.Substrate("e", A("charge"), energy=10.0)
        worlds = em_constructor.perform(charge_sub)
        self.assertEqual(len(worlds), 2)  # charge + photon
        
        # Find photon and test absorption
        photon = next(w for w in worlds if w.attr == ctf.PHOTON)
        absorbed = em_constructor.perform(photon)
        self.assertEqual(len(absorbed), 1)
        self.assertEqual(absorbed[0].attr, A("charge"))

    # 45. Multiple backend combination
    def test_multiple_backend_combination(self):
        uc = ctf.UniversalConstructor()
        
        # Build constructor with multiple backends
        multi_constructor = uc.build_from_backends([
            "electromagnetism", 
            "quantum_gravity"
        ])
        
        # Test with charge substrate
        charge_sub = ctf.Substrate("e", A("charge_site"), energy=20.0)
        em_worlds = multi_constructor.perform(charge_sub)
        self.assertEqual(len(em_worlds), 2)  # charge + photon
        
        # Test with mass substrate
        mass_sub = ctf.Substrate("m", A("mass"), energy=15.0)
        qg_worlds = multi_constructor.perform(mass_sub)
        self.assertEqual(len(qg_worlds), 2)  # mass + graviton

    # 46. Custom backend creation
    def test_custom_backend_creation(self):
        # Create a custom backend
        class TestBackend(ctf.TaskOntologyBackend):
            def get_tasks(self):
                return [
                    ctf.Task("test_task", A("test_in"), [(A("test_out"), 0, 0)])
                ]
            
            def get_name(self) -> str:
                return "test_backend"
        
        custom_backend = TestBackend()
        uc = ctf.UniversalConstructor()
        custom_constructor = uc.build_with_backend(custom_backend)
        
        test_sub = ctf.Substrate("test", A("test_in"), energy=1.0)
        result = custom_constructor.perform(test_sub)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].attr, A("test_out"))

    # 47. Backend error handling
    def test_backend_error_handling(self):
        registry = ctf.BackendRegistry()
        
        # Test getting non-existent backend
        with self.assertRaises(ValueError):
            registry.get_backend("non_existent")
        
        # Test that error message contains backend name
        try:
            registry.get_backend("missing_backend")
        except ValueError as e:
            self.assertIn("missing_backend", str(e))


if __name__ == "__main__":
    unittest.main()
