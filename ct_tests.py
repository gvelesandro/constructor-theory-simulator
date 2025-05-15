import time
import math
import unittest
import matplotlib.pyplot as plt
import ct_framework as ctf

A = ctf.Attribute  # shorthand


class TestCTFramework(unittest.TestCase):
    def setUp(self):
        # basic attributes
        self.g, self.e = A("g"), A("e")
        self.alt = A("alt")
        self.up, self.down = A("up"), A("down")
        self.entangled = A("ent")

        # single-substrate tasks
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
        c = 299_792_458
        β = 0.6
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
        out = mc.perform([cs1, cs2])[0]
        m1n, m2n = out
        self.assertNotEqual(m1n.p, cs1.p)
        self.assertNotEqual(m2n.p, cs2.p)

    # 23. phase-space visualiser
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
            "d2", 1.0, 2.0, px=0.0, py=0.0, mass=1.0, potential_fn=fn, dt=1.0
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
            "rk2d", 0.0, 0.0, px=1.0, py=2.0, mass=1.0, potential_fn=fn, dt=0.1
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
            "se2d", 0.0, 0.0, px=1.0, py=2.0, mass=1.0, potential_fn=fn, dt=0.1
        )
        w = ctf.SymplecticEuler2DTask().apply(cs2)[0]
        self.assertAlmostEqual(w.px, 1.0, places=5)
        self.assertAlmostEqual(w.py, 2.0, places=5)
        self.assertAlmostEqual(w.x, 0.1, places=5)
        self.assertAlmostEqual(w.y, 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
