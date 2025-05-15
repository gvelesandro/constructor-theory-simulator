import time
import math
import unittest
import ct_framework as ctf

# Shorthand
A = ctf.Attribute
ActionConstructor = ctf.ActionConstructor


class TestCTSuite(unittest.TestCase):
    def setUp(self):
        # basic attributes
        self.g, self.e = A("g"), A("e")
        self.alt = A("alt")
        self.up, self.down = A("up"), A("down")
        self.entangled = A("ent")

        # tasks
        excite = ctf.Task(
            "excite",
            self.g,
            [(self.e, -1, 0)],
            duration=0.001,
            clock_inc=1,
        )
        decay = ctf.Task(
            "decay",
            self.e,
            [(self.g, +1, 0), (self.alt, +1, 0)],
            duration=0.001,
            clock_inc=1,
        )
        entangle = ctf.Task(
            "ent",
            self.up,
            [(self.entangled, 0, 0)],
            quantum=True,
        )
        decohere = ctf.Task(
            "deco",
            self.entangled,
            [(self.up, 0, 0), (self.down, 0, 0)],
            irreversible=True,
        )  # one-way

        self.cons = ctf.Constructor([excite, decay, entangle, decohere])

    # 1. excite
    def test_excite(self):
        s = ctf.Substrate("e-", self.g, 1)
        out = self.cons.perform(s)[0]
        self.assertEqual(out.attr, self.e)

    # 2. many-worlds branching
    def test_many_worlds_decay(self):
        worlds = self.cons.perform(ctf.Substrate("e-", self.e, 2))
        self.assertEqual({w.attr for w in worlds}, {self.g, self.alt})

    # 3. irreversible task guard
    def test_irreversible(self):
        s = ctf.Substrate("x", self.entangled, 1)
        self.cons.perform(s)
        self.assertEqual(self.cons.perform(s), [])

    # 4. entanglement / decoherence branches
    def test_decoherence(self):
        a = ctf.Substrate("A", self.up, 1)
        a.entangled_with = a
        first_branch = self.cons.perform(a)[0]
        branches = self.cons.perform(first_branch)
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
            time.sleep(0.004)
        self.assertEqual(t1.attr, t2.attr)

    # 7. composition principle (two-step run)
    def test_composition(self):
        s = ctf.Substrate("p", self.g, 2)
        mid = self.cons.perform(s)[0]
        outs = self.cons.perform(mid)
        self.assertTrue(outs)

    # 8. charge conservation impossibility
    def test_charge_conservation(self):
        bad = ctf.Task("bad", self.g, [(self.e, -1, +1)])
        self.assertFalse(bad.possible(ctf.Substrate("ion", self.g, 1, charge=0)))

    # 9. clock increment
    def test_clock_increment(self):
        s = ctf.Substrate("clk", self.g, 1)
        out = self.cons.perform(s)[0]
        self.assertEqual(out.clock, 1)

    # 10. special-relativistic dilation
    def test_special_relativity(self):
        c = 299_792_458
        beta = 0.6
        γ = 1 / math.sqrt(1 - beta**2)
        fast = ctf.Substrate("μ", self.g, 1, velocity=beta * c)
        self.assertAlmostEqual(fast.adjusted_duration(0.02), 0.02 / γ, places=7)

    # 11. general-relativistic red-shift (weak-field)
    def test_gravity_redshift(self):
        c = 299_792_458
        g = 9.8
        deep = ctf.Substrate("clock", self.g, 1, grav=g)
        self.assertAlmostEqual(
            deep.adjusted_duration(0.02), 0.02 * (1 + g / c**2), places=10
        )

    # 12. fungibility predicate
    def test_fungibility(self):
        a = ctf.Substrate("a", self.g, 1, fungible_id="q")
        b = ctf.Substrate("b", self.g, 1, fungible_id="q")
        self.assertTrue(a.is_fungible_with(b))

    # 13. free swap of fungible substrates
    def test_swap_constructor(self):
        a = ctf.Substrate("a", self.g, 1, fungible_id="q")
        b = ctf.Substrate("b", self.g, 1, fungible_id="q")
        ctf.SwapConstructor.swap(a, b)
        self.assertEqual({a.name, b.name}, {"a", "b"})

    # 14. ascii visualiser
    def test_ascii_branch(self):
        worlds = self.cons.perform(ctf.Substrate("y", self.e, 2))
        art = ctf.ascii_branch(worlds)
        self.assertIn("* g", art)

    # 15. clock constructor tick
    def test_clock_constructor(self):
        clk = ctf.Substrate("tick", A("clock"), 1)
        ticker = ctf.ClockConstructor(0.001)
        ticker.perform(clk)
        self.assertEqual(clk.clock, 1)

    # 16. Mach–Zehnder interference demo (minimal)
    def test_mach_zehnder(self):
        bs = ctf.Task("BS", A("src"), [(A("p1"), 0, 0), (A("p2"), 0, 0)], quantum=True)
        rc = ctf.Task("RC", A("p1"), [(A("int"), 0, 0)], quantum=True)
        mz = ctf.Constructor([bs, rc])
        ph = ctf.Substrate("γ", A("src"), 1)
        paths = mz.perform(ph)
        out = mz.perform(paths[0])[0]
        self.assertEqual(out.attr.label, "int")

    # 17. irreversible guard second form (call on output)
    def test_irrev_on_output(self):
        s = ctf.Substrate("z", self.entangled, 1)
        branch = self.cons.perform(s)[0]
        self.assertEqual(self.cons.perform(branch), [])

    # 18. null constructor identity for composition
    def test_null_identity(self):
        s = ctf.Substrate("id", self.g, 0)
        self.assertEqual(ctf.NullConstructor().perform(s)[0].attr, self.g)

    # 19. adjusted-duration identity when no velocity/grav
    def test_duration_identity(self):
        s = ctf.Substrate("rest", self.g, 0)
        self.assertEqual(s.adjusted_duration(0.01), 0.01)

    # 20. single-task ActionConstructor behaves like normal
    def test_action_single(self):
        inp = A("in")
        out = A("out")
        t = ctf.Task("only", inp, [(out, 0, 0)], action_cost=0.5)
        ac = ActionConstructor([t])
        result = ac.perform(ctf.Substrate("X", inp, 1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].attr, out)

    # 21. principle of least action selection
    def test_least_action(self):
        inp = A("in")
        low = ctf.Task("low", inp, [(A("A"), 0, 0)], action_cost=1.0)
        high = ctf.Task("high", inp, [(A("B"), 0, 0)], action_cost=2.0)
        ac = ActionConstructor([low, high])
        result = ac.perform(ctf.Substrate("X", inp, 1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].attr.label, "A")


if __name__ == "__main__":
    unittest.main()
