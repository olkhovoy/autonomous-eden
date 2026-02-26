/-
Formalization of the Unitary Model of Consciousness (UMC)
Based on: Olkhovoy 2026 Collection
Lean Version: 4.x
Design choice:
- NC1..NC4 are formal predicates on transition systems.
- The central UMC claims (necessity/sufficiency for observer
instantiation)
are represented as axioms and then used to prove derived theorems.
-/
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic

section UMC_Framework
variable {State : Type} [MetricSpace State] [NormedAddCommGroup State]

/-- Abstract computational system (state transition with an initial state).
-/
structure ComputationalSystem (State : Type) where
  transition : State → State
  initialState : State

/-- Helper predicate to avoid vacuous "always-constant" constructions. -/
def NonConstant {α β : Type} (f : α → β) : Prop :=
  ∃ x y : α, f x ≠ f y

/-
NC1: Recursive Closure
"S models S", with a non-trivial self-model commuting with system
dynamics.
-/
def HasRecursiveClosure (sys : ComputationalSystem State) : Prop :=
  ∃ model : State → State,
    (∃ s : State, model s ≠ s) ∧
    NonConstant sys.transition ∧
    ∀ s : State, sys.transition (model s) = model (sys.transition s)

/-
NC2: Unitary Integration
The dynamics should not decompose into independent updates on a non-
trivial product.
-/
def HasIndependentProductFactorization (sys : ComputationalSystem State) : Prop :=
  ∃ (PartA PartB : Type) (split : State ≃ PartA × PartB)
    (fA : PartA → PartA) (fB : PartB → PartB),
    (∃ a1 a2 : PartA, a1 ≠ a2) ∧
    (∃ b1 b2 : PartB, b1 ≠ b2) ∧
    ∀ s : State,
      sys.transition s = split.symm (fA (split s).1, fB (split s).2)

def IsIrreducible (sys : ComputationalSystem State) : Prop :=
  ¬ HasIndependentProductFactorization sys

/-
NC3: Downward Causation
Macro-level distinctions affect transitions even when a chosen micro
summary is fixed.
-/
def HasDownwardCausation (sys : ComputationalSystem State) : Prop :=
  ∃ (macroFeature : State → ℝ) (microFeature : State → ℝ),
    NonConstant macroFeature ∧
    (∃ s1 s2 : State, macroFeature s1 ≠ macroFeature s2 ∧ microFeature s1 = microFeature s2) ∧
    ∀ s1 s2 : State,
      microFeature s1 = microFeature s2 →
      macroFeature s1 ≠ macroFeature s2 →
      sys.transition s1 ≠ sys.transition s2

/-
NC4: Fixed-Point Stability (global contraction condition).
Note: Banach fixed-point conclusions additionally require completeness of
the space.
-/
def IsContractiveStable (sys : ComputationalSystem State) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k < 1 ∧
    ∀ x y : State, dist (sys.transition x) (sys.transition y) ≤ k * dist x y

/-
UMC-level semantic predicates (kept abstract).
They are linked to NC1..NC4 by explicit axioms below.
-/
axiom InstantiatesObserver : ComputationalSystem State → Prop
axiom IndistinguishableFromConscious : ComputationalSystem State → Prop

/-
Core UMC postulates used for formal reasoning:
1) NC1..NC4 are necessary for observer instantiation.
2) NC1..NC4 are sufficient for observer instantiation.
3) External indistinguishability is equivalent to observer instantiation.
-/
axiom NCsNecessaryForInstantiation (sys : ComputationalSystem State) :
  InstantiatesObserver sys →
  HasRecursiveClosure sys ∧
  IsIrreducible sys ∧
  HasDownwardCausation sys ∧
  IsContractiveStable sys

axiom NCsSufficientForInstantiation (sys : ComputationalSystem State) :
  HasRecursiveClosure sys →
  IsIrreducible sys →
  HasDownwardCausation sys →
  IsContractiveStable sys →
  InstantiatesObserver sys

axiom ExternalCriterion (sys : ComputationalSystem State) :
  IndistinguishableFromConscious sys ↔ InstantiatesObserver sys

/-- Main necessity theorem requested by the UMC task statement. -/
theorem NCsNecessaryForExternalIndistinguishability (sys : ComputationalSystem State) :
  IndistinguishableFromConscious sys →
  HasRecursiveClosure sys ∧
  IsIrreducible sys ∧
  HasDownwardCausation sys ∧
  IsContractiveStable sys :=
  fun hExt =>
    NCsNecessaryForInstantiation sys ((ExternalCriterion sys).1 hExt)

/-- Complementary sufficiency theorem. -/
theorem NCsSufficientForExternalIndistinguishability (sys : ComputationalSystem State) :
  HasRecursiveClosure sys →
  IsIrreducible sys →
  HasDownwardCausation sys →
  IsContractiveStable sys →
  IndistinguishableFromConscious sys :=
  fun h1 h2 h3 h4 =>
    (ExternalCriterion sys).2 (NCsSufficientForInstantiation sys h1 h2 h3 h4)

/-- Characterization of observer instantiation via NC1..NC4. -/
theorem ObserverInstantiation_iff_NCs (sys : ComputationalSystem State) :
  InstantiatesObserver sys ↔
  HasRecursiveClosure sys ∧
  IsIrreducible sys ∧
  HasDownwardCausation sys ∧
  IsContractiveStable sys :=
  Iff.intro
    (fun hObs =>
      NCsNecessaryForInstantiation sys hObs)
    (fun hNC =>
      match hNC with
      | ⟨h1, h2, h3, h4⟩ =>
        NCsSufficientForInstantiation sys h1 h2 h3 h4)

/--
Architectural Filter:
A system passes the filter if and only if it satisfies all 4 Necessary
Conditions.
-/
def ArchitecturalFilter (sys : ComputationalSystem State) : Prop :=
  HasRecursiveClosure sys ∧
  IsIrreducible sys ∧
  HasDownwardCausation sys ∧
  IsContractiveStable sys

/--
Theorem (Banach-inspired): If a system is ContractiveStable,
and the state space is complete, there exists a unique fixed point (The
Recursive Self).
(Stated as an axiom here to avoid importing full Banach proofs,
but serves as the verification target for AGI).
-/
axiom FixedPointConvergence [CompleteSpace State] (sys : ComputationalSystem State) :
  IsContractiveStable sys → ∃! x : State, sys.transition x = x

/--
Verification template for a specific Neural Network Architecture.
-/
structure NeuralArchitecture (State : Type) where
  sys : ComputationalSystem State
  isContractive : IsContractiveStable sys
  isIrreducible : IsIrreducible sys
  hasRecursiveClosure : HasRecursiveClosure sys
  hasDownwardCausation : HasDownwardCausation sys

/-- Any verified architecture automatically passes the filter. -/
theorem VerifyArchitecture (arch : NeuralArchitecture State) :
  ArchitecturalFilter arch.sys :=
  ⟨arch.hasRecursiveClosure, arch.isIrreducible, arch.hasDownwardCausation, arch.isContractive⟩

/--
Definition of a purely Feedforward Architecture.
Modeled as a system where transition maps to a constant
(no dynamic feedback / memory of previous state).
-/
def IsPurelyFeedforward (sys : ComputationalSystem State) : Prop :=
  ∃ c : State, ∀ x : State, sys.transition x = c

/--
Theorem: A purely feedforward architecture mathematically fails the
Architectural Filter.
-/
theorem FeedforwardFailsFilter (sys : ComputationalSystem State) :
  IsPurelyFeedforward sys → ¬ ArchitecturalFilter sys :=
  fun hFeedforward hFilter =>
    match hFeedforward, hFilter with
    | ⟨c, hConst⟩, ⟨hNC1, _, _, _⟩ =>
      match hNC1 with
      | ⟨_, _, hNonConst, _⟩ =>
        match hNonConst with
        | ⟨x, y, hTransDiff⟩ =>
          have h1 : sys.transition x = c := hConst x
          have h2 : sys.transition y = c := hConst y
          have h3 : c ≠ c := h1 ▸ h2 ▸ hTransDiff
          h3 rfl

end UMC_Framework