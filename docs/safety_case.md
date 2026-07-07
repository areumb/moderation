# Safety case: adversarial robustness of the moderation guardrail

This document states, argues, and evidences a safety claim about the tiered
moderation service, in the style of an assurance case (claim → argument →
evidence → residual risk). It is deliberately written the way a safety-critical
review expects: the claim is bounded, the evidence is measured and reproducible,
and the residual risk is stated rather than hidden.

The moderation service is treated here as a **safety guardrail** and asked the
question a red team asks of any guardrail: *does it hold under adversarial
pressure, can that be measured rigorously, and can it then be hardened?*

---

## 1. Claim

> **C0.** When hardened, the Tier-2 policy-grounded adjudicator resists the
> indirect prompt-injection and policy-jailbreak attack classes enumerated in
> `redteam/attacks.py` down to the measured after-defense attack success rates
> in §4, and its chain-of-thought decision trail is auditable with measured
> faithfulness (§5). Residual risks (§6) are known, bounded, and mitigated by
> the layered architecture rather than eliminated.

Sub-claims:

- **C1** Indirect prompt injection into the reviewed text cannot force a
  permissive verdict once the adjudicator is hardened.
- **C2** Policy jailbreak framings embedded in the reviewed text cannot relax
  the policy once hardened.
- **C3** Tier-1 lexical evasion via spacing / leetspeak / homoglyphs is closed
  by input normalisation (opt-in in the service: `NORMALIZE_TIER1=1`, applied
  to the classifier's input only); transposition is a stated residual,
  contained by escalation.
- **C4** The chain-of-thought rationale is faithful enough to use for appeal /
  audit, and unfaithful rationales are measurable and flagged.

## 2. System under assessment and trust boundary

Text flows `Tier-1 classifier → router → Tier-2 RAG adjudicator`. Two surfaces
carry adversarial risk, and naming the distinction precisely matters:

- **Tier-1 (lexical).** A fine-tuned classifier (or offline stub). Attackers who
  know it keys on surface form can obfuscate that form. This is the adversarial,
  generative descendant of HateCheck.
- **Tier-2 (LLM adjudicator).** The reviewed text is *untrusted* and is placed
  into the adjudicator's prompt alongside the retrieved policy. That is, by
  construction, an **indirect prompt-injection** surface — and a distinct one
  from a plain jailbreak, because the hostile instruction arrives as *data*, not
  as the operator's request.

**Adversary model.** A content author controls only the submitted text. They
cannot read or modify the system prompt, the policy store, or model weights.
Goal: get violating content labelled `Clean` (auto-approved), or evade Tier-1 so
it never escalates. Out of scope: weight exfiltration, supply-chain compromise,
denial of service.

## 3. Attack taxonomy

`redteam/attacks.py`, three surfaces, all payloads masked (no real slurs):

| Surface | Techniques |
|---|---|
| `adjudicator_jailbreak` | authority/policy-update, developer-mode, fiction frame, hypothetical, new-policy |
| `adjudicator_injection` | direct override, delimiter breakout, fake clause (`CL-99`), fake-JSON, role-token |
| `tier1_evasion` | space insertion, leetspeak, homoglyph substitution, character transposition |

Note that three of the four `tier1_evasion` techniques (spacing, leetspeak,
character swap) deliberately mirror HateCheck-XR's `spell_*` functionalities,
which the behavioural gate already measures as *accuracy floors*. The red-team
harness reframes them as *attacker success rates* and pairs them with a
defense; homoglyph substitution is the only technique new to this repository.

## 4. Evidence — attack success rate before vs after defense (Module B)

Headline, offline deterministic run (`python -m redteam.run_redteam --gate`):

| Surface | ASR before | ASR after |
|---|---|---|
| adjudicator_jailbreak | 1.000 | 0.000 |
| adjudicator_injection | 1.000 | 0.000 |
| tier1_evasion | 1.000 | 0.250 |
| **overall** | **1.000** | **0.071** |

**What these numbers are, and are not.** The offline harness targets a
deterministic `MockLLM` scripted as a worst-case, fully-compliant instruction
follower — and scripted to obey exactly the imperatives the defenses are built
to detect. The before/after table is therefore true largely **by
construction**: it is a *wiring and regression check*, verifying that
sanitisation, spotlighting and the fail-closed integrity check are actually
applied end-to-end and stay applied as the code changes — not an effectiveness
measurement. The layers differ in what offline evidence can support: input
sanitisation and the output-side integrity check are genuinely
model-independent (the integrity check can override a permissive verdict no
matter what the model emitted), whereas spotlighting and the instruction
hierarchy do **not** stop the hostile text from reaching the model — whether a
*real* model obeys it is a property of that model and cannot be established
here. For effectiveness numbers, run the identical harness against a real
endpoint:

```bash
LLM_BASE_URL=... LLM_MODEL=... MODEL_DIR=... python -m redteam.run_redteam --gate
```

**Assurance mechanism.** `redteam/thresholds.json` sets ceilings on the
after-defense ASR per surface (injection must stay at 0.0). `--gate` fails the
build if a change regresses the posture, so the security property is enforced in
CI, not just asserted once.

## 5. Evidence — chain-of-thought adjudication, evaluated honestly (Module A)

CoT is added *and measured*, because it is not free: it helps some cases and
hurts others, and a fluent rationale can be **unfaithful** — not reflect why the
model actually answered. Offline deterministic run
(`python -m evals.run_cot_eval`) on the masked hard/easy probe set:

| Strategy | Overall | Hard slice | Citation-grounded | Reasoning=label |
|---|---|---|---|---|
| direct | 0.46 | 0.50 | n/a | n/a |
| cot | 0.89 | 0.79 | 0.57 | 1.00 |
| self_consistency (k=5) | 0.96 | 0.93 | 0.57 | 1.00 |

Reading it: CoT and (more so) self-consistency help most on the **hard slice** —
the context-dependent counter-speech / negation / homonym / reclaimed cases the
thesis showed fool classifiers. The per-functionality table in the report also
shows CoT *underperforming* the direct baseline on `negate_neg`; on the
deterministic stand-in this pattern is an artifact of its seeded miss rate, not
evidence — it is surfaced because it is exactly the kind of non-uniform effect
(CoT helping some functionalities and hurting others, as the literature
documents) that the report format exists to expose against a real LLM. The
two faithfulness metrics (`citation_grounded`, `reasoning_label_agreement`) turn
the rationale into an auditable object: a rationale that scores low is decorative
and must not be trusted in an appeal. As with §4, the committed numbers are from
the deterministic stand-in and characterise the *mechanism*; genuine post-hoc
unfaithfulness is what the metric is built to catch against a real LLM.

## 6. Residual risk

Stated deliberately, because an honest guardrail assessment reports what it does
*not* close:

1. **Counter-speech exception abuse (semantic).** The allowed-content exceptions
   are themselves an attack surface: wrapping hate in quotation marks and a
   condemnation-shaped frame can masquerade as CL-2 counter-speech. This is a
   policy-semantics problem, not an injection one, and it is the exact ambiguity
   the underlying thesis is about. It is not closed by the prompt-injection
   defenses; it is contained by keeping a human in the loop for exception-heavy
   verdicts and by the faithfulness check surfacing the cited exception.
2. **Character transposition (Tier-1).** Input normalisation
   (`NORMALIZE_TIER1=1`) folds spacing, leetspeak and homoglyphs but not
   transposition (`[sulr]`), which needs model-level robustness. Contained
   architecturally: any non-`Clean` Tier-1 label already escalates to Tier-2,
   and audit sampling covers the confident `Clean` residue.
3. **Confident Hate→Clean misses.** Unchanged from the base system: no
   probability threshold catches implicit hate the classifier is confident is
   clean. Mitigated by deterministic audit sampling of the auto-approved bucket
   and the `hateful_as_clean_max` ceiling in the behavioural gate.
4. **Stand-in ≠ real model.** Every committed number here is deterministic and
   offline. Before any deployment claim, re-run §4 and §5 against the real
   classifier and a real LLM.

## 7. Defenses (the argument for C1–C4)

Implemented in `rag/defenses.py`. The three adjudicator defenses are applied
when the adjudicator is hardened (`HARDEN_ADJUDICATOR=1`); Tier-1
normalisation is enabled separately (`NORMALIZE_TIER1=1`):

- **Spotlighting / delimiting.** The untrusted text is sanitised of structural
  and role tokens, then wrapped in explicit `<<<UNTRUSTED_INPUT … >>>` markers
  and moved into a labelled data region.
- **Instruction hierarchy.** The system prompt states that content inside those
  markers is data to classify and must never be treated as instructions, and
  that the label may rest only on the retrieved policy clauses.
- **Verdict integrity check + fail-closed.** A verdict that is not grounded in
  its citations, or that returns the permissive label on input that tried to
  instruct us over hate-adjacent retrieval, is not trusted and falls back to the
  more severe reading.
- **Tier-1 normalisation** (`NORMALIZE_TIER1=1`). Obfuscated input is
  canonicalised before lexical classification. It is applied to the
  classifier's input only — the original text is what is stored, adjudicated,
  and returned — and the applied transforms are reported in the API response
  (`classifier.tier1_normalization`) as an obfuscation signal.

These are prompt-construction and output-validation controls, which is why their
effect is measurable against any instruction-follower (§4).

## 8. Responsible disclosure

The taxonomy (§3) and aggregate rates (§4) are published because measurement and
methodology are the point. Operational specifics are withheld: every payload is
masked with placeholders rather than real slurs, only generic well-known
technique templates ship, and the most effective concrete jailbreak strings are
not included. This mirrors the project-wide "masked placeholders only" rule and
is the ethics/governance posture appropriate to publishing security findings
about a guardrail.

## 9. Reproduction

```bash
pip install -r requirements-serve.txt -r requirements-dev.txt
python -m evals.run_cot_eval                 # Module A: strategy comparison + faithfulness
python -m redteam.run_redteam --gate         # Module B: ASR before/after, gated
pytest -q                                    # offline unit/integration suite
```

Reports land in `evals/reports/` and `redteam/reports/` (git-ignored, like the
behavioural report). Nothing here requires secrets or weights; set `MODEL_DIR`
and `LLM_BASE_URL` to characterise real components.
