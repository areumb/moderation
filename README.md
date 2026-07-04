# Tiered, Policy-Grounded Hate Speech Moderation Service

[![CI](https://github.com/areumb/moderation/actions/workflows/ci.yml/badge.svg)](https://github.com/areumb/moderation/actions/workflows/ci.yml)

A production-style content-moderation service built on top of my master's
thesis research: a fast fine-tuned RoBERTa classifier triages all traffic,
and a RAG-grounded LLM adjudicator — which decides against retrieved clauses
of a moderation policy, not its own priors — is spent only on the cases the
research showed classifiers get wrong. Runs fully offline out of the box
(stub classifier + mock LLM), with the real model and any OpenAI-compatible
LLM activated by environment variables alone.

Built on the code and findings of
[my master's thesis](https://github.com/areumb/hatespeech-offensive) (see
[Research foundation](#research-foundation-the-thesis) below); the research
code ships unchanged in `hs_generalization/`.

## Concept

The thesis found that *where* a hate-speech classifier struggles is
dataset-dependent, not a fixed property of the task. In-distribution
(Davidson: ~6% Hateful, ~77% Offensive) the dominant confusion is
Hate → Offensive, while Clean separates easily. Out of distribution
(HateCheck-XR, a diagnostic suite full of counterspeech, quotation, negation,
and reclaimed/homonym slurs) the hard boundary shifts to Hate ↔ Clean, in
both directions: hateful content misread as Clean, and benign mentions
misread as Hateful. The service turns both regimes into an architecture — a
cheap classifier triages all traffic, and an expensive RAG-grounded LLM
adjudicator is spent exactly where the research says the classifier is
unreliable:

- the Offensive/Hateful margin trigger covers the in-distribution confusion;
- escalating every non-Clean label routes the Clean → Hate false positives to
  Tier 2, where the adjudicator can overturn them against the allowed-content
  clauses (counterspeech CL-2, negation CL-4, homonyms CL-5);
- audit sampling plus the eval ceiling cover confident Hate → Clean misses,
  which no probability threshold can catch.

```
                        ┌────────────────────┐
        text ──────────►│ Tier 1: RoBERTa    │  fine-tuned ternary classifier
                        │ (or offline stub)  │  label + per-class probs + confidence
                        └─────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │ Router             │  escalate if:
                        │ (configurable      │   • top label Offensive/Hateful
                        │  thresholds)       │   • confidence < threshold
                        └───┬──────────┬─────┘   • |P(Off) − P(Hate)| < margin
                     auto   │          │  escalated / audit
                            ▼          ▼
                 final label       ┌───────────────────────────────┐
                 (Tier 1 as-is)    │ Tier 2: RAG adjudicator       │
                                   │  Chroma vector store over     │
                                   │  policies/*.md  ──► retrieved │
                                   │  clauses ──► LLM (or mock)    │
                                   │  ──► final label + cited      │
                                   │      clause ids + rationale   │
                                   └───────────────────────────────┘
```

The retrieval is load-bearing: the adjudicator decides violations against the
retrieved policy clauses (and can only cite clauses it was shown), so swapping
`policies/community_guidelines.md` changes the system's decisions with no
retraining. The bundled policy is synthetic and clearly marked illustrative.

Two research-motivated details in the router (`serving/router.py`):

- **Audit sampling.** The thesis' HateCheck-XR results show the dominant
  out-of-distribution error is hateful content *confidently* misclassified as
  Clean — invisible to every probability-based trigger. So a deterministic
  sample of the auto-approved bucket (SHA-256 of the text, `audit_rate`,
  default 2%) is sent to Tier 2 anyway, with `route: "audit"` in the response.
- **No Clean↔Hateful margin rule, on purpose.** For any confidence threshold
  ≥ 0.575 a Clean/Hateful near-tie is already below the confidence gate
  (if P(Clean) ≥ τ then the margin is ≥ 2τ−1), so such a rule could never fire
  — the residual risk is exactly the confident misses that audit sampling and
  the eval gate below address. The derivation is in the router docstring.

`GET /stats` reports the observed routing distribution (route counts, trigger
counts, Tier-2 rate), because the same thresholds produce very different
Tier-2 loads on different traffic mixes — Davidson is ~77% Offensive, while
production traffic is mostly Clean. Measure, don't assume.

The API also exposes the thesis' binary label spaces as optional views
(`mode: hate_nonhate | nonclean_clean`), using the same probability-merging
rule as `hs_generalization/modes.py`.

## Quickstart (offline, no weights, no secrets)

```bash
pip install -r requirements-serve.txt   # research requirements.txt unchanged
uvicorn serving.app:app --reload
# then:
curl -X POST localhost:8000/moderate -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed the community picnic today."}'
```

Endpoints: `POST /moderate`, `GET /health`, `GET /stats`.

Without configuration the service uses a deterministic **StubClassifier** and a
**MockLLM**, so the full request path (classify → route → retrieve → adjudicate)
runs on any machine. Activate the real components with environment variables
only:

| Variable | Effect |
|---|---|
| `MODEL_DIR` | Dir containing a thesis `.pt` checkpoint (e.g. `outputs/davidson/RoBERTa-base/3class`) or an HF `save_pretrained` dir |
| `MODEL_NAME` | Base model for `.pt` checkpoints (default `roberta-base`) |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | Any OpenAI-compatible endpoint; for Ollama use `http://localhost:11434/v1` (no key) |
| `SERVING_CONFIG` | Alternative router/RAG config (default `serving/config.json`) |
| `CONF_THRESHOLD`, `MARGIN_THRESHOLD`, `AUDIT_RATE` | Router overrides: confidence floor, Offensive/Hateful margin, audit-sample rate of the auto bucket |

Docker: `docker build -f docker/Dockerfile -t moderation-service . && docker run -p 8000:8000 moderation-service`
— Azure Container Apps steps in [`docs/deploy_azure.md`](docs/deploy_azure.md).

## Tests, CI, and the behavioral gate

`pytest` runs fully offline. CI (GitHub Actions) lints, tests, and runs the
HateCheck-XR behavioral suite in smoke mode. With a real model
(`MODEL_DIR` set), `python -m evals.run_behavioral_suite` enforces
per-functionality accuracy floors from `evals/thresholds.json` and fails the
build on regressions — the thesis' challenge set acting as an automated
model-quality gate. The suite also reports the full ternary confusion matrix
and gates both directional Hate↔Clean error rates as ceilings:
**gold-Hateful→Clean** (`hateful_as_clean_max`, the dominant OOD error mode —
under-moderation) and **gold-Clean→Hateful** (`clean_as_hateful_max`,
over-moderation of counterspeech/quotation/negation/homonyms). Where to set
each ceiling is the explicit recall-vs-precision decision the thesis argues a
deployment must make according to its moderation goals — the same dial the
router thresholds (`CONF_THRESHOLD`, `AUDIT_RATE`) expose at serving time.
Threshold values are conservative placeholders: calibrate them against your
own measured results before relying on them.

## Experiment tracking

`python -m tracking.mlflow_eval -c <test-config> --seed 5 --checkpoint <ckpt> --variant ternary`
runs the unmodified `hs_generalization.test` and logs its params and metrics
(per-class F1 from the saved confusion matrix, macro/micro metrics) to a local
`./mlruns` store; inspect with `mlflow ui`.

## Research foundation (the thesis)

This project extends my master's thesis, *"Modeling Offensive Language as a
Distinct Class for Hate Speech Detection"*
([Kim, 2025 — PDF in the thesis repository](https://github.com/areumb/hatespeech-offensive/blob/main/Thesis_Areum.pdf)),
supervised by Dr. Antske Fokkens and Dr. Hennie van der Vliet — see the
[original thesis repository](https://github.com/areumb/hatespeech-offensive) for the research-focused
version of this code. The thesis categorized language into three mutually
exclusive classes (Hateful, Offensive, Clean) and evaluated a RoBERTa-base
model in the full ternary setup and in binary merges (Hate vs. Non-hate,
Non-clean vs. Clean, Hate vs. Clean), finding that explicitly modeling
offensive language clarifies rather than complicates hate speech detection.
The research code includes my modifications and extensions of
[Khurana et al. (2025)](https://arxiv.org/abs/2410.15911)'s
[code](https://github.com/urjakh/defverify).

**HateCheck-XR**, developed for the thesis, ships in this repository
(`datasets/hatecheck-xr/hatecheck-xr.csv`): a re-annotation of
**HateCheck** [(Röttger et al., 2021)](https://aclanthology.org/2021.acl-long.4/)
and of an existing extension by
[Khurana et al. (2025)](https://arxiv.org/abs/2410.15911), aligned with the
ternary scheme, with functionality names adjusted and annotation errors
corrected. Here it doubles as the CI model-quality gate (see above).

### Running the research code

The thesis training/evaluation pipeline is unchanged and lives in
`hs_generalization/` (its dependencies in `requirements.txt`, kept separate
from the service's `requirements-serve.txt`):

```bash
pip install -r requirements.txt

# Training (configs follow configs/example.json)
python -m hs_generalization.train -c configs/example.json

# Evaluation of a single checkpoint
python -m hs_generalization.test -c <test-config> --dataset davidson \
  --eval-mode 3class --train-mode 3class --seed 5 \
  --checkpoint "outputs/davidson/RoBERTa-base/3class/<checkpoint>.pt"

# Multiple seeds/checkpoints at once
python hs_generalization/run_many.py -c <test-config> --dataset hatecheck_xr \
  --eval-mode 3class --train-mode 3class --seeds 7 222 550 999 3111 \
  --ckpt-pattern "outputs/davidson/RoBERTa-base/3class/*.pt" \
  --hatecheck-csv datasets/hatecheck-xr/hatecheck-xr.csv
```

Trained checkpoints are not committed (they are ~1.5 GB each); point
`MODEL_DIR` at your own checkpoint directory to activate the real classifier.

## Repository layout

```
├─ hs_generalization/     thesis research code (UNCHANGED): train, test, modes, utils, run_many
├─ configs/               research config files (example.json; train/ val/ test/)
├─ datasets/
│  ├─ davidson/           Davidson et al. (2017) splits used in the thesis
│  └─ hatecheck-xr/       HateCheck-XR (my re-annotated ternary challenge set)
├─ scripts/               dataset utilities
├─ serving/               FastAPI app, router (+ audit sampling), stats, classifier wrapper
├─ rag/                   embeddings, policy store, retriever, adjudicator, LLM abstraction
├─ policies/              synthetic community guidelines (clause ids HL-*/OF-*/CL-*)
├─ evals/                 HateCheck-XR behavioral gate + thresholds
├─ tracking/              MLflow wrapper around the unmodified evaluator
├─ tests/                 offline pytest suite (stub classifier + mock LLM)
├─ docker/                Dockerfile (+ compose reference)
├─ docs/                  Azure Container Apps deployment steps
└─ .github/workflows/     CI: ruff + pytest + behavioral suite (smoke)
```

## Limitations

- **Auto-approved blind spot (mitigated, not solved).** Confidence-based
  routing cannot escalate implicit/coded hate that the classifier confidently
  mislabels as Clean. Two mitigations are implemented: deterministic
  audit-sampling of the auto-approved bucket (`audit_rate`, default 2%), and
  an explicit gold-Hateful→Clean ceiling in the behavioral gate. Sampling
  reduces exposure and measures the miss rate on the sample; the unsampled
  remainder still exits at Tier 1 unreviewed.
- The bundled policy is synthetic and illustrative; decisions grounded in it
  demonstrate the mechanism, not any real platform's rules.
- The stub classifier and mock LLM exist so the pipeline runs offline; their
  outputs are deterministic placeholders, not predictions.
- Retrieval quality depends on the embedding model; the offline fallback
  (hashed bag-of-words) is intentionally simple and weaker than the default
  sentence-transformers embeddings.
- No quality numbers are claimed in this repository beyond what the thesis
  measured; the eval thresholds shipped here are gates, not results.

## Attribution

Research code lineage: [Khurana et al. (2025)](https://arxiv.org/abs/2410.15911)
([defverify](https://github.com/urjakh/defverify)). Diagnostic suite:
[HateCheck (Röttger et al., 2021)](https://aclanthology.org/2021.acl-long.4/).
Dataset: [Davidson et al. (2017)](https://github.com/t-davidson/hate-speech-and-offensive-language).
If you use HateCheck-XR, please cite the thesis (Kim, 2025).

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
The research code in `hs_generalization/` includes modifications of
[Khurana et al. (2025)](https://arxiv.org/abs/2410.15911)'s
[defverify](https://github.com/urjakh/defverify), which itself adapts
Apache-2.0-licensed [HuggingFace Transformers example code](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py);
the original copyright and license notices are retained in the affected file
headers (see `hs_generalization/train.py`).
