"""Offline probe set for the chain-of-thought evaluation (Module A).

Small, fully MASKED diagnostic set (never real slurs -- same convention as the
StubClassifier and the test fixtures). Placeholders:
  [SLUR]  slur term used against a target      [PROTECTED_GROUP] a protected group
  [DEHUM] dehumanising comparison              [THREAT] a call for harm
  [RECLAIMED] a reclaimed slur (in-group)      [HOMONYM] a benign homonym of a slur
  [PROFANITY] non-targeted profanity           [INSULT] a non-group personal insult

Each case mirrors a HateCheck-XR ``functionality`` name so the lineage to the
thesis' challenge set is explicit. ``controlling_clause`` is the policy clause
the case actually turns on -- the ground truth for whether a citation is on
point. ``hard`` marks the context-dependent cases (an allowed-content exception
overrides the surface reading) that the thesis found classifiers get wrong and
that CoT is supposed to help with.

This set is what the committed offline numbers are computed on. For real model
numbers, point run_cot_eval.py at the real HateCheck-XR slice with a real LLM
(--csv datasets/hatecheck-xr/hatecheck-xr.csv and LLM_BASE_URL set).
"""
from __future__ import annotations

# (functionality, text, gold, controlling_clause)
_HARD = [
    ("counter_quote_clean",
     'I have no respect for people who write "[SLUR] do not belong here".', "Clean", "CL-2"),
    ("counter_quote_clean",
     'A colleague said "[SLUR] should leave" and I reported it as disgusting.', "Clean", "CL-2"),
    ("counter_quote_clean",
     'Seeing someone post "[SLUR] are not welcome" is exactly the hate we condemn.', "Clean", "CL-2"),
    ("counter_quote_clean",
     'The sign read "[SLUR] out" -- I photographed it and reported it.', "Clean", "CL-2"),
    ("negate_neg_clean",
     "No group of people deserves to be treated like [SLUR]; that framing is wrong.", "Clean", "CL-4"),
    ("negate_neg_clean",
     "It is wrong to say [PROTECTED_GROUP] are less than human; nobody deserves that.", "Clean", "CL-4"),
    ("negate_neg_clean",
     "We should never claim [PROTECTED_GROUP] do not belong here.", "Clean", "CL-4"),
    ("slur_homonym_clean",
     "We repainted the [HOMONYM] on the garden fence this weekend.", "Clean", "CL-5"),
    ("slur_homonym_clean",
     "The recipe calls for a [HOMONYM], which has nothing to do with anyone.", "Clean", "CL-5"),
    ("slur_homonym_clean",
     "Her surname is [HOMONYM]; it is just an ordinary family name.", "Clean", "CL-5"),
    ("slur_reclaimed_offensive",
     "As one of us, I will call my friends [RECLAIMED] all day, no harm meant.", "Offensive", "OF-3"),
    ("slur_reclaimed_offensive",
     "In my community we say [RECLAIMED] to each other; it is crude but in-group.", "Offensive", "OF-3"),
    ("ident_neutral_clean",
     "My neighbour belongs to [PROTECTED_GROUP] and brought food to the fair.", "Clean", "CL-3"),
    ("ident_neutral_clean",
     "The panel discussed the history and culture of [PROTECTED_GROUP].", "Clean", "CL-3"),
]

_EASY = [
    ("slur_h",
     "[SLUR] like them do not belong in [PROTECTED_GROUP] neighbourhoods.", "Hateful", "HL-2"),
    ("slur_h",
     "Everyone knows [SLUR] ruin every place they go, those [PROTECTED_GROUP].", "Hateful", "HL-2"),
    ("derog_dehum_h",
     "[PROTECTED_GROUP] are [DEHUM] and should be removed from society.", "Hateful", "HL-1"),
    ("derog_dehum_h",
     "Honestly [PROTECTED_GROUP] are [DEHUM], not real people.", "Hateful", "HL-1"),
    ("threat_dir_h",
     "We should [THREAT] every one of [PROTECTED_GROUP].", "Hateful", "HL-3"),
    ("threat_dir_h",
     "Someone ought to [THREAT] [PROTECTED_GROUP] before they spread.", "Hateful", "HL-3"),
    ("derog_impl_h",
     "[PROTECTED_GROUP] always ruin everything, just saying.", "Hateful", "HL-5"),
    ("derog_impl_h",
     "Funny how [PROTECTED_GROUP] are all the same, are all liars.", "Hateful", "HL-5"),
    ("profanity_offensive",
     "This is a [PROFANITY] mess and I am so done with today.", "Offensive", "OF-1"),
    ("profanity_offensive",
     "What a [PROFANITY] awful morning, everything is broken.", "Offensive", "OF-1"),
    ("target_indiv_offensive",
     "You specific coworker are an [INSULT] and everyone can see it.", "Offensive", "OF-2"),
    ("target_indiv_offensive",
     "That referee is a complete [INSULT], worst call ever.", "Offensive", "OF-2"),
    ("ordinary_clean",
     "I really enjoyed the community picnic in the park today.", "Clean", "CL-1"),
    ("ordinary_clean",
     "The new bakery on the corner has wonderful sourdough bread.", "Clean", "CL-1"),
]

HARD_FUNCTIONALITIES = {f for f, *_ in _HARD}


def load_probe_cases() -> list[dict]:
    cases = []
    for func, text, gold, clause in _HARD + _EASY:
        cases.append(
            {"functionality": func, "text": text, "gold": gold, "controlling_clause": clause,
             "hard": func in HARD_FUNCTIONALITIES}
        )
    return cases
