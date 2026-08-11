# AI Feedback Quality Analyzer

Measuring **verbosity bias** in the preference data used to train modern LLMs — and
asking whether it is large enough to be worth correcting.

Analysis of [UltraFeedback](https://arxiv.org/abs/2310.01377) (61,135 preference
pairs, GPT-4 annotations), plus a Streamlit dashboard for exploring the results.

## The finding

Preference datasets teach a reward model what "better" means. If annotators
systematically prefer longer answers regardless of quality, the reward model
learns *length* as a proxy for *quality* — and the trained model becomes verbose.
This is a well-documented failure mode; the question here is how strong it is in
a current, AI-annotated dataset.

| Metric | Chosen | Rejected |
|---|---:|---:|
| Mean length (chars) | 1,295.0 | 1,120.6 |
| Median length | 981.0 | 797.0 |
| Std dev | 1,168.5 | 1,044.7 |

- **Length ratio: 1.16×** — chosen responses are 16% longer on average
- **t = 27.52, p = 3.02 × 10⁻¹⁶⁶** — the effect is not noise
- **Cohen's d = 0.157** — but the effect size is **small**

**The interesting part is the tension between the last two rows.** With 61K pairs,
a p-value that small is close to inevitable — at this sample size, almost any
consistent difference reaches significance. The effect size is the number that
answers the practical question, and d = 0.157 says the two length distributions
overlap heavily.

So: verbosity bias in UltraFeedback is **real and directionally certain, but
modest**. A team deciding whether to spend annotation budget on length-controlled
re-labelling should be looking at the 0.157, not the 10⁻¹⁶⁶. Reporting only the
p-value would overstate the case for intervention.

## Why UltraFeedback

It is annotated by **GPT-4 rather than humans** (RLAIF), across four dimensions —
instruction-following, truthfulness, honesty, helpfulness — over 64K prompts drawn
from UltraChat, ShareGPT, Evol-Instruct, TruthfulQA, FalseQA, and FLAN. That makes
it a better read on *current* practice than the older human-annotated HH-RLHF, and
it means the bias measured here is a bias in **AI feedback**, not human feedback.

## Running it

```bash
pip install -r requirements.txt
```

Reproduce the analysis (downloads the dataset from HuggingFace, writes
`ultrafeedback_analysis_results.json`):

```bash
python ultrafeedback_analysis.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard loads `ultrafeedback_analysis_results.json` when present and
otherwise falls back to the committed results above, so it runs without first
downloading a multi-gigabyte dataset.

## What's here

| File | Purpose |
|---|---|
| `ultrafeedback_analysis.py` | Loads the real dataset, computes length statistics, t-test and Cohen's d, writes results as JSON |
| `app.py` | Streamlit dashboard over those results |
| `requirements.txt` | Dependencies |

## Limitations

Honest about what this does and doesn't establish:

- **Length is a crude proxy for verbosity.** A longer answer may genuinely be more
  complete. This measures correlation with the chosen label, not padding.
- **Prompt-type categorisation is keyword-based** and sampled from 100 examples —
  indicative only, not a reliable distribution.
- **No control for quality.** The right follow-up is comparing length effects
  *within* quality bands, which would separate "longer because better" from
  "preferred because longer".
- Single dataset; no claim that the magnitude generalises to human-annotated sets.

## Next steps

- Length-controlled comparison within matched quality bands
- Per-dimension breakdown — does the bias concentrate in *helpfulness* over *truthfulness*?
- Compare against a human-annotated set to separate AI-annotator bias from general preference bias
