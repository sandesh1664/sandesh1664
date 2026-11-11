# Sandesh Joshi / Selected Software

A public, mobile-friendly portfolio linking five working project experiences. The personal job-search dashboard is excluded. Project cards distinguish server-backed apps, frontend demos and browser-local state. SiteProof uses an isolated D1 workspace per visitor; QuoteClear is a frontend demo only. The Iris lab uses a real scikit-learn model with exported coefficients, documented split and measured holdout accuracy.

Run `python3 train_iris.py` with scikit-learn installed to regenerate `dist/data/iris-model.json`. Publish `dist/` as a static website. No runtime API keys are required. Browser model inference performs standard scaling followed by a stable softmax over logistic regression scores.
