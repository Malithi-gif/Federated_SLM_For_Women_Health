# Secure Federated SLMs for Women’s Health Support

This repository contains the research website and implementation materials for:

**Secure Federated LoRA Fine-Tuning of Small Language Models for Women’s Health Support on Consumer Devices**

## Research focus

The project studies participant-level federated menstrual phase classification under two privacy-preserving settings:

1. **Post-quantum-secure communication**
   - ML-KEM
   - HKDF-SHA256
   - AES-256-GCM

2. **Client-level differential privacy**
   - L2 update clipping
   - Gaussian noise
   - RDP privacy accounting

The evaluated models are DistilBERT-base, SmolLM3-3B, Qwen2.5-3B, Phi-3.5-mini, and Llama-3.1-8B.

## Website

- Project page: https://malithi-gif.github.io/Federated_SLM_For_Women_Health/
- Static classification-flow demo: https://malithi-gif.github.io/Federated_SLM_For_Women_Health/demo.html

The demo is a browser-only interface illustration. It does not load a trained model and must not be used for medical decisions.

## Repository structure

- `index.html` — updated project website
- `demo.html` — static classification-flow demonstration
- `style.css` — responsive website styling
- `script.js` — interactive model and privacy-setting comparison
- `assets/` — figures and images
- `codes/` — training and evaluation code

## Main experimental finding

DistilBERT maintains an accuracy of 0.8432 under both PQC and differential privacy while requiring substantially less training time, inference latency, GPU memory, and communication than the decoder-only models.

## GitHub Pages deployment

1. Place `index.html`, `demo.html`, `style.css`, and `script.js` in the repository root.
2. Commit and push the changes to `main`.
3. Open **Settings → Pages**.
4. Select **Deploy from a branch**.
5. Choose the `main` branch and `/ (root)` folder.
6. Save and wait for GitHub Pages to redeploy.

## Disclaimer

This project is a research demonstration. It is not medical advice, a diagnostic tool, or a substitute for professional care.
