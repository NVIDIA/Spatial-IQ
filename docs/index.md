<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

---
myst:
  html_meta:
    description lang=en: "Spatial-IQ: Deconstructing Spatial Intelligence via Hierarchical Capability Tests."
---

# Spatial-IQ

```{raw} html
<style>
  /* Hide the Sphinx-generated page H1 — the paper title is rendered in the hero below. */
  article.bd-article > section > h1:first-child,
  div.document > div.documentwrapper section > h1:first-of-type {
    display: none;
  }
</style>
<section class="siq-hero">
  <h1 class="siq-title">Spatial-IQ: Deconstructing Spatial Intelligence<br/>via Hierarchical Capability Tests</h1>
  <div class="siq-subtitle">A diagnostic framework that decomposes 3D object counting into nine hierarchical spatial perception and cognition sub-tasks.</div>

  <div class="siq-authors">
    <span class="siq-author"><a href="https://patrickrim.com/">Patrick Rim</a><sup>1,2</sup></span>
    <span class="siq-author"><a href="https://www.linkedin.com/in/tom1ong">Tom Long</a><sup>1</sup></span>
    <span class="siq-author"><a href="https://research.nvidia.com/person/ekta-prashnani">Ekta Prashnani</a><sup>1</sup></span>
    <span class="siq-author"><a href="https://research.nvidia.com/person/ruth-rosenholtz">Ruth Rosenholtz</a><sup>1</sup></span>
    <span class="siq-author"><a href="https://research.nvidia.com/person/ben-boudaoud">Ben Boudaoud</a><sup>1</sup></span>
    <span class="siq-author"><a href="https://research.nvidia.com/person/peter-xenopoulos">Peter Xenopoulos</a><sup>1</sup></span>
    <span class="siq-author">Alex Wong<sup>2</sup></span>
    <span class="siq-author"><a href="https://research.nvidia.com/person/joohwan-kim">Joohwan Kim</a><sup>1</sup></span>
    <span class="siq-author"><a href="https://research.nvidia.com/person/jaehyun-jung">Jae-Hyun Jung</a><sup>1</sup></span>
  </div>
  <div class="siq-affiliations">
    <span class="siq-affil"><sup>1</sup>NVIDIA Research</span>
    <span class="siq-affil"><sup>2</sup>Yale University</span>
  </div>

  <div class="siq-buttons">
    <a class="siq-btn" href="https://arxiv.org/abs/2607.22864"><span class="siq-btn-icon">&#128196;</span>Paper (arXiv)</a>
    <a class="siq-btn siq-btn-muted" href="https://huggingface.co/datasets/patrickqrim/spatial-iq"><span class="siq-btn-icon">&#128190;</span>Dataset</a>
    <a class="siq-btn siq-btn-muted" href="https://github.com/NVIDIA/Spatial-IQ"><span class="siq-btn-icon">&#60;&#47;&#62;</span>Code</a>
  </div>
</section>
```

```{figure} _static/figures/demo.gif
:class: siq-figure-img siq-figure-demo
:align: center

**DEMO:** One scene, two reasoners. The human (top) walks through each sub-task correctly and the components compose into the total. Gemini 3 Pro (bottom) hits *visible count* but its underlying hierarchy is broken and the intermediates don't compose.
```

```{figure} _static/figures/teaser.png
:class: siq-figure-img
:align: center

**Figure 1.** Humans count stacked 3D objects by hierarchically decomposing the structure into columns, layers, and visible vs. hidden blocks (left). Current MLLMs miscount; fine-tuning on Spatial-IQ with chain-of-thought supervision over our proposed hierarchy teaches them to do the same (right).
```

## Motivation

In Piaget & Inhelder (1956) and the subsequent human developmental-psychology tradition, human spatial cognition is **not** a single monolithic capability. It develops *hierarchically*: simpler perceptual operations — object individuation, perceptual grouping, parsing scenes into columns and layers — accumulate into more complex ones (visible counting, hidden-count inference from physical support), which in turn compose into target tasks like 3D object counting. Crucially, the relation runs both ways: solving the higher target task **presupposes** mastery of its prerequisites, and mastery of the prerequisites **enables** the higher task. Headline accuracy and sub-capability accuracy are coupled in humans because they are facets of the same compositional process.

**Does the same hierarchy hold for current MLLM spatial reasoning?** Do models that succeed on *Object Counting* also succeed on the prerequisite sub-capabilities? Do strong sub-capability scores translate to strong target-task scores? Or are hierarchy adherence and target-task performance independent properties in current models — such that two models with the same headline accuracy can be doing fundamentally different things underneath?

Existing benchmarks cannot answer this. They evaluate models as black boxes, scoring only the target task. When a model fails, it is unclear whether the failure is *perceptual* (recognizing object boundaries) or *cognitive* (reasoning about occlusion to infer hidden geometry). To answer the question, we needed a benchmark that **opens the box**.

```{raw} html
<div class="siq-tldr">
  <strong>TL;DR.</strong> In human developmental psychology, spatial cognition is <strong>hierarchical</strong> &mdash; simpler perceptual sub-capabilities compose into complex ones, and the relation runs both ways. To ask whether current MLLM spatial reasoning shares this structure, we built <strong>Spatial-IQ</strong>: a hierarchical benchmark of <strong>~80,000 procedurally generated Isaac Sim scenes</strong>, decomposing 3D object counting into <strong>9 sub-tasks + 2 target tasks</strong>, evaluated across <strong>3 response modalities</strong> (free-response text, image-option MCQ, image editing) <strong>against a human baseline collected in both text and MCQ</strong>. Comparing 8 frontier text models and 3 image-editing models, we find that top MLLMs frequently hit the target task <em>without</em> preserving the underlying hierarchy &mdash; and that two models at the same headline score can have very different mechanisms. Applying the same hierarchy as a <em>training signal</em> (CoT supervision over the decomposition + RL with verifiable rewards) lifts a 32B Qwen2.5-VL from 2.9% &rarr; 62.6% on <em>Object Counting</em> <strong>while preserving the full hierarchical chain</strong>.
</div>
```

## Abstract

Multimodal large language models (MLLMs) excel at visual interpretation but fail on spatial reasoning tasks that humans solve reliably. Existing benchmarks evaluate these models as black boxes, limiting their ability to identify the underlying causes of lower performance: when a model fails a spatial reasoning task, it remains difficult to ascertain whether the hurdle is *perceptual* (recognizing object boundaries) or *cognitive* (reasoning about occlusion to infer hidden geometry).

We introduce **Spatial-IQ**, a hierarchical diagnostic framework that decomposes object counting in stacked 3D structures into 9 perceptual and cognitive sub-tasks organized by the developmental stages of human spatial cognition, with mental rotation as an additional target probe. Using **NVIDIA Isaac Sim**, we procedurally generated a diverse dataset of roughly **80,000 stacked 3D structures** with per-task ground truth. We evaluate models across three output formats — free-response text, multiple-choice images, and image editing — alongside a human baseline.

The Spatial-IQ framework shows that top-performing models often succeed at the target task (object counting) without succeeding on the lower-level sub-tasks intended to support it, and that models differ in how much of these hierarchical chains they preserve, often revealing shortcut behavior that raw target-task accuracy alone would obscure. Finally, we demonstrate that training models with chain-of-thought (CoT) supervision over our hierarchical sub-tasks, combined with reinforcement learning with verifiable rewards, significantly improves both spatial consistency across sub-tasks and target-task accuracy.

```{raw} html
<div class="siq-highlights">
  <div class="siq-card">
    <h3>MLLMs pass the target without the prerequisites</h3>
    <p>Top-performing MLLMs frequently succeed at <em>Object Counting</em> without succeeding at the sub-tasks the count ostensibly requires. The hierarchy that couples target and prerequisites in humans is largely broken in current models &mdash; the gap is <strong>hierarchical</strong>, not just numeric.</p>
  </div>
  <div class="siq-card">
    <h3>Same headline score, different mechanism</h3>
    <p>Qwen and Gemini reach comparable headline <em>Object Counting</em> accuracy, but Qwen approximates the human <em>Summation Mechanism</em> (<strong>88.9%</strong> both-correct, <strong>64.2%</strong> exact match) while Gemini reaches similar accuracy through a much weaker mechanism (<strong>48.0%</strong> / <strong>18.6%</strong>). Hierarchy adherence and target accuracy are partially independent.</p>
  </div>
  <div class="siq-card">
    <h3>Humans win by composition</h3>
    <p>Humans reach <strong>82.1%</strong> on <em>Object Counting</em>; the best MLLM reaches <strong>17.7%</strong>. Human accuracy is concentrated in the both-component-correct cell (<strong>90.6%</strong>) with a <strong>69.1%</strong> exact-match <em>Summation Mechanism</em> &mdash; the empirical signature of composition that current models miss.</p>
  </div>
  <div class="siq-card">
    <h3>The hierarchy is also a training signal</h3>
    <p>Chain-of-thought supervision over the decomposition + RL with verifiable rewards lifts a 32B Qwen2.5-VL from <strong>2.9% &rarr; 62.6%</strong> on <em>Object Counting</em> <strong>while preserving the full hierarchical chain</strong>.</p>
  </div>
</div>
```

## The Spatial-IQ Framework

Spatial-IQ treats spatial intelligence as a hierarchy of dependencies. Following the classical human developmental psychology of spatial cognition (**Piaget & Inhelder, 1956**), we view spatial competence as developing through a progression of more elementary sub-capabilities that compose into higher target tasks: *column, layer, and cluster understanding* for structural reasoning, and *support-object and column identification* for the gravitational reasoning needed to infer hidden supporting blocks. Successfully solving the higher target task therefore *presupposes* mastery of these lower-level capabilities — in humans, headline performance and sub-capability performance are coupled because they are facets of the same compositional process.

Within this framework, we adopt a target task drawn from standardized cognitive assessment: the **Block Counting subtest of the Kaufman Assessment Battery for Children (KABC)**, in which the responder reports the total number of objects in a stacked 3D structure, including those occluded but logically required for physical support. The Spatial-IQ hierarchy is the explicit decomposition this target task recruits — visible counting, hidden inference, and the support reasoning that links them.

Spatial-IQ therefore evaluates models on both the target task and its sub-capabilities, in order to test (i) whether models exhibit a human-like hierarchical structure in spatial understanding, and (ii) whether such hierarchical competence translates to better performance on the target task.


```{figure} _static/figures/tasks.png
:class: siq-figure-img
:align: center

**Figure 2.** [Left] The nine sub-tasks (S1&ndash;S9) and two target tasks (T1, T2) of Spatial-IQ. S1&ndash;S4 probe primitive perceptual grouping (object individuation, clustering, columns, layers). S5 counts visible objects. S6&ndash;S8 test structural reasoning (top layer, direct support, support columns). S9 counts hidden objects. T1 is the total *Object Count*; T2 is *Mental Rotation*. [Right] An example scene with per-task ground-truth masks for the image-editing format.
```

```{figure} _static/figures/chains.png
:class: siq-figure-img
:align: center

**Figure 3.** We pre-specify four sub-task relations for hierarchy-dependency analysis: the **Internal Referential Chain** (top layer &rarr; direct support &rarr; support column), the **Visible Support Hierarchy**, the **Hidden Support Hierarchy**, and the **Summation Mechanism** (visible count + hidden count &rarr; total).
```

**Dataset.** We procedurally generate scenes in NVIDIA Isaac Sim 5.1, each consisting of a foreground structure of multiple identical objects (textured cubes, factory boxes, soup cans, or potted meat) constrained to a 4×4×4 voxel grid. Hidden objects are identified through a pixel-level depth-buffer occlusion test that retains an object only if it is camera-visible *or* required to support a retained object above it — guaranteeing every hidden object in the scene is *logically required by physical support* and never merely concealed behind a visible neighbor at an unrecoverable depth. The full dataset comprises ~80,000 scenes; our primary benchmark uses a 3,000-sample evaluation set, with a disjoint 68,000-sample set reserved for training.

## Validating the Benchmark

We evaluate frontier proprietary text-output models (Gemini 3 Pro, GPT 5.4, Claude Opus 4.6), open-weight models (Qwen3.5-27B, Kimi K2.5, GLM 4.6), the small-model anchors Qwen3.5-3B and VLA-0, and three image-editing models (Gemini 3.1 Flash Image, Qwen-Image-Edit, HunyuanImage-Instruct).

```{figure} _static/figures/fig_c1.png
:class: siq-figure-img
:align: center

**Figure 4.** Per-task accuracy in the text free-response modality, ordered according to our task taxonomy. Humans (black) perform well above all current models on *Object Counting*, while model performance varies by an order of magnitude. Models are shown in order of *Object Counting* performance.
```

**The target task is solvable by humans and meaningfully ranks models.** Humans reach **82.1%** on *Object Counting*, demonstrating that the task is solvable yet not at ceiling. The performance spread distinguishes humans from models without saturating: humans sit well above the best model (Qwen, **17.7%**), which in turn outperforms the worst model by an order of magnitude (Qwen2.5-3B, 2.1%). All consecutive differences between adjacent models are statistically significant under paired McNemar tests with multiple-comparisons correction.

```{figure} _static/figures/wrong_answer_mcp.png
:class: siq-figure-img
:align: center

**Figure 5.** Wrong-answer preferences in the five-choice MCQ condition. Each column denotes a wrong-answer type; each cell reports the raw percentage of responses assigned to that type. Only cells significant under a one-sided binomial test against the 20% chance baseline are shown. Models exhibit task-dependent wrong-answer preferences — *phantom block above top*, *merge two columns*, *drop visible blocks* — whereas humans show no significant preference.
```

```{figure} _static/figures/fig_c2.png
:class: siq-figure-img
:align: center

**Figure 6.** *Object Counting* accuracy as a function of five difficulty controls in the text modality. Total objects, hidden objects, layers, columns, and fill ratio all affect difficulty in the expected direction (more is harder, except fill ratio, where denser is easier), for both humans and models.
```

## Decomposing Model Performance

Three columns of the table below report raw task accuracies (*Obj. Count*, *Visible*, *Hidden*). The remaining four columns capture pre-specified hierarchy relations: the **Internal Referential Chain** asks whether models preserve object reference along *Top Layer* &rarr; *Direct Support* &rarr; *Support Column*; the **Visible** and **Hidden Support Hierarchies** ask whether models preserve higher-level support capability within each branch; the **Summation Mechanism** asks whether the model integrates correct visible and hidden counts into a correct total. The latter four columns are *conditional* accuracies of the lower-level task given correctness on the designated higher-level tasks.

### Text modality

```{raw} html
<div class="siq-table-caption"><span class="siq-fig-label">Table 1.</span> Text modality evaluative summary (% accuracy). Per-column model rankings (humans excluded): <span style="background:#8cb4ff;padding:0 4px;">first</span>, <span style="background:#bed7ff;padding:0 4px;">second</span>, <span style="background:#e6f0ff;padding:0 4px;">third</span>.</div>
<div class="siq-table-wrap">
<table class="siq-table">
  <thead>
    <tr>
      <th>Responder</th>
      <th>Obj. Count</th>
      <th>Int. Ref. Chain</th>
      <th>Visible</th>
      <th>Vis. Support</th>
      <th>Hidden</th>
      <th>Hid. Support</th>
      <th>Sum. Mech.</th>
    </tr>
  </thead>
  <tbody>
    <tr class="siq-human"><td>Human</td><td>82.1</td><td>90.1</td><td>76.8</td><td>84.5</td><td>79.9</td><td>78.5</td><td>69.1</td></tr>
    <tr class="siq-divider"><td>Qwen</td><td class="siq-rank-1">17.7</td><td>50.7</td><td class="siq-rank-1">19.2</td><td class="siq-rank-1">78.2</td><td class="siq-rank-2">49.2</td><td class="siq-rank-1">47.9</td><td class="siq-rank-1">64.2</td></tr>
    <tr><td>Gemini</td><td class="siq-rank-2">14.8</td><td class="siq-rank-1">64.5</td><td class="siq-rank-2">17.7</td><td class="siq-rank-2">56.7</td><td>43.6</td><td class="siq-rank-2">34.6</td><td>18.6</td></tr>
    <tr><td>Claude</td><td class="siq-rank-3">14.1</td><td class="siq-rank-3">52.4</td><td class="siq-rank-3">15.7</td><td>48.5</td><td class="siq-rank-1">51.4</td><td class="siq-rank-3">32.5</td><td class="siq-rank-2">43.7</td></tr>
    <tr><td>Kimi</td><td>13.6</td><td>47.2</td><td>11.5</td><td>52.9</td><td>33.1</td><td>24.5</td><td>25.0</td></tr>
    <tr><td>GPT</td><td>8.3</td><td class="siq-rank-2">54.3</td><td>8.7</td><td class="siq-rank-3">56.2</td><td class="siq-rank-3">47.4</td><td>28.4</td><td class="siq-rank-3">32.8</td></tr>
    <tr><td>VLA-0</td><td>4.2</td><td>16.7</td><td>1.2</td><td>0.0</td><td>19.0</td><td>7.4</td><td>19.9</td></tr>
    <tr><td>GLM</td><td>3.5</td><td>39.9</td><td>2.9</td><td>32.8</td><td>16.4</td><td>9.5</td><td>8.2</td></tr>
    <tr><td>Qwen3B</td><td>2.1</td><td>17.1</td><td>0.2</td><td>0.0</td><td>19.0</td><td>6.4</td><td>24.7</td></tr>
  </tbody>
</table>
</div>
```

```{figure} _static/figures/fig_e1a.png
:class: siq-figure-img
:align: center

**Figure 7.** *Object Counting* accuracy, conditioned on correct performance on *Visible Object Count* (V) and *Hidden Object Count* (H), text free-response modality. Rows and columns mark correct (&check;) or incorrect (&times;) performance on the corresponding sub-task; cell color and label indicate conditional *Object Counting* performance over *n* scenes.
```

Three observations emerge:

1. **The human baseline shows the empirical signature of the Summation Mechanism.** *Object Counting* accuracy is concentrated in the both-correct cell (90.6%) and decays monotonically as either component count is missed, with a Summation Mechanism exact-match rate of 69.1%.
2. **Model recovery of the Summation Mechanism is uneven and partially independent of headline accuracy.** Qwen most closely approximates the human signature (88.9% in both-correct, 64.2% exact match). Gemini reaches comparable headline accuracy through a markedly weaker mechanism (48.0% / 18.6%). *Object Counting* accuracy and Summation Mechanism strength are not the same property.
3. **The pre-specified hierarchy is largely preserved.** Model rows preserve the qualitative ordering observed in humans: the *Internal Referential Chain* is consistently cleaner than the *Visible* and *Hidden Support Hierarchies*. VLA-0 outperforms its Qwen3B backbone across the support relations despite being trained only on robot trajectories — hinting that action-grounded training transfers positively to spatial reasoning.

### Multiple-choice modality

Under five-choice MCQ, every model collapses to chance on *Object Counting* while humans hold 86.2% accuracy. Reducing to four and three choices partially restores the expected Summation Mechanism, but recovery is uneven across models.

```{raw} html
<div class="siq-table-caption"><span class="siq-fig-label">Table 2.</span> MCQ modality evaluative summary (chance-adjusted % accuracy, &nbsp;adj_acc = (acc&minus;1/k)/(1&minus;1/k)). 3CQ / 4CQ / 5CQ shown side-by-side. Humans measured at 5CQ only.</div>
<div class="siq-table-wrap">
<table class="siq-table">
  <thead>
    <tr class="siq-mcq-group">
      <th rowspan="2">Responder</th>
      <th colspan="3">Obj. Count</th>
      <th colspan="3">Int. Ref. Chain</th>
      <th colspan="3">Visible</th>
      <th colspan="3">Vis. Support</th>
      <th colspan="3">Hidden</th>
      <th colspan="3">Hid. Support</th>
      <th colspan="3">Sum. Mech.</th>
    </tr>
    <tr>
      <th>3CQ</th><th>4CQ</th><th>5CQ</th>
      <th>3CQ</th><th>4CQ</th><th>5CQ</th>
      <th>3CQ</th><th>4CQ</th><th>5CQ</th>
      <th>3CQ</th><th>4CQ</th><th>5CQ</th>
      <th>3CQ</th><th>4CQ</th><th>5CQ</th>
      <th>3CQ</th><th>4CQ</th><th>5CQ</th>
      <th>3CQ</th><th>4CQ</th><th>5CQ</th>
    </tr>
  </thead>
  <tbody>
    <tr class="siq-human"><td>Human</td>
      <td>&mdash;</td><td>&mdash;</td><td>82.7</td>
      <td>&mdash;</td><td>&mdash;</td><td>97.0</td>
      <td>&mdash;</td><td>&mdash;</td><td>99.3</td>
      <td>&mdash;</td><td>&mdash;</td><td>99.3</td>
      <td>&mdash;</td><td>&mdash;</td><td>89.3</td>
      <td>&mdash;</td><td>&mdash;</td><td>89.6</td>
      <td>&mdash;</td><td>&mdash;</td><td>83.6</td>
    </tr>
    <tr class="siq-divider"><td>Claude</td>
      <td class="siq-rank-1">23.7</td><td class="siq-rank-1">17.8</td><td>0.0</td>
      <td class="siq-rank-1">40.7</td><td class="siq-rank-2">33.3</td><td class="siq-rank-2">25.0</td>
      <td class="siq-rank-2">44.6</td><td class="siq-rank-1">43.6</td><td class="siq-rank-1">40.7</td>
      <td class="siq-rank-2">44.7</td><td class="siq-rank-1">43.8</td><td class="siq-rank-1">40.4</td>
      <td>8.0</td><td class="siq-rank-2">10.7</td><td>11.7</td>
      <td>5.3</td><td>10.6</td><td class="siq-rank-2">12.7</td>
      <td class="siq-rank-1">26.8</td><td class="siq-rank-1">26.1</td><td>&minus;2.2</td>
    </tr>
    <tr><td>Gemini</td>
      <td>14.0</td><td>12.8</td><td>&minus;0.6</td>
      <td class="siq-rank-2">38.0</td><td>24.2</td><td>21.3</td>
      <td class="siq-rank-1">45.9</td><td class="siq-rank-2">35.6</td><td class="siq-rank-2">32.6</td>
      <td class="siq-rank-1">47.3</td><td class="siq-rank-2">36.6</td><td class="siq-rank-2">37.8</td>
      <td class="siq-rank-1">21.9</td><td class="siq-rank-1">17.7</td><td class="siq-rank-1">14.1</td>
      <td class="siq-rank-1">22.9</td><td class="siq-rank-1">18.8</td><td class="siq-rank-1">17.0</td>
      <td class="siq-rank-2">21.4</td><td class="siq-rank-2">20.0</td><td>&minus;2.9</td>
    </tr>
    <tr><td>GPT</td>
      <td>7.9</td><td>5.9</td><td class="siq-rank-1">0.9</td>
      <td>36.7</td><td class="siq-rank-1">34.6</td><td class="siq-rank-1">35.4</td>
      <td>42.1</td><td>24.3</td><td>24.0</td>
      <td>42.0</td><td>25.0</td><td>23.6</td>
      <td class="siq-rank-2">15.0</td><td>10.4</td><td class="siq-rank-2">12.6</td>
      <td class="siq-rank-2">13.9</td><td class="siq-rank-2">11.1</td><td>12.3</td>
      <td>12.9</td><td>16.9</td><td class="siq-rank-1">2.2</td>
    </tr>
    <tr><td>Qwen</td>
      <td class="siq-rank-2">20.5</td><td class="siq-rank-2">14.8</td><td class="siq-rank-2">0.2</td>
      <td>32.0</td><td>21.6</td><td>10.2</td>
      <td>36.5</td><td>17.3</td><td>8.0</td>
      <td>34.4</td><td>16.7</td><td>7.3</td>
      <td>&minus;20.9</td><td>&minus;12.6</td><td>&minus;2.0</td>
      <td>&minus;20.6</td><td>&minus;13.1</td><td>&minus;1.6</td>
      <td>7.4</td><td>5.7</td><td class="siq-rank-2">1.0</td>
    </tr>
  </tbody>
</table>
</div>
```

### Image-editing modality

In the image-editing format, the model directly edits the reference image to color the objects specified by the sub-task (e.g., highlighting the top layer or the visible objects). The qualitative pattern from text and MCQ replicates: the *Internal Referential Chain* remains the most format-stable backbone, and target-task success and local hierarchy adherence are partially dissociable. Qwen-Image-Edit gives the clearest example — strongest Internal Referential Chain integrity in the modality (0.676), but raw *Visible Object Count* of only 3.3%.

```{raw} html
<div class="siq-table-caption"><span class="siq-fig-label">Table 3.</span> Image-editing modality evaluative summary (% accuracy). Columns parallel Table 1.</div>
<div class="siq-table-wrap">
<table class="siq-table">
  <thead>
    <tr>
      <th>Responder</th>
      <th>Obj. Count</th>
      <th>Int. Ref. Chain</th>
      <th>Visible</th>
      <th>Vis. Support</th>
      <th>Hidden</th>
      <th>Hid. Support</th>
      <th>Sum. Mech.</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Gemini Flash Image</td><td class="siq-rank-1">14.7</td><td class="siq-rank-2">53.7</td><td class="siq-rank-1">36.1</td><td class="siq-rank-2">37.0</td><td class="siq-rank-1">46.5</td><td class="siq-rank-1">36.5</td><td class="siq-rank-1">29.5</td></tr>
    <tr><td>Qwen-Image-Edit</td><td class="siq-rank-2">12.0</td><td class="siq-rank-1">67.6</td><td>3.3</td><td class="siq-rank-1">58.7</td><td class="siq-rank-2">38.7</td><td>11.2</td><td class="siq-rank-2">23.8</td></tr>
    <tr><td>HunyuanImage-Instruct</td><td>5.9</td><td>50.1</td><td class="siq-rank-2">10.8</td><td>32.2</td><td>30.5</td><td class="siq-rank-2">19.0</td><td>9.4</td></tr>
  </tbody>
</table>
</div>
```

## Training on Spatial-IQ Sub-Tasks

The structure of the benchmark suggests a natural training signal: a model trained to walk through the sub-task decomposition before committing to a final count should learn the prerequisite competencies. We test this on **Qwen2.5-VL-7B-Instruct** and **Qwen2.5-VL-32B-Instruct** under three conditions:

- **SFT-plain** — supervised fine-tuning on the integer total alone, no decomposition.
- **SFT-CoT** — SFT on a chain-of-thought target listing sub-task answers in the order required for additive composition.
- **DAPO-tight** — brief SFT warmup on 10% of the training data, followed by RL with verifiable rewards (GRPO under the DAPO recipe) on the remaining 90%.

```{raw} html
<div class="siq-table-caption"><span class="siq-fig-label">Table 4.</span> Trained Qwen2.5-VL checkpoints, text modality (% accuracy). Values in parentheses are absolute point gains over the corresponding zero-shot backbone. Per-column rankings within each scale group (humans excluded; zero-shot baselines included): <span style="background:#8cb4ff;padding:0 4px;">first</span>, <span style="background:#bed7ff;padding:0 4px;">second</span>.</div>
<div class="siq-table-wrap">
<table class="siq-table">
  <thead>
    <tr>
      <th>Responder</th>
      <th>Obj. Count</th>
      <th>Int. Ref. Chain</th>
      <th>Visible</th>
      <th>Vis. Support</th>
      <th>Hidden</th>
      <th>Hid. Support</th>
      <th>Sum. Mech.</th>
    </tr>
  </thead>
  <tbody>
    <tr class="siq-human"><td>Human</td><td>82.1</td><td>90.1</td><td>76.8</td><td>84.5</td><td>79.9</td><td>78.5</td><td>69.1</td></tr>
    <tr class="siq-divider"><td>DAPO-tight 7B</td>
      <td class="siq-rank-1">50.7 <small>(+46.4)</small></td>
      <td class="siq-rank-1">93.7 <small>(+45.6)</small></td>
      <td class="siq-rank-1">51.7 <small>(+48.3)</small></td>
      <td class="siq-rank-2">66.9 <small>(+65.6)</small></td>
      <td class="siq-rank-2">76.0 <small>(+30.7)</small></td>
      <td class="siq-rank-1">85.5 <small>(+28.9)</small></td>
      <td class="siq-rank-1">90.0 <small>(+45.3)</small></td>
    </tr>
    <tr><td>SFT-CoT 7B</td>
      <td class="siq-rank-2">40.8 <small>(+36.5)</small></td>
      <td class="siq-rank-1">93.7 <small>(+45.6)</small></td>
      <td class="siq-rank-2">41.9 <small>(+38.5)</small></td>
      <td>60.5 <small>(+59.2)</small></td>
      <td class="siq-rank-1">76.7 <small>(+31.4)</small></td>
      <td class="siq-rank-2">85.2 <small>(+28.6)</small></td>
      <td class="siq-rank-2">89.6 <small>(+44.9)</small></td>
    </tr>
    <tr><td>SFT-plain 7B</td>
      <td>39.7 <small>(+35.4)</small></td>
      <td>0.0 <small>(&minus;48.1)</small></td>
      <td>29.2 <small>(+25.8)</small></td>
      <td class="siq-rank-1">80.3 <small>(+79.0)</small></td>
      <td>0.0 <small>(&minus;45.3)</small></td>
      <td>0.0 <small>(&minus;56.6)</small></td>
      <td>0.0 <small>(&minus;44.7)</small></td>
    </tr>
    <tr><td>Qwen2.5-VL-7B (zero-shot)</td><td>4.3</td><td>48.1</td><td>3.4</td><td>1.3</td><td>45.3</td><td>56.6</td><td>44.7</td></tr>
    <tr class="siq-divider"><td>DAPO-tight 32B</td>
      <td class="siq-rank-1">62.6 <small>(+59.7)</small></td>
      <td class="siq-rank-2">95.0 <small>(+53.9)</small></td>
      <td class="siq-rank-1">64.5 <small>(+62.6)</small></td>
      <td class="siq-rank-2">73.7 <small>(+62.3)</small></td>
      <td class="siq-rank-2">78.2 <small>(+27.8)</small></td>
      <td class="siq-rank-1">86.1 <small>(+5.8)</small></td>
      <td class="siq-rank-1">93.3 <small>(+42.3)</small></td>
    </tr>
    <tr><td>SFT-CoT 32B</td>
      <td class="siq-rank-2">46.7 <small>(+43.8)</small></td>
      <td class="siq-rank-1">95.5 <small>(+54.4)</small></td>
      <td class="siq-rank-2">48.9 <small>(+47.0)</small></td>
      <td>61.3 <small>(+49.9)</small></td>
      <td class="siq-rank-1">79.2 <small>(+28.8)</small></td>
      <td class="siq-rank-2">85.7 <small>(+5.4)</small></td>
      <td class="siq-rank-2">92.9 <small>(+41.9)</small></td>
    </tr>
    <tr><td>SFT-plain 32B</td>
      <td>40.1 <small>(+37.2)</small></td>
      <td>0.0 <small>(&minus;41.1)</small></td>
      <td>30.9 <small>(+29.0)</small></td>
      <td class="siq-rank-1">85.0 <small>(+73.6)</small></td>
      <td>0.0 <small>(&minus;50.4)</small></td>
      <td>0.0 <small>(&minus;80.3)</small></td>
      <td>0.0 <small>(&minus;51.0)</small></td>
    </tr>
    <tr><td>Qwen2.5-VL-32B (zero-shot)</td><td>2.9</td><td>41.1</td><td>1.9</td><td>11.4</td><td>50.4</td><td>80.3</td><td>51.0</td></tr>
  </tbody>
</table>
</div>
```

**Sub-task decomposition is a real training signal, and the gain compounds with scale.** At 7B, SFT-CoT and SFT-plain are nearly tied on *Object Counting* (1.1-pt gap), but at 32B the decomposition produces a 6.6-pt gap in its favor. More importantly, the two diverge sharply on the hierarchy columns: **SFT-plain collapses to zero** on *Internal Referential Chain*, *Hidden Support Hierarchy*, and *Summation Mechanism* — indicating it has learned a route to the right total that *bypasses the decomposition entirely*. SFT-CoT preserves all four hierarchy relations.

**RL with verifiable rewards adds substantial gain on top of SFT-CoT.** DAPO-tight beats SFT-CoT by 9.9 absolute points at 7B and 15.9 at 32B, with the 32B DAPO-tight model reaching **62.6%** *Object Counting* accuracy where the zero-shot backbone achieved 2.9%. The reward signal is on the integer total only, so these gains come from sharpening the policy onto chain-of-thought trajectories that lead to correct totals — not from new sub-task supervision.

## Mental Rotation as a Complementary Probe

Our second target task addresses a distinct facet of spatial cognition: viewpoint transformation. Empirically, the proposed hierarchy aligns much more strongly with *Object Counting* than with *Mental Rotation*, while the two rotation tasks ($90^\circ$ and $180^\circ$) align primarily with each other. Humans perform worse on *Mental Rotation* (46.4% / 45.8%) than on *Object Counting*, which limits the headroom available for distinguishing models. The Spatial-IQ hierarchy is therefore a faithful decomposition of *Object Counting* but not of *Mental Rotation* — they probe largely separable competencies.

## Putting It Together: Hierarchy as Both Diagnostic and Training Signal

The motivation, benchmark design, evaluation results, and training results above converge on a single observation: **for the compositional spatial tasks Spatial-IQ probes, what distinguishes humans from current MLLMs is not better counting but the ability to *compose* the answer from prerequisite sub-capabilities** — and the same hierarchy that surfaces this gap is also the training signal that closes it.

1. **Humans show the empirical signature that the hierarchical view predicts.** *Object Counting* accuracy concentrates in the both-correct cell (90.6%) and decays monotonically as either component count is missed; the Summation Mechanism exact-match rate is 69.1%. Humans reach the right total because *visible count + hidden count → total* actually runs, exactly as the developmental hierarchy says it should.

2. **Current MLLMs do *not* automatically inherit this structure — target-task performance and hierarchy adherence dissociate.** Qwen and Gemini sit within a few points of each other on raw *Object Counting* (17.7% vs 14.8%), yet their Summation Mechanism exact-match rates differ by more than 3× (64.2% vs 18.6%). A model can hit the integer by shortcut.

3. **Forcing the hierarchy during training installs the missing structure; bypassing it caps the gain.** At 32B, **SFT-plain** reaches 40.1% but collapses to **0%** on the Internal Referential Chain / Hidden Support / Summation Mechanism. **SFT-CoT** preserves all four hierarchy relations and reaches 46.7%; **DAPO-tight** reaches **62.6%** with the chain intact (Internal Referential Chain 95.0%, Summation Mechanism 93.3%). The reward never names the sub-tasks — RL sharpens the policy onto chain-of-thought trajectories that lead to correct totals.

In humans, headline accuracy *implies* hierarchy adherence; in current MLLMs by default, it does not. The diagnostic dissociation (#2) and the causal training result (#3) tell the same story: teach a model the composition and the final integer follows; let it skip the composition and target-task accuracy caps at the level of a shortcut.

## Discussion

Spatial-IQ reframes spatial-intelligence evaluation as a hierarchical decomposition and shows that the same hierarchy serves both as a **diagnostic instrument** and as a **training signal**. Top models often succeed on *Object Counting* while failing the underlying sub-tasks. Chain-of-thought supervision over the hierarchy, combined with verifiable-reward reinforcement learning, closes much of the gap to the human ceiling while preserving the intended chain. The finding is task-specific: the Spatial-IQ hierarchy faithfully decomposes *Object Counting* but not *Mental Rotation*, suggesting that **compositional** spatial tasks and **transformational** spatial tasks recruit separable competencies — and that future spatial-intelligence benchmarks should pair each target task with its own task-appropriate hierarchy rather than treating "spatial intelligence" as a single scalar. Future work includes extending this framework to other classical spatial-intelligence tasks and a broader investigation of whether action-grounded training in vision-language-action models systematically benefits spatial reasoning.

## BibTeX

```{raw} html
<pre class="siq-bibtex">@misc{rim2026spatialiq,
  title         = {Spatial-IQ: Deconstructing Spatial Intelligence via Hierarchical Capability Tests},
  author        = {Rim, Patrick and Long, Tom and Prashnani, Ekta and Rosenholtz, Ruth and
                   Boudaoud, Ben and Xenopoulos, Peter and Wong, Alex and Kim, Joohwan and
                   Jung, Jae-Hyun},
  year          = {2026},
  eprint        = {2607.22864},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2607.22864},
  url           = {https://arxiv.org/abs/2607.22864},
}</pre>
```
