# DCTMemory Paper Brief for Abstract and Introduction

## Working Title

DCTMemory: Frequency-Domain Action Memory for Long-Horizon Vision-Language-Action Policies

## One-Sentence Summary

We introduce a two-timescale vision-language-action policy that predicts a low-frequency coarse future action trajectory with DCT tokens, refines short-horizon actions with a fast action head, and uses DCTMemory to compactly summarize previously executed actions for long-horizon manipulation.

## Core Problem

Current vision-language-action policies usually predict short action chunks from the current observation and language instruction. This is efficient for reactive control, but it makes long-horizon manipulation difficult because the model has limited explicit knowledge of what it has already executed. In multi-stage tasks, the same visual observation can be ambiguous unless the policy remembers recent progress, completed subgoals, or the trajectory already taken.

A naive solution is to feed the full action history or observation history into the policy. However, raw histories are expensive, grow with time, and can introduce train-inference mismatch. For robotic control, action trajectories are temporally smooth and often dominated by low-frequency structure. This makes frequency-domain compression a natural way to represent both future plans and past execution history.

## Main Idea

The method uses action trajectories as the interface between planning and control. Instead of relying only on visual-language features, the policy explicitly models actions at two temporal scales:

1. A slow VLM pathway predicts a coarse long-horizon action trajectory in the DCT frequency domain.
2. A fast action head predicts short-horizon residual actions conditioned on the current observation and the coarse action prior.
3. DCTMemory summarizes previously executed actions in a compact frequency-domain memory, allowing the policy to know what it has already done without storing a long raw trajectory.

This creates a lightweight memory mechanism that is directly tied to the robot's executed behavior, rather than requiring the VLM to infer task progress only from images.

## Two-Chunk Action Prediction

The policy separates action generation into a long chunk and a short chunk.

The slow branch uses VLM action tokens to predict low-frequency DCT coefficients for a longer future horizon, such as 32 control steps. Applying inverse DCT gives a coarse action trajectory. This coarse trajectory captures smooth, global motion trends and serves as a prior.

The fast branch predicts a shorter executable chunk, such as 8 steps. Instead of predicting the full action from scratch, it predicts a residual relative to the coarse trajectory. The final action is:

```text
final_action = coarse_action + residual_action
```

This decomposition lets the VLM focus on long-horizon semantic motion structure, while the fast action head handles local visual feedback and fine control.

## DCTMemory

DCTMemory stores previously executed actions in a compact frequency-domain form. Rather than keeping all past actions, the system periodically compresses completed action chunks using DCT and maintains a low-frequency summary of the past trajectory.

For example, executed actions can be grouped into fixed-size chunks. Each completed chunk is transformed into DCT coefficients. The memory summary is then updated by reconstructing the previous summary, appending the new chunk, and compressing the merged trajectory again. This repeated compression keeps the memory size fixed while preserving the dominant low-frequency structure of the action history.

DCTMemory is useful because past actions provide an explicit signal of task progress. In long-horizon manipulation, the policy often needs to know not only what is visible now, but also what has already been attempted or completed. A compact action memory gives the model this information without requiring long visual context windows.

## Why DCT Is Suitable

Robot action trajectories are usually smooth over short and medium horizons. Low-frequency DCT coefficients capture the dominant shape of the trajectory, while discarding high-frequency details that are often less important for long-term progress estimation. DCT also has practical advantages:

- It is deterministic and lightweight.
- It has no recurrent hidden state that must be learned from scratch.
- It produces fixed-size memory tokens independent of episode length.
- It naturally supports progressive online updates during rollout.
- It provides an interpretable compression of past and future actions.

## Training and Inference Consistency

A key design requirement is that the memory used during training should match the memory available during inference. During training, DCTMemory should be constructed from the action history preceding the sampled frame. During inference, the memory is updated online from actions predicted and executed by the policy.

To reduce train-inference mismatch, the memory format should be identical in both cases: completed action chunks are compressed into a DCT summary, while incomplete recent actions are either excluded from the summary or handled by a separate recent-history mechanism. The summary represents only completed chunks, which makes its update rule simple and consistent.

Noise, dropout, or gating can be applied to memory tokens during training so that the policy does not over-rely on memory. This is important because inference-time rollouts may deviate from successful training trajectories, and memory should act as an auxiliary progress cue rather than a brittle source of truth.

## Method Overview

The full model consists of three main components:

1. Vision-language backbone: encodes the current observation and task instruction, and produces action tokens for long-horizon planning.
2. DCT coarse planner: predicts low-frequency DCT coefficients for a long future action chunk, which are decoded into a coarse action prior.
3. Fast residual action head: predicts short-horizon residual actions using the current visual features, VLM action tokens, the coarse prior, and optionally DCTMemory tokens.

DCTMemory is added as additional conditioning information. It summarizes what the policy has already executed and gives the action head a compact representation of task progress.

## Expected Contributions

- A two-timescale action prediction framework that separates long-horizon coarse planning from short-horizon residual control.
- A DCT-based action prior that represents future motion with low-frequency coefficients.
- DCTMemory, a compact frequency-domain memory for previously executed actions.
- A train-inference consistent memory update rule based on completed action chunks.
- A lightweight memory mechanism that improves long-horizon manipulation without storing long raw histories.

## Suggested Abstract Direction

The abstract should emphasize that long-horizon robotic manipulation requires memory of previously executed behavior, but raw history is inefficient and difficult to align between training and inference. The proposed method uses DCT to represent both future action priors and past action memory. A slow VLM branch predicts coarse long-horizon DCT actions, a fast branch refines short executable chunks through residual prediction, and DCTMemory compactly summarizes completed action chunks. The method provides an efficient and interpretable temporal memory for VLA policies and is designed for consistent online rollout.

## Suggested Introduction Structure

1. Start from the challenge of long-horizon VLA control: current observation and instruction are often insufficient because the policy must know what has already happened.
2. Explain why simply increasing context length or storing raw histories is costly and can cause train-inference mismatch.
3. Introduce the observation that robot action trajectories are smooth and can be compactly represented in the frequency domain.
4. Present the two-timescale design: DCT coarse planning plus short-horizon residual refinement.
5. Introduce DCTMemory as a compact memory of executed actions, enabling the model to track progress through its own behavior.
6. End with the main contributions and expected benefits for long-horizon manipulation benchmarks.

## Important Framing

DCTMemory should be framed as an action-centric memory mechanism. It is not merely a cache of hidden states or visual tokens. It stores what the robot has actually done, compressed into a stable and interpretable frequency-domain representation. This makes it especially suitable for manipulation tasks where progress is defined by the sequence of executed motions.

The method should also be framed as complementary to VLM reasoning. The VLM provides semantic task understanding and high-level motion structure, while DCTMemory provides rollout progress information. The memory does not replace visual-language reasoning; it gives the action head a compact temporal context that is difficult to infer from a single frame.
