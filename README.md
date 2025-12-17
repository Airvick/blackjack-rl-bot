# Blackjack Reinforcement Learning Bot

This project implements a Blackjack simulator and a reinforcement learning–based policy trained through large-scale self-play simulations.

## Overview
The goal of this project is to explore decision-making policies in Blackjack using reinforcement learning techniques and to evaluate their performance through millions of simulated rounds.

## Key Features
- Custom Blackjack game engine
- Reinforcement learning policy training in chunks
- CLI-based policy usage
- Large-scale simulation and evaluation
- Policy analysis and visualization scripts

## Training Environment
The policy was trained on the FHGR Mercury server using NVIDIA A100 GPUs.

## Project Structure
- `blackjack_engine.py` – core game logic
- `blackjack_train_policy_chunks.py` – RL training pipeline
- `simulate_with_learned_policy.py` – large-scale simulation
- `blackjack_policy_cli.py` – CLI decision helper
- `analyze_policy_weakspots.py` – policy analysis
- `simulate_with_plot.py` – result visualization

## Example Result
- Simulated rounds: 10,000,000
- Average profit per round: approximately -0.0098 units

## Disclaimer
This project is intended for educational and research purposes only.
It does not constitute gambling advice.
