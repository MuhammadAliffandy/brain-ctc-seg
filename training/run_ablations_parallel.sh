#!/bin/bash
# run_ablations_parallel.sh
# Runs the ablation experiments for the IEEE TMI revision.

# Ensure we're in the right directory
cd "$(dirname "$0")"

# We will run them sequentially on GPU 0 to avoid OOM if memory is tight,
# but can be customized.
export CUDA_VISIBLE_DEVICES=0

echo "Starting Ablation Training Pipeline..."

# -------------------------------------------------------------
# TABLE 1 ABLATION (A1 to A8)
# We use focal_dice_boundary as the "Boundary Loss = Yes"
# and focal_dice as the "Boundary Loss = No" to isolate the boundary component.
# Actually, the user's table just says "Boundary Loss", let's use:
# No Boundary = focal_dice
# Yes Boundary = focal_dice_boundary
# -------------------------------------------------------------

# A1: SE2=0, 2.5D=1, Boundary=0
python train_ablation.py --resume --variant A1 --dataset ct --se2 0 --slices 1 --loss focal_dice
# A2: SE2=1, 2.5D=1, Boundary=0
python train_ablation.py --resume --variant A2 --dataset ct --se2 1 --slices 1 --loss focal_dice
# A3: SE2=0, 2.5D=3, Boundary=0
python train_ablation.py --resume --variant A3 --dataset ct --se2 0 --slices 3 --loss focal_dice
# A4: SE2=0, 2.5D=1, Boundary=1
python train_ablation.py --resume --variant A4 --dataset ct --se2 0 --slices 1 --loss focal_dice_boundary
# A5: SE2=1, 2.5D=3, Boundary=0
python train_ablation.py --resume --variant A5 --dataset ct --se2 1 --slices 3 --loss focal_dice
# A6: SE2=1, 2.5D=1, Boundary=1
python train_ablation.py --resume --variant A6 --dataset ct --se2 1 --slices 1 --loss focal_dice_boundary
# A7: SE2=0, 2.5D=3, Boundary=1
python train_ablation.py --resume --variant A7 --dataset ct --se2 0 --slices 3 --loss focal_dice_boundary
# A8: SE2=1, 2.5D=3, Boundary=1 (Full CT-SE2)
python train_ablation.py --resume --variant A8 --dataset ct --se2 1 --slices 3 --loss focal_dice_boundary


# -------------------------------------------------------------
# TABLE 2: LOSS CONFIGURATION (SE2=1, 2.5D=3)
# -------------------------------------------------------------
python train_ablation.py --resume --variant Loss_DiceOnly --dataset ct --se2 1 --slices 3 --loss dice
python train_ablation.py --resume --variant Loss_FocalDice --dataset ct --se2 1 --slices 3 --loss focal_dice
python train_ablation.py --resume --variant Loss_DiceBound --dataset ct --se2 1 --slices 3 --loss dice_boundary
# Loss_FocalDiceBound is already A8


# -------------------------------------------------------------
# TABLE 3: INPUT CONTEXT (SE2=1, Boundary=1)
# -------------------------------------------------------------
# 2D is already A6 (SE2=1, 1-slice, Boundary=1)
# 2.5D is already A8 (SE2=1, 3-slice, Boundary=1)
# 5-slice:
python train_ablation.py --resume --variant Context_5D --dataset ct --se2 1 --slices 5 --loss focal_dice_boundary

echo "All trainings completed!"
