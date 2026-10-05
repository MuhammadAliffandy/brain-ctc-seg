#!/bin/bash
echo "========================================="
echo "🚀 STARTING REVIEWER REBUTTAL SCRIPTS 🚀"
echo "========================================="

echo ""
echo "[1/2] Re-calculating ROC Curves without subsampling..."
python paper_evalute/fix_roc_curve.py --dataset ctc
python paper_evalute/fix_roc_curve.py --dataset kaggle
python paper_evalute/fix_roc_curve.py --dataset kaggle_hemorrhage

echo ""
echo "[2/2] Re-calculating HD95 with correct physical pixel spacing..."
python paper_evalute/fix_hd95_metrics.py --dataset ctc
python paper_evalute/fix_hd95_metrics.py --dataset kaggle
python paper_evalute/fix_hd95_metrics.py --dataset kaggle_hemorrhage

echo ""
echo "✅ ALL REBUTTAL METRICS COMPLETED SUCCESSFULLY!"
