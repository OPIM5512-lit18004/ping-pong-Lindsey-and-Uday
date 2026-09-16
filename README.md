# Ping Pong: California Housing MLPRegressor
 
## What this project does
This is a collaborative GitHub "ping-pong" exercise for OPIM 5512. We practice
branch → pull request → review → merge workflows while building a small
regression pipeline on the scikit-learn California Housing dataset.
 
The pipeline:
- Loads the California Housing dataset
- Splits it into train/test sets
- Trains an MLPRegressor neural network to predict median house value
- Saves "actual vs. predicted" plots for both the train and test sets to figures/
 
## How to run
1. Clone the repo and cd into it.
2. Install dependencies:
   pip install pandas numpy matplotlib scikit-learn
3. Run the pipeline:
   python Source/nn_pipeline.py
4. Output plots are saved to the figures/ folder:
- figures/train_actual_vs_pred.png
- figures/test_actual_vs_pred.png
 
## Partners
- Lindsey ([@lindsey3291](https://github.com/lindsey3291))
- Uday ([@UdayAnalyst](https://github.com/UdayAnalyst))