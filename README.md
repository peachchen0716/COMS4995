# CU-DL-project
This repo contains baseline model for our stock price prediction model.

## Dataset
We use fulldata.csv to train/test our model.

## Training model and Testing
To train this model, first replace your local gym environment with our gym environment. Then run the following command to obtain train result
```
python -m baselines.run --alg=ddpg --env=Stock-v0 --network=mlp --num_timesteps=1e4
```
To test the trained model
```
python -m baselines.run --alg=ddpg --env=Stock-v0 --network=mlp --num_timesteps=1e4 --play
```
