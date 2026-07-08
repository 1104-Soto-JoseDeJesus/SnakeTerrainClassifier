# Snake Terrain Classifier

This project is a software parallel to a robotic snake terrain-recognition system.  
It generates synthetic joint-angle and IMU-like sensor data for a virtual snake robot, then tests multiple machine learning and deep learning methods for terrain classification.

## Project Goal

The goal is to classify terrain type from time-series sensor data, similar to how a robotic snake might use joint and IMU sensors to infer the surface it is moving across.

Terrains simulated:

- Smooth
- Carpet
- Gravel
- Incline
- Bumpy

## Pipeline

1. Generate synthetic snake sensor data
2. Plot and inspect signals
3. Convert raw time-series data into windows
4. Train baseline machine learning models
5. Train deep learning models
6. Engineer signal features
7. Train tuned tree-based models
8. Apply specialist/hierarchical classifiers
9. Evaluate window-level and trial-level accuracy

## Best Result

The best result was achieved using:

- 4-second windows
- engineered time-series features
- hierarchical specialist Extra Trees classifier
- trial-level majority voting

Best trial-level accuracy:

```text
86.67%