# Model Comparison

We evaluated several CNN architectures with different optimizers and configurations for an image classification task with 38 classes. The following table presents the performance metrics for each model variant:

| Training Instance | Optimizer | Regularizer | Epochs | Early Stopping | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|-----------|-------------|---------|----------------|------------------|---------------|-----------|-----------|---------|------------|
| Model 1 | Nadam | l2(0.001) | 10 | No | 6 (2 Conv2D, 2 Dense) | 0.0001 | 0.8774 | 0.8785 | 0.8774 | 0.8876 |
| Model 2 | RMSprop | l1(0.0005) | 10 | No | 6 (2 Conv2D, 2 Dense) | 0.0001 | 0.9125 | 0.9119 | 0.9125 | 0.9145 |
| Model 3 | Adam | None | 10 | No | 6 (2 Conv2D, 2 Dense) | 0.001 | 0.8808 | 0.8856 | 0.8808 | 0.9020 |
| Model 4 | SGD | l2(0.001) | 10 | No | 6 (2 Conv2D, 2 Dense) | 0.001 | 0.9118 | 0.9112 | 0.9118 | 0.9183 |
| Model 5 (Simple NN) | None | None | 10 | No | 7 (3 Conv2D, 2 Dense) | Default | 0.9176 | 0.9177 | 0.9178 | 0.9213 |

## Key Findings

- **Best Performance**: Model 5 (Simple NN) achieved the highest overall performance with accuracy of 91.76%
- **Optimizer Impact**: RMSprop and Nadam optimizers showed strong performance, achieving >91% accuracy
- **Architecture**: Adding a third Conv2D layer in the simple NN improved model performance
- **Consistency**: All models performed well, with accuracies ranging from 87.74% to 91.76%
