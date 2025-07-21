# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an interactive web-based autoencoder visualization demo that demonstrates how neural networks compress high-dimensional data (784D handwritten digits) into 2D latent space representations. The application uses TensorFlow.js for the neural network and D3.js for data visualization.

## Running the Application

Start a local web server in the project directory:
```bash
python3 -m http.server 8000
```
Then open http://localhost:8000 in a web browser.

## Architecture

### Core Components

1. **AutoencoderDemo Class** (`autoencoder.js`)
   - Main application controller
   - Manages model, training data, and visualization state
   - Key properties:
     - `model`: Full autoencoder (encoder + decoder)
     - `encoder`: Standalone encoder for latent space projection
     - `decoder`: Standalone decoder for reconstruction
     - `latentSpaceData`: Stores encoded training samples for visualization
     - `trainingData`: Raw training samples {data, label}

2. **Neural Network Architecture**
   - Encoder: 784 → 128 → 64 → 2 (no activation on final layer)
   - Decoder: 2 → 64 → 128 → 784 (sigmoid activation)
   - Loss: Binary crossentropy
   - Optimizer: Adam (learning rate 0.002)

3. **Visualization System**
   - D3.js-based latent space scatter plot
   - Real-time animation during training (shows all 50 epochs)
   - Fixed axis scaling during training to show movement
   - Color-coded digits with custom palette

### Key Methods

- `trainModel()`: Handles the training process with visual updates every epoch
- `updateLatentSpaceAnimation()`: Updates visualization during training
- `makePrediction()`: K-NN based digit classification in latent space
- `initializeLatentPoints()`: Sets up initial positions before training animation

## Important Implementation Details

1. **Canvas Dimensions**: Drawing canvas is 280x280, scaled down to 28x28 for model input
2. **Display Canvases**: Input/output displays are 120x120 (not 140x140)
3. **Training Animation**: Shows every epoch (50 total) with 100ms transitions
4. **Mobile Support**: Tab-based navigation on screens ≤768px wide
5. **Consistent Point IDs**: Each training sample gets ID `sample_${index}` for smooth D3 transitions

## Common Issues and Solutions

1. **Null Reference Errors**: The 'nearestDigit' element was removed from HTML but may still be referenced in older code
2. **Canvas Sizing**: Ensure all canvas operations use correct dimensions (120x120 for displays)
3. **Training Animation**: Use `updateLatentSpaceSmoothNoRescale()` to prevent axis jumping during training
4. **Loss Calculation**: Uses manual binary crossentropy implementation due to TensorFlow.js API limitations

## Training Best Practices

- Collect 5-10 samples per digit (0-9) before training
- Train once with 50 epochs for best results
- Multiple training sessions risk overfitting
- Reset model to start fresh if results are poor