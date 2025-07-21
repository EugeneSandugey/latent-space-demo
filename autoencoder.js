// Autoencoder Latent Space Visualization
// Main application logic

class AutoencoderDemo {
    constructor() {
        this.model = null;
        this.encoder = null;
        this.decoder = null;
        this.trainingData = [];
        this.isDrawing = false;
        this.ctx = null;
        this.latentSpaceData = [];
        // Beautiful color palette for digits
        this.colorScale = d3.scaleOrdinal()
            .domain([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .range(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                    '#F7DC6F', '#BB8FCE', '#85C1F2', '#F8B195', '#C7CEEA']);
        this.totalEpochs = 0;
        
        this.init();
    }

    init() {
        // Initialize canvas
        this.setupDrawingCanvas();
        
        // Initialize model
        this.createModel();
        
        // Initialize latent space visualization
        this.setupLatentSpace();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Setup mobile tabs
        this.setupMobileTabs();
        
        // Initial UI update
        this.updateUI();
    }

    setupDrawingCanvas() {
        const canvas = document.getElementById('drawingCanvas');
        this.ctx = canvas.getContext('2d');
        
        // Set white background
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Adjust line width for smaller canvas
        this.lineWidth = canvas.width === 240 ? 15 : 20;
        
        // Canvas drawing events
        canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        canvas.addEventListener('mousemove', (e) => this.draw(e));
        canvas.addEventListener('mouseup', () => this.stopDrawing());
        canvas.addEventListener('mouseleave', () => this.stopDrawing());
        
        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => this.startDrawing(e.touches[0]));
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        });
        canvas.addEventListener('touchend', () => this.stopDrawing());
    }

    startDrawing(e) {
        this.isDrawing = true;
        const rect = e.target.getBoundingClientRect();
        const canvas = e.target;
        
        // Calculate the actual canvas scale
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        // Get coordinates and scale them
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
    }

    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = e.target.getBoundingClientRect();
        const canvas = e.target;
        
        // Calculate the actual canvas scale
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        // Get coordinates and scale them
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        this.ctx.lineWidth = this.lineWidth;
        this.ctx.lineCap = 'round';
        this.ctx.strokeStyle = 'black';
        
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        
        // Update preview in real-time
        if (this.model) {
            this.updatePreview();
        }
    }

    stopDrawing() {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        this.ctx.beginPath();
        
        // Update preview
        if (this.model) {
            this.updatePreview();
        }
    }

    clearCanvas() {
        const canvas = document.getElementById('drawingCanvas');
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Clear displays
        const inputDisplay = document.getElementById('inputDisplay').getContext('2d');
        inputDisplay.fillStyle = 'white';
        inputDisplay.fillRect(0, 0, 60, 60);
        
        const outputDisplay = document.getElementById('outputDisplay').getContext('2d');
        outputDisplay.fillStyle = 'white';
        outputDisplay.fillRect(0, 0, 60, 60);
        
        document.getElementById('reconstructionLoss').textContent = '-';
        this.clearPrediction();
        
    }

    createModel() {
        // Encoder with better initialization
        const encoderInput = tf.input({shape: [28, 28, 1]});
        const x1 = tf.layers.flatten().apply(encoderInput);
        const x2 = tf.layers.dense({
            units: 128, 
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }).apply(x1);
        const x3 = tf.layers.dense({
            units: 64, 
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }).apply(x2);
        const latent = tf.layers.dense({
            units: 2, 
            name: 'latent',
            kernelInitializer: 'glorotNormal',
            // No activation on latent layer to allow full range
        }).apply(x3);
        
        this.encoder = tf.model({inputs: encoderInput, outputs: latent});
        
        // Decoder with better initialization
        const decoderInput = tf.input({shape: [2]});
        const y1 = tf.layers.dense({
            units: 64, 
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }).apply(decoderInput);
        const y2 = tf.layers.dense({
            units: 128, 
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }).apply(y1);
        const y3 = tf.layers.dense({
            units: 784, 
            activation: 'sigmoid',
            kernelInitializer: 'glorotNormal'
        }).apply(y2);
        const decoderOutput = tf.layers.reshape({targetShape: [28, 28, 1]}).apply(y3);
        
        this.decoder = tf.model({inputs: decoderInput, outputs: decoderOutput});
        
        // Full autoencoder
        const autoencoderOutput = this.decoder.apply(this.encoder.apply(encoderInput));
        this.model = tf.model({inputs: encoderInput, outputs: autoencoderOutput});
        
        // Compile model with better learning rate
        this.model.compile({
            optimizer: tf.train.adam(0.002),
            loss: 'binaryCrossentropy'
        });
    }

    getCanvasData() {
        const canvas = document.getElementById('drawingCanvas');
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Scale down to 28x28
        tempCtx.imageSmoothingEnabled = true;
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        
        // Get image data
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = [];
        
        // Convert to grayscale and normalize
        for (let i = 0; i < imageData.data.length; i += 4) {
            // Invert colors (white background to black, black drawing to white)
            const value = (255 - imageData.data[i]) / 255;
            data.push(value);
        }
        
        return data;
    }

    async updatePreview() {
        const data = this.getCanvasData();
        const input = tf.tensor(data).reshape([1, 28, 28, 1]);
        
        // Get latent representation
        const latent = await this.encoder.predict(input);
        const latentValues = await latent.array();
        
        // Get reconstruction
        const output = await this.model.predict(input);
        const outputData = await output.array();
        
        // Calculate loss manually using binary crossentropy formula
        const loss = tf.tidy(() => {
            // Binary crossentropy: -mean(y * log(y_pred) + (1 - y) * log(1 - y_pred))
            const epsilon = 1e-7;
            const y_pred = tf.clipByValue(output, epsilon, 1 - epsilon);
            const bce = tf.mean(
                tf.neg(
                    tf.add(
                        tf.mul(input, tf.log(y_pred)),
                        tf.mul(tf.sub(1, input), tf.log(tf.sub(1, y_pred)))
                    )
                )
            );
            return bce;
        });
        const lossValue = await loss.data();
        loss.dispose();
        
        // Update displays
        this.updateInputDisplay(data);
        this.updateOutputDisplay(outputData[0]);
        this.updateLatentDisplay(latentValues[0]);
        document.getElementById('reconstructionLoss').textContent = lossValue[0].toFixed(4);
        
        // Make prediction if we have training data
        if (this.latentSpaceData.length > 0) {
            // Make detailed prediction
            this.makePrediction(latentValues[0]);
        } else {
            this.clearPrediction();
        }
        
        // Update latent space visualization
        this.updateCurrentPoint(latentValues[0]);
        
        // Cleanup
        input.dispose();
        latent.dispose();
        output.dispose();
        
    }

    updateInputDisplay(data) {
        const canvas = document.getElementById('inputDisplay');
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(28, 28);
        
        for (let i = 0; i < data.length; i++) {
            const value = Math.floor(data[i] * 255);
            imageData.data[i * 4] = value;
            imageData.data[i * 4 + 1] = value;
            imageData.data[i * 4 + 2] = value;
            imageData.data[i * 4 + 3] = 255;
        }
        
        // Draw scaled up
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);
        
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, 60, 60);
        
    }

    updateOutputDisplay(data) {
        const canvas = document.getElementById('outputDisplay');
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(28, 28);
        
        for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
                const idx = i * 28 + j;
                const value = Math.floor(data[i][j][0] * 255);
                imageData.data[idx * 4] = value;
                imageData.data[idx * 4 + 1] = value;
                imageData.data[idx * 4 + 2] = value;
                imageData.data[idx * 4 + 3] = 255;
            }
        }
        
        // Draw scaled up
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);
        
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, 60, 60);
    }

    updateLatentDisplay(latentValues) {
        document.getElementById('encodedDisplay').textContent = 
            `(${latentValues[0].toFixed(1)},${latentValues[1].toFixed(1)})`;
    }

    findNearestDigit(latentPoint) {
        let minDistance = Infinity;
        let nearestDigit = '-';
        
        this.latentSpaceData.forEach(point => {
            const distance = Math.sqrt(
                Math.pow(point.x - latentPoint[0], 2) + 
                Math.pow(point.y - latentPoint[1], 2)
            );
            if (distance < minDistance) {
                minDistance = distance;
                nearestDigit = point.label;
            }
        });
        
        return nearestDigit;
    }

    makePrediction(latentPoint) {
        // Calculate distances to all training points
        const distances = [];
        
        this.latentSpaceData.forEach(point => {
            const distance = Math.sqrt(
                Math.pow(point.x - latentPoint[0], 2) + 
                Math.pow(point.y - latentPoint[1], 2)
            );
            distances.push({
                label: point.label,
                distance: distance
            });
        });
        
        // Sort by distance
        distances.sort((a, b) => a.distance - b.distance);
        
        // Use K-nearest neighbors (K=5)
        const k = Math.min(5, distances.length);
        const nearestK = distances.slice(0, k);
        
        // Count votes for each digit
        const votes = new Array(10).fill(0);
        const weights = new Array(10).fill(0);
        
        nearestK.forEach(neighbor => {
            const weight = 1 / (neighbor.distance + 0.001); // Inverse distance weighting
            votes[neighbor.label]++;
            weights[neighbor.label] += weight;
        });
        
        // Normalize weights to get probabilities
        const totalWeight = weights.reduce((a, b) => a + b, 0);
        const probabilities = weights.map(w => w / totalWeight);
        
        // Find the predicted digit
        let maxProb = 0;
        let predictedDigit = 0;
        probabilities.forEach((prob, digit) => {
            if (prob > maxProb) {
                maxProb = prob;
                predictedDigit = digit;
            }
        });
        
        // Generate explanation
        const reason = this.generatePredictionReason(predictedDigit, nearestK, probabilities);
        
        // Update UI
        document.getElementById('predictedDigit').textContent = predictedDigit;
        document.getElementById('predictionConfidence').textContent = `${(maxProb * 100).toFixed(0)}%`;
        
        
        // Update confidence bars
        this.updateConfidenceBars(probabilities);
    }

    generatePredictionReason(predictedDigit, nearestK, probabilities) {
        const confidence = probabilities[predictedDigit] * 100;
        
        // Count how many of the nearest neighbors are the predicted digit
        const sameDigitCount = nearestK.filter(n => n.label === predictedDigit).length;
        
        if (confidence > 80) {
            return `Strong clustering with ${sameDigitCount}/${nearestK.length} nearest neighbors being ${predictedDigit}s`;
        } else if (confidence > 60) {
            return `Good match - positioned near the ${predictedDigit} cluster region`;
        } else if (confidence > 40) {
            // Find second best digit
            let secondBest = -1;
            let secondBestProb = 0;
            probabilities.forEach((prob, digit) => {
                if (digit !== predictedDigit && prob > secondBestProb) {
                    secondBestProb = prob;
                    secondBest = digit;
                }
            });
            return `Between clusters - could be ${predictedDigit} or ${secondBest}`;
        } else {
            return `Low confidence - far from established clusters or insufficient training data`;
        }
    }

    updateConfidenceBars(probabilities) {
        const container = document.getElementById('confidenceBars');
        container.innerHTML = '';
        
        // Sort digits by probability
        const sortedDigits = probabilities
            .map((prob, digit) => ({ digit, prob }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5); // Show top 5
        
        sortedDigits.forEach(({ digit, prob }) => {
            const percentage = (prob * 100).toFixed(1);
            const bar = document.createElement('div');
            bar.className = 'confidence-bar';
            bar.innerHTML = `
                <div class="confidence-bar-label">${digit}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="confidence-bar-value">${percentage}%</div>
            `;
            container.appendChild(bar);
        });
    }

    clearPrediction() {
        document.getElementById('predictedDigit').textContent = '-';
        document.getElementById('predictionConfidence').textContent = '0%';
        document.getElementById('confidenceBars').innerHTML = '';
    }

    setupLatentSpace() {
        const margin = {top: 10, right: 10, bottom: 30, left: 30};
        const container = document.getElementById('latentSpace');
        const containerWidth = container.clientWidth;
        const containerHeight = 200; // From CSS
        // Use all available space
        const width = containerWidth - margin.left - margin.right;
        const height = containerHeight - margin.top - margin.bottom;
        
        // Create SVG - fill the entire container
        this.svg = d3.select('#latentSpace')
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Add training stage indicator
        this.trainingIndicator = this.svg.append('text')
            .attr('class', 'training-indicator')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .style('fill', '#6366f1')
            .style('opacity', 0);
        
        // Create scales with auto-adjusting domain
        this.xScale = d3.scaleLinear()
            .domain([-5, 5])  // Wider initial range
            .range([0, width]);
        
        this.yScale = d3.scaleLinear()
            .domain([-5, 5])  // Wider initial range
            .range([height, 0]);
        
        // Add axes
        this.svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(this.xScale));
        
        this.svg.append('g')
            .attr('class', 'y-axis')
            .call(d3.axisLeft(this.yScale));
        
        // Add axis labels
        this.svg.append('text')
            .attr('class', 'axis-label')
            .attr('text-anchor', 'middle')
            .attr('x', width / 2)
            .attr('y', height + 35)
            .text('Latent Dimension 1');
        
        this.svg.append('text')
            .attr('class', 'axis-label')
            .attr('text-anchor', 'middle')
            .attr('transform', 'rotate(-90)')
            .attr('y', -25)
            .attr('x', -height / 2)
            .text('Latent Dimension 2');
        
        // Add legend
        const legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${width - 100}, 20)`);
        
        for (let i = 0; i < 10; i++) {
            const legendRow = legend.append('g')
                .attr('transform', `translate(0, ${i * 20})`);
            
            legendRow.append('rect')
                .attr('width', 15)
                .attr('height', 15)
                .attr('fill', this.colorScale(i));
            
            legendRow.append('text')
                .attr('x', 20)
                .attr('y', 12)
                .text(`Digit ${i}`);
        }
    }

    updateLatentSpace() {
        // Auto-adjust scales if we have data
        if (this.latentSpaceData.length > 0) {
            const xExtent = d3.extent(this.latentSpaceData, d => d.x);
            const yExtent = d3.extent(this.latentSpaceData, d => d.y);
            
            // Add padding
            const xPadding = (xExtent[1] - xExtent[0]) * 0.1 || 1;
            const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 1;
            
            this.xScale.domain([xExtent[0] - xPadding, xExtent[1] + xPadding]);
            this.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);
            
            // Update axes
            this.svg.select('.x-axis')
                .transition()
                .duration(500)
                .call(d3.axisBottom(this.xScale));
            
            this.svg.select('.y-axis')
                .transition()
                .duration(500)
                .call(d3.axisLeft(this.yScale));
        }
        
        // Update points
        const circles = this.svg.selectAll('.dot')
            .data(this.latentSpaceData, d => d.id);
        
        circles.enter()
            .append('circle')
            .attr('class', 'dot')
            .attr('r', 5)
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y))
            .attr('fill', d => this.colorScale(d.label))
            .style('opacity', 0)
            .transition()
            .duration(500)
            .style('opacity', 0.7);
        
        circles.transition()
            .duration(500)
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y));
        
        circles.exit()
            .transition()
            .duration(500)
            .style('opacity', 0)
            .remove();
    }

    updateCurrentPoint(latentValues) {
        // Remove existing current point
        this.svg.selectAll('.current-point').remove();
        
        // Add new current point
        this.svg.append('circle')
            .attr('class', 'current-point')
            .attr('r', 8)
            .attr('cx', this.xScale(latentValues[0]))
            .attr('cy', this.yScale(latentValues[1]))
            .style('opacity', 0)
            .transition()
            .duration(300)
            .style('opacity', 1);
    }

    async addToTrainingSet() {
        const data = this.getCanvasData();
        const label = document.getElementById('digitLabel').value;
        
        // Add to training data
        this.trainingData.push({
            data: data,
            label: parseInt(label)
        });
        
        
        // If model has been trained, get latent representation
        if (this.totalEpochs > 0) {
            const input = tf.tensor(data).reshape([1, 28, 28, 1]);
            const latent = await this.encoder.predict(input);
            const latentValues = await latent.array();
            
            this.latentSpaceData.push({
                id: Date.now(),
                x: latentValues[0][0],
                y: latentValues[0][1],
                label: parseInt(label)
            });
            
            this.updateLatentSpace();
            
            input.dispose();
            latent.dispose();
        }
        
        this.updateUI();
        this.clearCanvas();
        
        // Visual feedback
        const addButton = document.getElementById('addToTraining');
        const originalText = addButton.textContent;
        addButton.textContent = '✓';
        addButton.style.background = '#059669';
        setTimeout(() => {
            addButton.textContent = originalText;
            addButton.style.background = '';
        }, 500);
    }

    async trainModel() {
        if (this.trainingData.length === 0) {
            alert('Please add some training samples first!');
            return;
        }
        
        // Warn if already trained
        if (this.totalEpochs > 0) {
            const proceed = confirm(
                `Model already trained for ${this.totalEpochs} epochs.\n\n` +
                `Continue training? This might overfit your specific drawings.\n\n` +
                `Tip: Click 'Reset Model' to start fresh.`
            );
            if (!proceed) return;
        }
        
        // Show progress
        document.querySelector('.training-progress').style.display = 'block';
        
        // Prepare training data
        const xs = [];
        const ys = [];
        
        this.trainingData.forEach(sample => {
            xs.push(sample.data);
            ys.push(sample.data); // Autoencoder uses same data as input and output
        });
        
        const xsTensor = tf.tensor(xs).reshape([xs.length, 28, 28, 1]);
        const ysTensor = tf.tensor(ys).reshape([ys.length, 28, 28, 1]);
        
        // Initialize points for animation
        await this.initializeLatentPoints();
        
        // Train model with visual updates for EVERY epoch
        const epochs = 50;
        
        await this.model.fit(xsTensor, ysTensor, {
            epochs: epochs,
            batchSize: Math.min(16, xs.length),
            shuffle: true,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    const progress = ((epoch + 1) / epochs) * 100;
                    document.getElementById('trainingProgressBar').style.width = `${progress}%`;
                    document.getElementById('progressText').textContent = `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}`;
                    
                    // Update latent space visualization for EVERY epoch
                    await this.updateLatentSpaceAnimation(epoch + 1, epochs);
                    // Delay to make animation visible (150ms per epoch = 7.5 seconds total)
                    await new Promise(resolve => setTimeout(resolve, 150));
                }
            }
        });
        
        this.totalEpochs += epochs;
        
        // Final update with all points
        await this.updateAllLatentPoints();
        
        // Hide progress and training indicator
        setTimeout(() => {
            document.querySelector('.training-progress').style.display = 'none';
            if (this.trainingIndicator) {
                this.trainingIndicator
                    .transition()
                    .duration(500)
                    .style('opacity', 0);
            }
        }, 500);
        
        // Cleanup
        xsTensor.dispose();
        ysTensor.dispose();
        
        this.updateUI();
    }

    async initializeLatentPoints() {
        // Get initial positions of all points
        const initialData = [];
        
        for (let i = 0; i < this.trainingData.length; i++) {
            const sample = this.trainingData[i];
            const input = tf.tensor(sample.data).reshape([1, 28, 28, 1]);
            const latent = this.encoder.predict(input);
            const latentValues = await latent.array();
            
            initialData.push({
                id: `sample_${i}`,
                x: latentValues[0][0],
                y: latentValues[0][1],
                label: sample.label
            });
            
            input.dispose();
            latent.dispose();
        }
        
        this.latentSpaceData = initialData;
        
        // Set initial scale based on initial positions
        const xExtent = d3.extent(initialData, d => d.x);
        const yExtent = d3.extent(initialData, d => d.y);
        
        // Start with a reasonable view that shows initial scatter
        const xPadding = (xExtent[1] - xExtent[0]) * 0.2 || 0.5;
        const yPadding = (yExtent[1] - yExtent[0]) * 0.2 || 0.5;
        
        this.xScale.domain([xExtent[0] - xPadding, xExtent[1] + xPadding]);
        this.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);
        
        // Update axes
        this.svg.select('.x-axis').call(d3.axisBottom(this.xScale));
        this.svg.select('.y-axis').call(d3.axisLeft(this.yScale));
        
        // Draw initial points
        this.updateLatentSpace();
    }

    async updateLatentSpaceAnimation(currentEpoch, totalEpochs) {
        // Show training stage
        if (this.trainingIndicator) {
            const stage = currentEpoch <= 10 ? 'Initial clustering...' :
                         currentEpoch <= 25 ? 'Separating digits...' :
                         currentEpoch <= 40 ? 'Refining positions...' :
                         'Finalizing clusters...';
            
            this.trainingIndicator
                .text(`Epoch ${currentEpoch}/${totalEpochs}: ${stage}`)
                .style('opacity', 1);
        }
        
        // Update latent points with consistent IDs for smooth animation
        const tempLatentData = [];
        
        for (let i = 0; i < this.trainingData.length; i++) {
            const sample = this.trainingData[i];
            const input = tf.tensor(sample.data).reshape([1, 28, 28, 1]);
            const latent = this.encoder.predict(input);
            const latentValues = await latent.array();
            
            tempLatentData.push({
                id: `sample_${i}`, // Consistent ID for each training sample
                x: latentValues[0][0],
                y: latentValues[0][1],
                label: sample.label
            });
            
            input.dispose();
            latent.dispose();
        }
        
        this.latentSpaceData = tempLatentData;
        this.updateLatentSpaceSmoothNoRescale();
    }
    
    updateLatentSpaceSmooth() {
        // Auto-adjust scales if we have data
        if (this.latentSpaceData.length > 0) {
            const xExtent = d3.extent(this.latentSpaceData, d => d.x);
            const yExtent = d3.extent(this.latentSpaceData, d => d.y);
            
            // Add padding
            const xPadding = (xExtent[1] - xExtent[0]) * 0.1 || 1;
            const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 1;
            
            this.xScale.domain([xExtent[0] - xPadding, xExtent[1] + xPadding]);
            this.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);
            
            // Update axes smoothly
            this.svg.select('.x-axis')
                .transition()
                .duration(800)
                .call(d3.axisBottom(this.xScale));
            
            this.svg.select('.y-axis')
                .transition()
                .duration(800)
                .call(d3.axisLeft(this.yScale));
        }
        
        // Update points with smooth transitions
        const circles = this.svg.selectAll('.dot')
            .data(this.latentSpaceData, d => d.id);
        
        // Enter new points
        circles.enter()
            .append('circle')
            .attr('class', 'dot')
            .attr('r', 0)
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y))
            .attr('fill', d => this.colorScale(d.label))
            .style('opacity', 0)
            .transition()
            .duration(800)
            .attr('r', 5)
            .style('opacity', 0.7);
        
        // Update existing points with smooth movement
        circles
            .transition()
            .duration(800)
            .ease(d3.easeCubicInOut)
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y))
            .attr('fill', d => this.colorScale(d.label));
        
        // Remove old points
        circles.exit()
            .transition()
            .duration(400)
            .attr('r', 0)
            .style('opacity', 0)
            .remove();
    }
    
    updateLatentSpaceSmoothNoRescale() {
        // Calculate new bounds for the data
        if (this.latentSpaceData.length > 0) {
            const xExtent = d3.extent(this.latentSpaceData, d => d.x);
            const yExtent = d3.extent(this.latentSpaceData, d => d.y);
            
            // Add padding
            const xPadding = (xExtent[1] - xExtent[0]) * 0.2 || 0.5;
            const yPadding = (yExtent[1] - yExtent[0]) * 0.2 || 0.5;
            
            // Update scales smoothly
            this.xScale.domain([xExtent[0] - xPadding, xExtent[1] + xPadding]);
            this.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);
            
            // Animate axes
            this.svg.select('.x-axis')
                .transition()
                .duration(100)
                .ease(d3.easeLinear)
                .call(d3.axisBottom(this.xScale));
            
            this.svg.select('.y-axis')
                .transition()
                .duration(100)
                .ease(d3.easeLinear)
                .call(d3.axisLeft(this.yScale));
        }
        
        // Update points with smooth transitions
        const circles = this.svg.selectAll('.dot')
            .data(this.latentSpaceData, d => d.id);
        
        // Enter new points
        circles.enter()
            .append('circle')
            .attr('class', 'dot')
            .attr('r', 5)
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y))
            .attr('fill', d => this.colorScale(d.label))
            .style('opacity', 0.7);
        
        // Update existing points with smooth movement
        circles
            .transition()
            .duration(100) // Quick transitions for each epoch
            .ease(d3.easeLinear) // Linear for consistent speed
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y));
        
        // Remove old points (shouldn't happen)
        circles.exit().remove();
    }

    async updateAllLatentPoints() {
        this.latentSpaceData = [];
        
        for (const sample of this.trainingData) {
            const input = tf.tensor(sample.data).reshape([1, 28, 28, 1]);
            const latent = await this.encoder.predict(input);
            const latentValues = await latent.array();
            
            this.latentSpaceData.push({
                id: Date.now() + Math.random(),
                x: latentValues[0][0],
                y: latentValues[0][1],
                label: sample.label
            });
            
            input.dispose();
            latent.dispose();
        }
        
        this.updateLatentSpace();
    }

    resetModel() {
        // Recreate model
        this.createModel();
        this.totalEpochs = 0;
        this.latentSpaceData = [];
        this.updateLatentSpace();
        this.updateUI();
        this.clearCanvas();
    }

    clearTrainingData() {
        this.trainingData = [];
        this.latentSpaceData = [];
        this.updateLatentSpace();
        this.updateUI();
    }

    setupEventListeners() {
        document.getElementById('clearCanvas').addEventListener('click', () => this.clearCanvas());
        document.getElementById('addToTraining').addEventListener('click', () => this.addToTrainingSet());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
    }

    setupMobileTabs() {
        // No tabs in new mobile design
    }

    resizeLatentSpace() {
        // Update SVG dimensions for mobile
        const container = document.getElementById('latentSpace');
        const width = container.clientWidth;
        
        if (this.svg && width > 0) {
            // Update scales and redraw
            const margin = {top: 20, right: 20, bottom: 40, left: 40};
            const newWidth = width - margin.left - margin.right;
            
            this.xScale.range([0, newWidth]);
            
            // Update axes positions
            this.svg.select('.x-axis')
                .attr('transform', `translate(0,${this.yScale.range()[0]})`);
            
            // Redraw points
            this.updateLatentSpace();
        }
    }

    updateUI() {
        document.getElementById('sampleCount').textContent = this.trainingData.length;
        const status = this.totalEpochs === 0 ? 'Untrained' : 
                       this.totalEpochs <= 50 ? 'Trained' : 
                       this.totalEpochs <= 100 ? 'Well-trained' : 'Over-trained';
        document.getElementById('trainStatus').textContent = status;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    const demo = new AutoencoderDemo();
});