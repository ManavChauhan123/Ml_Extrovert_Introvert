class MLDashboard {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.modelInfo = null;
        this.init();
    }

    async init() {
        await this.loadModelInfo();
        this.setupPredictionForm();
        this.setupEventListeners();
    }

    async loadModelInfo() {
        try {
            const response = await fetch(`${this.apiUrl}/model-info`);
            this.modelInfo = await response.json();
            this.displayModelInfo();
        } catch (error) {
            console.error('Error loading model info:', error);
        }
    }

    setupPredictionForm() {
        const container = document.getElementById('feature-inputs');
        if (!this.modelInfo || !this.modelInfo.feature_names) return;

        this.modelInfo.feature_names.forEach(feature => {
            const div = document.createElement('div');
            div.className = 'mb-3';
            div.innerHTML = `
                <label for="${feature}" class="form-label">${feature}</label>
                <input type="number" class="form-control" id="${feature}" 
                       name="${feature}" step="any" required>
            `;
            container.appendChild(div);
        });
    }

    setupEventListeners() {
        const form = document.getElementById('predictionForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handlePrediction(e));
        }
    }

    async handlePrediction(event) {
        event.preventDefault();
        const formData = new FormData(event.target);
        const features = {};

        for (let [key, value] of formData.entries()) {
            features[key] = parseFloat(value);
        }

        try {
            const response = await fetch(`${this.apiUrl}/classify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features })
            });

            const result = await response.json();
            this.displayPredictionResult(result);
        } catch (error) {
            console.error('Prediction error:', error);
            this.displayError('Error making prediction');
        }
    }

    displayPredictionResult(result) {
        const container = document.getElementById('prediction-result');
        container.innerHTML = `
            <div class="alert alert-success">
                <h5><i class="fas fa-check-circle"></i> Classification Result</h5>
                <p><strong>Predicted Class:</strong> ${result.predicted_class}</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
            </div>
        `;
    }

    displayModelInfo() {
        const container = document.getElementById('model-details');
        if (!this.modelInfo) return;

        container.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h5>Best Model: ${this.modelInfo.model_name}</h5>
                    <h6>Hyperparameters:</h6>
                    <pre>${JSON.stringify(this.modelInfo.best_params, null, 2)}</pre>
                </div>
                <div class="col-md-6">
                    <h6>Classification Report:</h6>
                    <pre>${this.modelInfo.classification_report}</pre>
                </div>
            </div>
        `;
    }

    async showConfusionMatrix() {
        const edaContent = document.getElementById('eda-content');
        edaContent.innerHTML = `
            <div class="text-center">
                <h5>Confusion Matrix</h5>
                <img src="${this.apiUrl}/static/eda/confusion-matrix.png" 
                     class="img-fluid" alt="Confusion Matrix">
            </div>
        `;
    }

    displayError(message) {
        const container = document.getElementById('prediction-result');
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> ${message}
            </div>
        `;
    }
}

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new MLDashboard();
});

// Global functions for EDA buttons
function showConfusionMatrix() {
    dashboard.showConfusionMatrix();
}
