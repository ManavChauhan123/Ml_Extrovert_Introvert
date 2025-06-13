class MLDashboard {
    constructor() {
        this.apiUrl = ''; // Base FastAPI URL
        this.modelInfo = null;
        console.log("🚀 MLDashboard Initialized");
        this.init();
    }

    async init() {
        await this.loadModelInfo();
        this.setupPredictionForm();
        this.setupEventListeners();
        this.loadClassificationHistory();
    }

    async loadModelInfo() {
        try {
            const response = await fetch(`${this.apiUrl}/model-info`);
            this.modelInfo = await response.json();
            this.displayModelInfo();
        } catch (error) {
            console.error('Error loading model info:', error);
            this.displayError('Failed to load model information.');
        }
    }

    setupPredictionForm() {
        const container = document.getElementById('feature-inputs');
        if (!this.modelInfo || !this.modelInfo.feature_names) return;

        container.innerHTML = ''; // Reset previous fields

        this.modelInfo.feature_names.forEach(feature => {
            const div = document.createElement('div');
            div.className = 'col-md-4 mb-3';
            div.innerHTML = `
                <label for="${feature}" class="form-label">${this.formatTitle(feature)}</label>
                <input type="number" class="form-control" id="${feature}" name="${feature}" step="any" required>
            `;
            container.appendChild(div);
        });
    }

    setupEventListeners() {
        const form = document.getElementById('classificationForm');
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

        const container = document.getElementById('classification-result');
        container.innerHTML = `
            <div class="text-center my-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;

        try {
            const response = await fetch(`${this.apiUrl}/classify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features })
            });

            const result = await response.json();
            this.displayPredictionResult(result);
            this.loadClassificationHistory();
        } catch (error) {
            console.error('Prediction error:', error);
            this.displayError('Error making prediction.');
        }
    }

    displayPredictionResult(result) {
        const container = document.getElementById('classification-result');

        const probList = Object.entries(result.prediction_probabilities)
            .map(([label, prob]) => `<li><strong>${label}</strong>: ${(prob * 100).toFixed(2)}%</li>`)
            .join("");

        container.innerHTML = `
            <div class="alert alert-success shadow-sm rounded">
                <h5><i class="fas fa-check-circle text-success"></i> Classification Result</h5>
                <p><strong>Predicted Class:</strong> ${result.predicted_class}</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                <p><strong>Model Used:</strong> ${result.model_used}</p>
                <p class="mb-1"><strong>Prediction Probabilities:</strong></p>
                <ul>${probList}</ul>
            </div>
        `;
    }

    displayModelInfo() {
        const container = document.getElementById('model-details');
        if (!container || !this.modelInfo) return;

        container.innerHTML = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <h5 class="text-primary">Best Model: ${this.modelInfo.model_name}</h5>
                    <h6 class="text-muted">Hyperparameters:</h6>
                    <pre class="bg-light p-2 rounded">${JSON.stringify(this.modelInfo.best_params, null, 2)}</pre>
                </div>
                <div class="col-md-6 mb-3">
                    <h6 class="text-muted">Classification Report:</h6>
                    <pre class="bg-light p-2 rounded">${this.modelInfo.classification_report}</pre>
                </div>
            </div>
        `;
    }

    showEDAImage(imageName) {
        const edaContent = document.getElementById('eda-content');
        edaContent.innerHTML = `
            <div class="text-center">
                <h5 class="mb-3">${this.formatTitle(imageName)}</h5>
                <img src="/eda/${imageName}.png"
                     class="img-fluid rounded shadow"
                     alt="${imageName}"
                     onerror="this.src='/static/images/not-found.png';" />
            </div>
        `;
    }

    async loadClassificationHistory(limit = 50) {
        const container = document.getElementById('classification-history-content');
        container.innerHTML = `
            <div class="text-center my-3">
                <div class="spinner-border text-secondary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;

        try {
            const response = await fetch(`${this.apiUrl}/classification-history?limit=${limit}`);
            const history = await response.json();
            this.displayClassificationHistory(history);
        } catch (error) {
            console.error('History load error:', error);
            this.displayHistoryError('Could not load classification history.');
        }
    }

    displayClassificationHistory(history) {
        const container = document.getElementById('classification-history-content');
        if (!container) return;

        if (history.length === 0) {
            container.innerHTML = `<p class="text-muted">No classification history available.</p>`;
            return;
        }

        const rows = history.map(record => `
            <tr>
                <td>${record.id}</td>
                <td>${record.predicted_class}</td>
                <td>${(record.confidence * 100).toFixed(2)}%</td>
                <td>${record.model_used}</td>
                <td>${new Date(record.timestamp).toLocaleString()}</td>
            </tr>
        `).join("");

        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-striped table-hover align-middle">
                    <thead class="table-dark">
                        <tr>
                            <th>ID</th>
                            <th>Predicted Class</th>
                            <th>Confidence</th>
                            <th>Model Used</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    formatTitle(name) {
        return name.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }

    displayError(message) {
        const container = document.getElementById('classification-result');
        container.innerHTML = `
            <div class="alert alert-danger shadow-sm rounded">
                <i class="fas fa-exclamation-triangle"></i> ${message}
            </div>
        `;
    }

    displayHistoryError(message) {
        const container = document.getElementById('classification-history-content');
        container.innerHTML = `
            <div class="alert alert-danger shadow-sm rounded">
                <i class="fas fa-exclamation-circle"></i> ${message}
            </div>
        `;
    }
}

// Auto-init after DOM loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new MLDashboard();
});

// Expose for global EDA buttons
function showEDAImage(imageName) {
    window.dashboard.showEDAImage(imageName);
}
