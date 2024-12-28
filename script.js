// URL of the FastAPI endpoint
const apiUrl = 'http://127.0.0.1:8000/predict';

// Function to fetch prediction
async function getPrediction() {
    try {
        const response = await fetch(apiUrl);
        const data = await response.json();

        // Update the prediction and confidence on the page
        document.getElementById('prediction').textContent = `Prediction: ${data.prediction}`;
        document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('prediction').textContent = 'Prediction: Error';
        document.getElementById('confidence').textContent = 'Confidence: -';
    }
}
