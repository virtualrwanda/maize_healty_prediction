const apiUrl = '/predict';
const configureUrl = '/configure_camera';

// Function to configure the camera
async function configureCamera(event) {
    event.preventDefault();
    
    const ip = document.getElementById('ip').value;
    const port = document.getElementById('port').value;
    
    try {
        const response = await fetch(configureUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'ip': ip,
                'port': port
            })
        });
        
        const data = await response.json();
        alert(data.message);
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to configure the camera.');
    }
}

// Function to get prediction
async function getPrediction() {
    try {
        const response = await fetch(apiUrl);
        const data = await response.json();

        document.getElementById('prediction').textContent = `Prediction: ${data.prediction}`;
        document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
        
        if (data.image) {
            const imageElement = document.getElementById('capturedImage');
            imageElement.src = `data:image/jpeg;base64,${data.image}`;
            imageElement.style.display = 'block';
        } else {
            document.getElementById('capturedImage').style.display = 'none';
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('prediction').textContent = 'Prediction: Error';
        document.getElementById('confidence').textContent = 'Confidence: -';
        document.getElementById('capturedImage').style.display = 'none';
    }
}
