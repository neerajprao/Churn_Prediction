document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const loadingEl = document.getElementById('loading');
    const resultsEl = document.getElementById('results');
    
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading, hide results
        loadingEl.classList.remove('hidden');
        resultsEl.classList.add('hidden');
        
        const formData = {
            creditScore: document.getElementById('creditScore').value,
            age: document.getElementById('age').value,
            tenure: document.getElementById('tenure').value,
            balance: document.getElementById('balance').value,
            numProducts: document.getElementById('numProducts').value,
            hasCard: document.getElementById('hasCard').value,
            isActive: document.getElementById('isActive').value,
            salary: document.getElementById('salary').value,
            geography: document.getElementById('geography').value,
            gender: document.getElementById('gender').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Unknown error occurred');
            }
            
            // Hide loading
            loadingEl.classList.add('hidden');
            
            // Update predictions
            updatePredictions(result.predictions);
            
            // Show results
            resultsEl.classList.remove('hidden');
            
            // Scroll to results
            resultsEl.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error:', error);
            loadingEl.classList.add('hidden');
            alert('Failed to get predictions: ' + error.message);
        }
    });
    
    function updatePredictions(predictions) {
        // Update individual model predictions
        if (predictions.node !== undefined) {
            document.getElementById('node-pred').textContent = formatPrediction(predictions.node);
        } else {
            document.getElementById('node-pred').textContent = "Not available";
        }
        
        if (predictions.saint !== undefined) {
            document.getElementById('saint-pred').textContent = formatPrediction(predictions.saint);
        } else {
            document.getElementById('saint-pred').textContent = "Not available";
        }
        
        if (predictions.tabtransformer !== undefined) {
            document.getElementById('tabtransformer-pred').textContent = formatPrediction(predictions.tabtransformer);
        } else {
            document.getElementById('tabtransformer-pred').textContent = "Not available";
        }

        // Add TabNet prediction - make sure it's properly updated
        if (predictions.tabnet !== undefined) {
            document.getElementById('tabnet-pred').textContent = formatPrediction(predictions.tabnet);
        } else {
            document.getElementById('tabnet-pred').textContent = "Not available";
        }
        
        if (predictions.catboost !== undefined) {
            document.getElementById('catboost-pred').textContent = formatPrediction(predictions.catboost);
        } else {
            document.getElementById('catboost-pred').textContent = "Not available";
        }
        
        if (predictions.lgbm !== undefined) {
            document.getElementById('lgbm-pred').textContent = formatPrediction(predictions.lgbm);
        } else {
            document.getElementById('lgbm-pred').textContent = "Not available";
        }
        
        if (predictions.rf !== undefined) {
            document.getElementById('rf-pred').textContent = formatPrediction(predictions.rf);
        } else {
            document.getElementById('rf-pred').textContent = "Not available";
        }
        
        if (predictions.xgb !== undefined) {
            document.getElementById('xgb-pred').textContent = formatPrediction(predictions.xgb);
        } else {
            document.getElementById('xgb-pred').textContent = "Not available";
        }
        
        // Calculate consensus
        const consensusEl = document.getElementById('consensus-value');
        const consensusExplanationEl = document.getElementById('consensus-explanation');
        
        // Get valid predictions (numerical only)
        const validPredictions = Object.values(predictions).filter(p => typeof p === 'number');
        
        if (validPredictions.length > 0) {
            // Count predictions
            let stayCount = 0;
            let churnCount = 0;
            
            validPredictions.forEach(pred => {
                if (pred === 0) stayCount++;
                else if (pred === 1) churnCount++;
            });
            
            // Calculate consensus
            const total = stayCount + churnCount;
            const stayPercentage = Math.round((stayCount / total) * 100);
            const churnPercentage = Math.round((churnCount / total) * 100);
            
            // Display consensus
            if (stayCount > churnCount) {
                consensusEl.className = 'stay';
                consensusEl.textContent = 'STAY (' + stayPercentage + '%)';
                consensusExplanationEl.textContent = `${stayCount} out of ${total} models predict the customer will stay with the bank. ${churnCount} models predict the customer will leave.`;
            } else if (churnCount > stayCount) {
                consensusEl.className = 'churn';
                consensusEl.textContent = 'CHURN (' + churnPercentage + '%)';
                consensusExplanationEl.textContent = `${churnCount} out of ${total} models predict the customer will leave the bank. ${stayCount} models predict the customer will stay.`;
            } else {
                consensusEl.className = '';
                consensusEl.textContent = 'UNDECIDED (50%)';
                consensusExplanationEl.textContent = `The models are evenly split: ${stayCount} predict stay and ${churnCount} predict churn.`;
            }
        } else {
            consensusEl.textContent = 'No valid predictions available';
            consensusExplanationEl.textContent = '';
        }
    }
    
    function formatPrediction(prediction) {
        if (typeof prediction === 'number') {
            if (prediction === 0) {
                return 'Stay (0)';
            } else if (prediction === 1) {
                return 'Churn (1)';
            } else {
                return prediction.toString();
            }
        } else {
            return prediction;
        }
    }
    
    // Set default values for form fields
    function setDefaultValues() {
        document.getElementById('creditScore').value = 650;
        document.getElementById('age').value = 35;
        document.getElementById('tenure').value = 5;
        document.getElementById('balance').value = 75000;
        document.getElementById('numProducts').value = 2;
        document.getElementById('hasCard').value = 1;
        document.getElementById('isActive').value = 1;
        document.getElementById('salary').value = 65000;
        document.getElementById('geography').value = 'France';
        document.getElementById('gender').value = 'Male';
    }
    
    // Set default values on page load
    //setDefaultValues();
});

// For the models page
function toggleModel(element) {
    const content = element.nextElementSibling;
    content.classList.toggle('active');
    const arrow = element.querySelector('.arrow');
    arrow.style.transform = content.classList.contains('active') ? 'rotate(180deg)' : '';
}