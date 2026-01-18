document.addEventListener('DOMContentLoaded', function() {
    const statusText = document.getElementById('statusText');
    const loader = document.getElementById('loader');
    const resultBox = document.getElementById('resultBox');
    const detailsBox = document.getElementById('detailsBox');
    
    const vidScoreEl = document.getElementById('vidScore');
    const audScoreEl = document.getElementById('audScore');
    const urlScoreEl = document.getElementById('urlScore');

    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        let currentUrl = tabs[0].url;
        
        if (!currentUrl.includes("youtube.com") && !currentUrl.includes("youtu.be")) {
            loader.style.display = "none";
            statusText.innerText = "Te rog intră pe un video YouTube!";
            statusText.style.color = "orange";
            return;
        }

        statusText.innerText = "Se analizează Video, Audio & URL...";
        analyzeUrl(currentUrl);
    });

    async function analyzeUrl(url) {
        try {
            const response = await fetch('http://127.0.0.1:5000/detect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();

            if (loader) loader.style.display = "none";
            if (statusText) statusText.style.display = "none";
            
            resultBox.style.display = "block";
            detailsBox.style.display = "block";

            let score = data.final_score;
            let verdict = data.result;

            if (verdict === "FAKE") {
                resultBox.style.backgroundColor = "#ffebee";
                resultBox.style.color = "#c62828";
                resultBox.innerText = `🚨 DEEPFAKE (${score}%)`;
            } else if (verdict === "REAL") {
                resultBox.style.backgroundColor = "#e8f5e9";
                resultBox.style.color = "#2e7d32";
                resultBox.innerText = `✅ REAL (${(100-score).toFixed(1)}%)`;
            } else {
                resultBox.style.backgroundColor = "#fff3e0";
                resultBox.style.color = "#ef6c00";
                resultBox.innerText = `⚠️ INCERT (${score}%)`;
            }

            // Extragem scorurile (punem 0 daca lipsesc)
            let vScore = 0, aScore = 0, uScore = 0;
            if (data.details) {
                vScore = data.details.video_score || 0;
                aScore = data.details.audio_score || 0;
                uScore = data.details.url_score || 0;
            }

            vidScoreEl.innerText = `${vScore}% Risc`;
            audScoreEl.innerText = `${aScore}% Risc`;
            urlScoreEl.innerText = `${uScore}% Risc`;
            
            // Colorare
            vidScoreEl.style.color = vScore > 50 ? "#c62828" : "#2e7d32";
            audScoreEl.style.color = aScore > 50 ? "#c62828" : "#2e7d32";
            urlScoreEl.style.color = uScore > 50 ? "#c62828" : "#2e7d32";

        } catch (error) {
            console.error(error);
            if (loader) loader.style.display = "none";
            if (statusText) {
                statusText.style.display = "block";
                statusText.innerText = "Eroare server. Verifica app.py";
                statusText.style.color = "red";
            }
        }
    }
});