document.addEventListener("DOMContentLoaded", () => {
    const queryInput = document.getElementById("query-input");
    const kSlider = document.getElementById("k-slider");
    const kValue = document.getElementById("k-value");
    const searchBtn = document.getElementById("search-btn");
    const consoleLogs = document.getElementById("console-logs");
    
    // Metrics
    const metricProvider = document.getElementById("metric-provider");
    const metricModel = document.getElementById("metric-model");
    const metricLatency = document.getElementById("metric-latency");
    const metricSimilarity = document.getElementById("metric-similarity");
    const similarityProgress = document.getElementById("similarity-progress");
    
    // Evaluation Triad
    const scoreContext = document.getElementById("score-context");
    const progressContext = document.getElementById("progress-context");
    const descContext = document.getElementById("desc-context");
    
    const scoreFaith = document.getElementById("score-faith");
    const progressFaith = document.getElementById("progress-faith");
    const descFaith = document.getElementById("desc-faith");
    
    const scoreAnswer = document.getElementById("score-answer");
    const progressAnswer = document.getElementById("progress-answer");
    const descAnswer = document.getElementById("desc-answer");
    
    const responseBox = document.getElementById("response-box");
    const referencesContainer = document.getElementById("references-container");

    // Update k slider label
    kSlider.addEventListener("input", (e) => {
        kValue.textContent = e.target.value;
    });

    // Console Logging helper
    function logToConsole(message, type = "info") {
        const timestamp = new Date().toLocaleTimeString();
        const line = document.createElement("span");
        line.className = `log-line ${type}`;
        line.innerHTML = `<span style="color: var(--text-muted)">[${timestamp}]</span> ${message}`;
        consoleLogs.appendChild(line);
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    }

    // Clear UI for loading state
    function resetUIForLoading() {
        searchBtn.disabled = true;
        searchBtn.querySelector("span").textContent = "Ingesting & Analyzing...";
        
        responseBox.innerHTML = `<div class="loading-spinner-container">
            <span class="text-muted">Analyzing clinical guidelines and generating response...</span>
        </div>`;
        
        // Reset metrics
        metricProvider.textContent = "--";
        metricModel.textContent = "Processing...";
        metricLatency.textContent = "0.00";
        metricSimilarity.textContent = "--";
        similarityProgress.style.width = "0%";
        
        // Reset evaluation bars
        scoreContext.textContent = "--%";
        progressContext.style.width = "0%";
        descContext.textContent = "Evaluating context relevance...";
        
        scoreFaith.textContent = "--%";
        progressFaith.style.width = "0%";
        descFaith.textContent = "Checking for hallucinations...";
        
        scoreAnswer.textContent = "--%";
        progressAnswer.style.width = "0%";
        descAnswer.textContent = "Assessing response completeness...";
        
        referencesContainer.innerHTML = `<span class="text-muted">Loading reference documents...</span>`;
    }

    // Trigger Search & Generation
    async function executeQuery() {
        const query = queryInput.value.trim();
        const k = kSlider.value;
        
        if (!query) {
            alert("Please enter a question first.");
            return;
        }

        resetUIForLoading();
        consoleLogs.innerHTML = ""; // Clear console
        logToConsole(`Initializing RAG query: "${query.substring(0, 50)}..."`);
        logToConsole(`Retrieving top k=${k} relevant chunks from Vector Database...`);
        
        try {
            const start = performance.now();
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query, k })
            });
            
            const duration = ((performance.now() - start) / 1000).toFixed(2);
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `HTTP ${response.status} Error`);
            }
            
            const data = await response.json();
            
            logToConsole(`Successfully received response in ${duration}s!`, "success");
            
            // Print Fallback Chain Logs
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(log => {
                    if (log.status === "success") {
                        logToConsole(`Provider "${log.provider}" (${log.model}) SUCCESS in ${log.latency}s`, "success");
                    } else if (log.status === "failed") {
                        logToConsole(`Provider "${log.provider}" FAILED: ${log.error}`, "error");
                    } else if (log.status === "skipped") {
                        logToConsole(`Provider "${log.provider}" SKIPPED: ${log.error}`, "warn");
                    } else {
                        logToConsole(`Provider "${log.provider}": ${log.status}`);
                    }
                });
            }
            
            // Update Metrics Cards
            metricProvider.textContent = data.provider;
            metricModel.textContent = data.model;
            metricLatency.textContent = data.latency;
            
            const simPercentage = Math.round(data.avg_similarity * 100);
            metricSimilarity.textContent = `${simPercentage}%`;
            similarityProgress.style.width = `${simPercentage}%`;
            
            // Update Answer
            responseBox.innerHTML = data.answer;
            
            // Update Triad Evaluation UI
            if (data.evaluation) {
                const ctx = data.evaluation.context_relevance;
                const faith = data.evaluation.faithfulness;
                const ans = data.evaluation.answer_relevance;
                
                if (ctx) {
                    scoreContext.textContent = `${ctx.score}%`;
                    progressContext.style.width = `${ctx.score}%`;
                    descContext.innerHTML = `<strong>Explanation:</strong> ${ctx.explanation}`;
                    logToConsole(`Evaluation Context Relevance: ${ctx.score}%`, ctx.score > 70 ? "success" : "warn");
                }
                if (faith) {
                    scoreFaith.textContent = `${faith.score}%`;
                    progressFaith.style.width = `${faith.score}%`;
                    descFaith.innerHTML = `<strong>Explanation:</strong> ${faith.explanation}`;
                    logToConsole(`Evaluation Groundedness: ${faith.score}%`, faith.score > 70 ? "success" : "warn");
                }
                if (ans) {
                    scoreAnswer.textContent = `${ans.score}%`;
                    progressAnswer.style.width = `${ans.score}%`;
                    descAnswer.innerHTML = `<strong>Explanation:</strong> ${ans.explanation}`;
                    logToConsole(`Evaluation Answer Relevance: ${ans.score}%`, ans.score > 70 ? "success" : "warn");
                }
            }
            
            // Update Retrieved References Cards
            if (data.references && data.references.length > 0) {
                referencesContainer.innerHTML = "";
                data.references.forEach((ref, idx) => {
                    const card = document.createElement("div");
                    card.className = "ref-card animate-hover";
                    
                    const meta = ref.metadata;
                    const fileName = meta.file_name || "Unknown Document";
                    const pageInfo = meta.page ? `Page ${meta.page}` : (meta.row ? `Sheet '${meta.sheet}' Row ${meta.row}` : "Doc chunk");
                    const similarityScore = Math.round(ref.score * 100);
                    
                    card.innerHTML = `
                        <div class="ref-meta">
                            <span class="ref-source">📄 [${idx + 1}] ${fileName} (${pageInfo})</span>
                            <span class="ref-score-badge">Similarity: ${similarityScore}%</span>
                        </div>
                        <p class="ref-text">${ref.text}</p>
                    `;
                    referencesContainer.appendChild(card);
                });
            } else {
                referencesContainer.innerHTML = `<span class="text-muted">No reference documents were retrieved.</span>`;
            }
            
        } catch (error) {
            logToConsole(`CRITICAL ERROR: ${error.message}`, "error");
            responseBox.innerHTML = `<span class="text-muted" style="color: #ef4444 !important;">
                <strong>Error processing request:</strong><br>${error.message}
            </span>`;
            referencesContainer.innerHTML = `<span class="text-muted">Retrieval failed.</span>`;
        } finally {
            searchBtn.disabled = false;
            searchBtn.querySelector("span").textContent = "Search & Generate";
        }
    }

    searchBtn.addEventListener("click", executeQuery);
    
    // Allow keyboard shortcut Ctrl+Enter to trigger search
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && e.ctrlKey) {
            e.preventDefault();
            executeQuery();
        }
    });

    // --- Knowledge Base Upload Handler ---
    const fileUploadInput = document.getElementById("file-upload");
    const selectedFileName = document.getElementById("selected-file-name");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadStatusBox = document.getElementById("upload-status-box");
    const uploadStatusText = document.getElementById("upload-status-text");
    const uploadStatusDot = uploadStatusBox.querySelector(".status-dot");
    const uploadProgressContainer = uploadStatusBox.querySelector(".upload-progress-container");
    const uploadProgressBar = document.getElementById("upload-progress");

    // Update label when file is chosen
    fileUploadInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            setUploadStatus("Selected file ready to index.", "idle");
        } else {
            selectedFileName.textContent = "No file selected";
            setUploadStatus("Ready to upload", "idle");
        }
    });

    // Status UI helper
    function setUploadStatus(text, status) {
        uploadStatusText.textContent = text;
        uploadStatusDot.className = "status-dot";
        
        if (status === "idle") {
            uploadStatusDot.classList.add("status-idle");
            uploadProgressContainer.style.display = "none";
        } else if (status === "running") {
            uploadStatusDot.classList.add("status-running");
            uploadProgressContainer.style.display = "block";
        } else if (status === "success") {
            uploadStatusDot.classList.add("status-success");
            uploadProgressContainer.style.display = "none";
        } else if (status === "error") {
            uploadStatusDot.classList.add("status-error");
            uploadProgressContainer.style.display = "none";
        }
    }

    // Upload & Index trigger
    uploadBtn.addEventListener("click", async () => {
        const file = fileUploadInput.files[0];
        if (!file) {
            alert("Please select a file to index first.");
            return;
        }

        uploadBtn.disabled = true;
        uploadBtn.querySelector("span").textContent = "Indexing...";
        setUploadStatus(`Reading local file: ${file.name}...`, "running");
        uploadProgressBar.style.width = "20%";

        const reader = new FileReader();
        
        reader.onload = async (event) => {
            uploadProgressBar.style.width = "40%";
            setUploadStatus("Uploading content payload...", "running");
            const fileData = event.target.result;
            
            try {
                uploadProgressBar.style.width = "60%";
                setUploadStatus("Server indexing: parsing & chunking...", "running");
                
                const response = await fetch("/api/upload", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        filename: file.name,
                        file_data: fileData
                    })
                });
                
                uploadProgressBar.style.width = "85%";
                
                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || `HTTP ${response.status} Error`);
                }
                
                const data = await response.json();
                uploadProgressBar.style.width = "100%";
                
                setUploadStatus(`Success: Added ${data.chunks_added} chunks!`, "success");
                logToConsole(`Uploaded and indexed successfully: "${file.name}" added ${data.chunks_added} chunks.`, "success");
                
                // Reset file selection
                fileUploadInput.value = "";
                selectedFileName.textContent = "No file selected";
                
            } catch (error) {
                setUploadStatus(`Failed: ${error.message}`, "error");
                logToConsole(`Failed indexing "${file.name}": ${error.message}`, "error");
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.querySelector("span").textContent = "Index Document";
            }
        };

        reader.onerror = () => {
            setUploadStatus("Failed reading local file content.", "error");
            uploadBtn.disabled = false;
            uploadBtn.querySelector("span").textContent = "Index Document";
        };

        reader.readAsDataURL(file);
    });
});
