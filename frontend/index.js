document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const queryInput = document.getElementById("query-input");
    const kSlider = document.getElementById("k-slider");
    const kValue = document.getElementById("k-value");
    const searchBtn = document.getElementById("search-btn");
    const consoleLogs = document.getElementById("console-logs");

    // Sidebar elements
    const chatMessages = document.getElementById("chat-messages");
    const chatSessionsList = document.getElementById("chat-sessions-list");
    const newChatBtn = document.getElementById("new-chat-btn");

    // Right Sidebar / Metrics Panel
    const metricProvider = document.getElementById("metric-provider");
    const metricModel = document.getElementById("metric-model");
    const metricLatency = document.getElementById("metric-latency");
    const metricLatencyDetails = document.getElementById("metric-latency-details");
    const metricSimilarity = document.getElementById("metric-similarity");
    const similarityProgress = document.getElementById("similarity-progress");

    // Evaluation Triad Cards
    const scoreContext = document.getElementById("score-context");
    const progressContext = document.getElementById("progress-context");
    const descContext = document.getElementById("desc-context");

    const scoreFaith = document.getElementById("score-faith");
    const progressFaith = document.getElementById("progress-faith");
    const descFaith = document.getElementById("desc-faith");

    const scoreAnswer = document.getElementById("score-answer");
    const progressAnswer = document.getElementById("progress-answer");
    const descAnswer = document.getElementById("desc-answer");

    const referencesContainer = document.getElementById("references-container");

    // State Variables
    let sessions = [];
    let activeSessionId = null;

    // --- localStore Chat History Logic ---

    // Load sessions from localStorage
    function loadSessionsFromStorage() {
        try {
            const data = localStorage.getItem("arthrilens_sessions");
            if (data) {
                sessions = JSON.parse(data);
            }
        } catch (e) {
            console.error("Failed loading sessions from localStorage", e);
        }

        if (!Array.isArray(sessions)) {
            sessions = [];
        }

        if (sessions.length === 0) {
            createNewSession();
        } else {
            // Filter out null or invalid session objects
            sessions = sessions.filter(s => s && typeof s === 'object' && !Array.isArray(s));
            // Validate all loaded sessions have a messages array
            sessions.forEach(s => {
                if (!Array.isArray(s.messages)) {
                    s.messages = [];
                }
            });
            if (sessions.length === 0) {
                createNewSession();
            } else {
                activeSessionId = sessions[0].id;
                renderSessionsList();
                renderActiveMessages();
            }
        }
    }

    // Save sessions to localStorage
    function saveSessionsToStorage() {
        try {
            localStorage.setItem("arthrilens_sessions", JSON.stringify(sessions));
        } catch (e) {
            console.error("Failed saving sessions to localStorage", e);
        }
    }

    // Create a new session
    function createNewSession() {
        const newSession = {
            id: "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9),
            title: "New Conversation",
            messages: []
        };
        sessions.unshift(newSession);
        activeSessionId = newSession.id;
        saveSessionsToStorage();
        renderSessionsList();
        renderActiveMessages();
        logToConsole("New conversation thread initialized.", "info");
    }

    // Delete a session
    function deleteSession(sessionId, event) {
        if (event) event.stopPropagation();

        sessions = sessions.filter(s => s.id !== sessionId);

        if (sessions.length === 0) {
            createNewSession();
        } else {
            if (activeSessionId === sessionId) {
                activeSessionId = sessions[0].id;
            }
            saveSessionsToStorage();
            renderSessionsList();
            renderActiveMessages();
        }
        logToConsole("Conversation deleted.", "info");
    }

    // Get active session
    function getActiveSession() {
        return sessions.find(s => s.id === activeSessionId);
    }

    // Render left sidebar sessions list
    function renderSessionsList() {
        chatSessionsList.innerHTML = "";

        sessions.forEach(session => {
            const item = document.createElement("div");
            item.className = `chat-session-item ${session.id === activeSessionId ? 'active' : ''}`;
            item.addEventListener("click", () => {
                activeSessionId = session.id;
                renderSessionsList();
                renderActiveMessages();
            });

            const titleSpan = document.createElement("span");
            titleSpan.className = "chat-session-title";
            titleSpan.textContent = session.title;
            titleSpan.title = session.title;

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "delete-session-btn";
            deleteBtn.innerHTML = "×";
            deleteBtn.addEventListener("click", (e) => deleteSession(session.id, e));

            item.appendChild(titleSpan);
            item.appendChild(deleteBtn);
            chatSessionsList.appendChild(item);
        });
    }

    // Render conversation messages in the center box
    function renderActiveMessages() {
        chatMessages.innerHTML = "";
        const session = getActiveSession();

        if (!session || !Array.isArray(session.messages) || session.messages.length === 0) {
            // Render welcome state
            const welcome = document.createElement("div");
            welcome.className = "chat-welcome-message";
            welcome.innerHTML = `
                <h2>Welcome to ArthriLens Chat</h2>
                <p>Type a joint health or clinical guideline question. The system will retrieve relevant context from indexed documents and generate an answer validated by the RAG Triad.</p>
                <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 1rem; border-top: 1px dashed rgba(255,255,255,0.05); padding-top: 1rem;">
                    💡 Try: <em>"What DMARD therapy is recommended for early RA?"</em> or <em>"Mind-body therapies for arthritis"</em>
                </div>
            `;
            chatMessages.appendChild(welcome);
            resetRightSidebarUI();
            return;
        }

        // Render message bubbles
        session.messages.forEach((msg, idx) => {
            const bubble = document.createElement("div");
            bubble.className = `chat-bubble ${msg.role}`;

            if (msg.role === "user") {
                bubble.innerHTML = `
                    <div class="bubble-content">${escapeHtml(msg.content)}</div>
                    <div class="bubble-meta">
                        <span>User Query</span>
                        <span>${msg.time || ""}</span>
                    </div>
                `;
            } else {
                bubble.innerHTML = `
                    <div class="bubble-content">${msg.content}</div>
                    <div class="bubble-meta">
                        <span>RAG Answer (${msg.provider || "None"})</span>
                        <span class="bubble-action-hint">Click to View Metrics</span>
                    </div>
                `;

                // Add click handler to view detailed RAG analysis for this message
                bubble.addEventListener("click", () => {
                    // Highlight selected bubble
                    document.querySelectorAll(".chat-bubble.assistant").forEach(b => b.classList.remove("selected"));
                    bubble.classList.add("selected");
                    updateRightSidebarWithMetrics(msg);
                });
            }

            chatMessages.appendChild(bubble);
        });

        // Automatically select the last assistant message and highlight it
        const assistantBubbles = chatMessages.querySelectorAll(".chat-bubble.assistant");
        if (assistantBubbles.length > 0) {
            const lastAssistantBubble = assistantBubbles[assistantBubbles.length - 1];
            lastAssistantBubble.classList.add("selected");

            // Find the last assistant message data object
            const assistantMsgs = session.messages.filter(m => m.role === "assistant");
            if (assistantMsgs.length > 0) {
                updateRightSidebarWithMetrics(assistantMsgs[assistantMsgs.length - 1]);
            }
        } else {
            resetRightSidebarUI();
        }

        scrollToBottom();
    }

    // Scroll chat window to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Reset Right Sidebar UI to blank state
    function resetRightSidebarUI() {
        metricProvider.textContent = "--";
        metricModel.textContent = "No active conversation";
        metricLatency.textContent = "0.00";
        metricLatencyDetails.textContent = "RAG: 0.00s | LLM: 0.00s";
        metricSimilarity.textContent = "--";
        similarityProgress.style.width = "0%";

        scoreContext.textContent = "--%";
        progressContext.style.width = "0%";
        descContext.textContent = "Relevance rating...";

        scoreFaith.textContent = "--%";
        progressFaith.style.width = "0%";
        descFaith.textContent = "Groundedness check...";

        scoreAnswer.textContent = "--%";
        progressAnswer.style.width = "0%";
        descAnswer.textContent = "Completeness assessment...";

        referencesContainer.innerHTML = `<span class="text-muted">No documents retrieved.</span>`;
    }

    // Populate Right Sidebar with metrics from a historical response
    function updateRightSidebarWithMetrics(msgData) {
        metricProvider.textContent = msgData.provider || "--";
        metricModel.textContent = msgData.model || "Unknown Model";
        metricLatency.textContent = msgData.latency || "0.00";

        const ragL = msgData.rag_latency !== undefined ? `${msgData.rag_latency}s` : "--";
        const llmL = msgData.llm_latency !== undefined ? `${msgData.llm_latency}s` : "--";
        metricLatencyDetails.textContent = `RAG: ${ragL} | LLM: ${llmL}`;

        const simVal = msgData.avg_similarity !== undefined ? Math.round(msgData.avg_similarity * 100) : 0;
        metricSimilarity.textContent = `${simVal}%`;
        similarityProgress.style.width = `${simVal}%`;

        if (msgData.evaluation) {
            const ctx = msgData.evaluation.context_relevance;
            const faith = msgData.evaluation.faithfulness;
            const ans = msgData.evaluation.answer_relevance;

            if (ctx) {
                scoreContext.textContent = `${ctx.score}%`;
                progressContext.style.width = `${ctx.score}%`;
                descContext.innerHTML = `<strong>Explanation:</strong> ${ctx.explanation}`;
            }
            if (faith) {
                scoreFaith.textContent = `${faith.score}%`;
                progressFaith.style.width = `${faith.score}%`;
                descFaith.innerHTML = `<strong>Explanation:</strong> ${faith.explanation}`;
            }
            if (ans) {
                scoreAnswer.textContent = `${ans.score}%`;
                progressAnswer.style.width = `${ans.score}%`;
                descAnswer.innerHTML = `<strong>Explanation:</strong> ${ans.explanation}`;
            }
        }

        // References Chunks list
        if (msgData.references && msgData.references.length > 0) {
            referencesContainer.innerHTML = "";
            msgData.references.forEach((ref, idx) => {
                const card = document.createElement("div");
                card.className = "ref-card animate-hover";

                const meta = ref.metadata || {};
                const fileName = meta.file_name || "Unknown Document";
                const pageInfo = meta.page ? `Page ${meta.page}` : (meta.row ? `Sheet '${meta.sheet}' Row ${meta.row}` : "Doc chunk");
                const similarityScore = Math.round(ref.score * 100);

                card.innerHTML = `
                    <div class="ref-meta">
                        <span class="ref-source">📄 [${idx + 1}] ${fileName} (${pageInfo})</span>
                        <span class="ref-score-badge">Similarity: ${similarityScore}%</span>
                    </div>
                    <p class="ref-text">${escapeHtml(ref.text)}</p>
                `;
                referencesContainer.appendChild(card);
            });
        } else {
            referencesContainer.innerHTML = `<span class="text-muted">No reference documents were retrieved for this answer.</span>`;
        }
    }

    // Helper to escape HTML characters safely
    function escapeHtml(unsafe) {
        if (!unsafe) return "";
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Slider range indicator update
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

    // Create a new session on btn click
    newChatBtn.addEventListener("click", () => {
        createNewSession();
    });

    // Execute chat query fetch
    async function executeQuery() {
        const query = queryInput.value.trim();
        const k = kSlider.value;

        if (!query) {
            alert("Please enter a question first.");
            return;
        }

        const session = getActiveSession();
        if (!session) return;

        // Clear input box
        queryInput.value = "";

        // Adjust textbox height
        queryInput.style.height = "auto";

        // Clear welcome message if it's the first message
        if (session.messages.length === 0) {
            chatMessages.innerHTML = "";
        }

        // Add user message to state & render bubble
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const userMsg = {
            role: "user",
            content: query,
            time: timestamp
        };
        session.messages.push(userMsg);

        // Update session title if it was default
        if (session.title === "New Conversation") {
            session.title = query.length > 25 ? query.substring(0, 22) + "..." : query;
            renderSessionsList();
        }

        saveSessionsToStorage();

        // Append user bubble to UI
        const userBubble = document.createElement("div");
        userBubble.className = "chat-bubble user";
        userBubble.innerHTML = `
            <div class="bubble-content">${escapeHtml(query)}</div>
            <div class="bubble-meta">
                <span>User Query</span>
                <span>${timestamp}</span>
            </div>
        `;
        chatMessages.appendChild(userBubble);
        scrollToBottom();

        // Add dummy assistant loading bubble
        const assistantBubble = document.createElement("div");
        assistantBubble.className = "chat-bubble assistant";
        assistantBubble.id = "loading-bubble";
        assistantBubble.innerHTML = `
            <div class="bubble-content">
                <div class="bubble-loader">
                    <span></span><span></span><span></span>
                </div>
                <span class="text-muted" style="margin-left: 0.5rem; font-size: 0.85rem;">Retrieving & Ingesting clinical guidelines...</span>
            </div>
        `;
        chatMessages.appendChild(assistantBubble);
        scrollToBottom();

        // Trigger loading state in submit button
        searchBtn.disabled = true;
        searchBtn.querySelector("span").textContent = "Analyzing...";

        // Log query start
        logToConsole(`Initializing RAG query: "${query.substring(0, 40)}..."`);
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

            // Print Fallback Chain Logs in console
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

            // Build the assistant message data object
            const assistantMsg = {
                role: "assistant",
                content: data.answer,
                provider: data.provider,
                model: data.model,
                latency: data.latency,
                llm_latency: data.llm_latency,
                rag_latency: data.rag_latency,
                avg_similarity: data.avg_similarity,
                evaluation: data.evaluation,
                references: data.references
            };

            // Replace loading bubble with active data
            const loadingBubble = document.getElementById("loading-bubble");
            if (loadingBubble) {
                loadingBubble.removeAttribute("id");
                loadingBubble.innerHTML = `
                    <div class="bubble-content">${data.answer}</div>
                    <div class="bubble-meta">
                        <span>RAG Answer (${data.provider})</span>
                        <span class="bubble-action-hint">Click to View Metrics</span>
                    </div>
                `;

                // Highlight the newly created assistant bubble
                document.querySelectorAll(".chat-bubble.assistant").forEach(b => b.classList.remove("selected"));
                loadingBubble.classList.add("selected");

                // Add click handler to this bubble
                loadingBubble.addEventListener("click", () => {
                    document.querySelectorAll(".chat-bubble.assistant").forEach(b => b.classList.remove("selected"));
                    loadingBubble.classList.add("selected");
                    updateRightSidebarWithMetrics(assistantMsg);
                });
            }

            // Add to session message state
            session.messages.push(assistantMsg);
            saveSessionsToStorage();

            // Update Right metrics panel directly
            updateRightSidebarWithMetrics(assistantMsg);

            // Log RAG details
            logToConsole(`Active model: ${data.model} | RAG Retrieval time: ${data.rag_latency}s | LLM Generation time: ${data.llm_latency}s`, "info");
            if (data.evaluation) {
                const ctx = data.evaluation.context_relevance;
                const faith = data.evaluation.faithfulness;
                const ans = data.evaluation.answer_relevance;
                if (ctx) logToConsole(`Evaluation Context Relevance: ${ctx.score}%`, ctx.score > 70 ? "success" : "warn");
                if (faith) logToConsole(`Evaluation Groundedness / Faithfulness: ${faith.score}%`, faith.score > 70 ? "success" : "warn");
                if (ans) logToConsole(`Evaluation Answer Relevance: ${ans.score}%`, ans.score > 70 ? "success" : "warn");
            }

        } catch (error) {
            logToConsole(`CRITICAL ERROR: ${error.message}`, "error");

            const loadingBubble = document.getElementById("loading-bubble");
            const errAnswer = `<span style="color: #f87171 !important;"><strong>Error:</strong> ${error.message}</span>`;

            if (loadingBubble) {
                loadingBubble.removeAttribute("id");
                loadingBubble.innerHTML = `
                    <div class="bubble-content">${errAnswer}</div>
                    <div class="bubble-meta">
                        <span style="color: #f87171 !important;">Error Occurred</span>
                    </div>
                `;
            }

            session.messages.push({
                role: "assistant",
                content: errAnswer,
                provider: "Error",
                model: "None",
                latency: "0.00",
                llm_latency: 0.0,
                rag_latency: 0.0,
                avg_similarity: 0.0,
                evaluation: null,
                references: []
            });
            saveSessionsToStorage();
            resetRightSidebarUI();
        } finally {
            searchBtn.disabled = false;
            searchBtn.querySelector("span").textContent = "Send";
            scrollToBottom();
        }
    }

    searchBtn.addEventListener("click", executeQuery);

    // Auto-resize textarea heights dynamically as user writes
    queryInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight - 4) + "px";
    });

    // Enter to send, Shift+Enter for newline
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
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
            selectedFileName.textContent = `${file.name.substring(0, 20)}... (${(file.size / 1024).toFixed(1)} KB)`;
            selectedFileName.classList.remove("text-muted");
            setUploadStatus("Ready to index", "idle");
        } else {
            selectedFileName.textContent = "Click to select PDF/Excel/TXT";
            selectedFileName.classList.add("text-muted");
            setUploadStatus("Idle", "idle");
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
        setUploadStatus("Reading file...", "running");
        uploadProgressBar.style.width = "20%";

        const reader = new FileReader();

        reader.onload = async (event) => {
            uploadProgressBar.style.width = "40%";
            setUploadStatus("Uploading content...", "running");
            const fileData = event.target.result;

            try {
                uploadProgressBar.style.width = "60%";
                setUploadStatus("Parsing & chunking file on server...", "running");

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

                uploadProgressBar.style.width = "80%";

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || `HTTP ${response.status} Error`);
                }

                const data = await response.json();
                uploadProgressBar.style.width = "100%";

                setUploadStatus("Indexed successfully!", "success");
                logToConsole(`Uploaded and indexed successfully: "${file.name}" added ${data.chunks_added} chunks.`, "success");

                // Reset file selection
                fileUploadInput.value = "";
                selectedFileName.textContent = "Click to select PDF/Excel/TXT";
                selectedFileName.classList.add("text-muted");

            } catch (error) {
                setUploadStatus("Index failed", "error");
                logToConsole(`Failed indexing "${file.name}": ${error.message}`, "error");
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.querySelector("span").textContent = "Index File";
            }
        };

        reader.onerror = () => {
            setUploadStatus("Failed reading local file.", "error");
            uploadBtn.disabled = false;
            uploadBtn.querySelector("span").textContent = "Index File";
        };

        reader.readAsDataURL(file);
    });

    // Bootstrapping: load active states from localStorage
    loadSessionsFromStorage();
});
