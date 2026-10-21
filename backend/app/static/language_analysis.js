const form = document.getElementById("analysis-form");
const textInput = document.getElementById("input-text");
const analyzeButton = document.getElementById("analyze-button");
const clearButton = document.getElementById("clear-button");

const formError = document.getElementById("form-error");
const apiError = document.getElementById("api-error");
const loadingState = document.getElementById("loading-state");
const emptyState = document.getElementById("empty-state");
const results = document.getElementById("results");
const resultStatus = document.getElementById("result-status");

const charCount = document.getElementById("char-count");

function generateObservationId() {
    return "OBS-" + (
        window.crypto && window.crypto.randomUUID
            ? window.crypto.randomUUID().slice(0, 8)
            : Date.now()
    );
}

document.getElementById("observation-id").value = generateObservationId();

function hideError(element) {
    element.textContent = "";
    element.classList.add("hidden");
}

function showError(element, message) {
    element.textContent = message;
    element.classList.remove("hidden");
}

function formatScore(value, digits = 3) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "N/A";
}

function setMeter(id, value) {
    const bar = document.getElementById(id);
    if (!bar) return;

    const number = Number(value);
    const percentage = Number.isFinite(number)
        ? Math.max(0, Math.min(100, number * 100))
        : 0;

    bar.style.width = `${percentage}%`;
}

function renderResults(data) {
    const metadata = data.metadata || {};
    const sentiment = metadata.sentiment || {};
    const mentalbert = metadata.mentalbert || {};
    const probabilities = mentalbert.class_probabilities || {};
    const negation = metadata.negation || {};
    const sarcasm = metadata.sarcasm || {};

    const negative = Number(sentiment.negative);
    const neutral = Number(sentiment.neutral);
    const positive = Number(sentiment.positive);

    document.getElementById("negative-score").textContent =
        formatScore(negative);
    document.getElementById("neutral-score").textContent =
        formatScore(neutral);
    document.getElementById("positive-score").textContent =
        formatScore(positive);

    setMeter("negative-bar", negative);
    setMeter("neutral-bar", neutral);
    setMeter("positive-bar", positive);

    document.getElementById("compound-score").textContent =
        formatScore(sentiment.compound, 4);

    document.getElementById("model-device").textContent =
        mentalbert.device || "Unknown device";

    document.getElementById("class0-score").textContent =
        formatScore(probabilities["0"]);
    document.getElementById("class1-score").textContent =
        formatScore(probabilities["1"]);

    setMeter("class0-bar", probabilities["0"]);
    setMeter("class1-bar", probabilities["1"]);

    document.getElementById("predicted-class").textContent =
        mentalbert.predicted_class !== undefined
            ? mentalbert.predicted_class
            : "Unavailable";

    document.getElementById("negation-detected").textContent =
    negation.detected === true ? "Yes" : "No";

    document.getElementById("negation-count").textContent =
    negation.count ?? 0;

    document.getElementById("negation-words").textContent =
    negation.words?.length ? negation.words.join(", ") : "None";

    document.getElementById("sarcasm-detected").textContent =
    sarcasm.possible_sarcasm === true ? "Possible" : "Not detected";

    document.getElementById("sarcasm-method").textContent =
    sarcasm.method || "Unavailable";

    document.getElementById("sarcasm-reason").textContent =
    sarcasm.reason || "Unavailable";


    document.getElementById("processed-text").textContent =
        textInput.value.trim();

    document.getElementById("token-count").textContent =
        `Token count: ${metadata.token_count ?? "N/A"}`;

    document.getElementById("json-output").textContent =
        JSON.stringify(data, null, 2);

    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    results.classList.remove("hidden");
    resultStatus.textContent = "ANALYSIS COMPLETED";
}

textInput.addEventListener("input", () => {
    charCount.textContent = `${textInput.value.length} characters`;
    hideError(formError);
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    hideError(formError);
    hideError(apiError);

    const text = textInput.value.trim();
    const userId = document.getElementById("user-id").value.trim();
    const observationId =
        document.getElementById("observation-id").value.trim();
    const sessionId =
        document.getElementById("session-id").value.trim();

    if (!text) {
        showError(formError, "Please enter text to analyze.");
        return;
    }

    if (!userId || !observationId) {
        showError(formError, "User ID and Observation ID are required.");
        return;
    }

    const requestBody = {
        text: text,
        user_id: userId,
        observation_id: observationId
    };

    if (sessionId) {
        requestBody.session_id = sessionId;
    }

    analyzeButton.disabled = true;
    loadingState.classList.remove("hidden");
    emptyState.classList.add("hidden");
    results.classList.add("hidden");
    resultStatus.textContent = "ANALYZING...";

    try {
        const response = await fetch("/api/v1/language/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestBody)
        });

        const payload = await response.json();

        if (!response.ok) {
            throw new Error(
                payload.error ||
                payload.message ||
                `Request failed: ${response.status}`
            );
        }

        if (!payload.data) {
            throw new Error("No analysis data was returned by the server.");
        }

        renderResults(payload.data);

        // A new ID avoids duplicate observation IDs on the next request.
        document.getElementById("observation-id").value =
            generateObservationId();

    } catch (error) {
        loadingState.classList.add("hidden");
        emptyState.classList.remove("hidden");
        resultStatus.textContent = "ANALYSIS FAILED";

        showError(
            apiError,
            error.message || "Unable to connect to the server."
        );
    } finally {
        analyzeButton.disabled = false;
    }
});

clearButton.addEventListener("click", () => {
    form.reset();

    document.getElementById("observation-id").value =
        generateObservationId();

    charCount.textContent = "0 characters";

    hideError(formError);
    hideError(apiError);

    results.classList.add("hidden");
    loadingState.classList.add("hidden");
    emptyState.classList.remove("hidden");

    resultStatus.textContent = "WAITING FOR INPUT";
});

