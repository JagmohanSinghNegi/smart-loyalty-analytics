const API_BASE_URL =
  window.location.hostname === "localhost"
    ? "http://127.0.0.1:5000"
    : "https://your-render-url.onrender.com";

document.addEventListener("DOMContentLoaded", function () {
  const loyaltyForm = document.getElementById("loyaltyForm");
  if (loyaltyForm) {
    loyaltyForm.addEventListener("submit", handleLoyaltySubmit);
  }

  const recommendationForm = document.getElementById("recommendationForm");
  if (recommendationForm) {
    recommendationForm.addEventListener("submit", handleRecommendationSubmit);
  }
});

async function handleLoyaltySubmit(e) {
  e.preventDefault();

  const customerId = document.getElementById("customerId").value.trim();
  if (!customerId) {
    showError("Please enter a Customer ID", "loyaltyForm");
    return;
  }

  await predictLoyalty(customerId);
}

async function predictLoyalty(customerId) {
  const loadingEl = document.getElementById("loading");
  const errorEl = document.getElementById("error");
  const resultEl = document.getElementById("result");

  // Reset UI
  loadingEl.classList.remove("hidden");
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");

  try {
    const response = await fetch(`${API_BASE_URL}/predict-loyalty`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ customer_id: customerId }),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Failed to predict loyalty", "loyaltyForm");
      loadingEl.classList.add("hidden");
      return;
    }

    // Display result
    document.getElementById("resultCustomerId").textContent = data.customer_id;
    document.getElementById("resultScore").textContent =
      (data.loyalty_score * 100).toFixed(1) + "%";

    const statusEl = document.getElementById("resultStatus");
    if (data.loyal) {
      statusEl.textContent = "✓ Loyal";
      statusEl.className = "loyal";
    } else {
      statusEl.textContent = "✗ Not Loyal";
      statusEl.className = "not-loyal";
    }

    loadingEl.classList.add("hidden");
    resultEl.classList.remove("hidden");
  } catch (error) {
    console.error("Error:", error);
    showError(
      "Network error. Make sure the Flask server is running on http://127.0.0.1:5000",
      "loyaltyForm"
    );
    loadingEl.classList.add("hidden");
  }
}

async function handleRecommendationSubmit(e) {
  e.preventDefault();

  const productName = document.getElementById("productName").value.trim();
  const topN = parseInt(document.getElementById("topN").value) || 5;

  if (!productName) {
    showError("Please enter a Product Name", "recommendationForm");
    return;
  }

  if (topN < 1 || topN > 20) {
    showError(
      "Number of recommendations must be between 1 and 20",
      "recommendationForm"
    );
    return;
  }

  await getRecommendations(productName, topN);
}

async function getRecommendations(productName, topN) {
  const loadingEl = document.getElementById("loading");
  const errorEl = document.getElementById("error");
  const resultEl = document.getElementById("result");

  // Reset UI
  loadingEl.classList.remove("hidden");
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");

  try {
    const response = await fetch(`${API_BASE_URL}/recommend-products`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ product: productName, top_n: topN }),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(
        data.error || "Failed to get recommendations",
        "recommendationForm"
      );
      loadingEl.classList.add("hidden");
      return;
    }

    // Display result
    document.getElementById("resultProduct").textContent = data.product;

    const listEl = document.getElementById("recommendationsList");
    listEl.innerHTML = "";

    if (data.recommendations && data.recommendations.length > 0) {
      data.recommendations.forEach((product, index) => {
        const li = document.createElement("li");
        li.textContent = `${index + 1}. ${product}`;
        listEl.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "No recommendations available for this product.";
      listEl.appendChild(li);
    }

    loadingEl.classList.add("hidden");
    resultEl.classList.remove("hidden");
  } catch (error) {
    console.error("Error:", error);
    showError(
      "Network error. Make sure the Flask server is running on http://127.0.0.1:5000",
      "recommendationForm"
    );
    loadingEl.classList.add("hidden");
  }
}

function showError(message, formId) {
  const errorEl = document.getElementById("error");
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");

  // Scroll to error
  setTimeout(() => {
    errorEl.scrollIntoView({ behavior: "smooth", block: "center" });
  }, 100);
}
