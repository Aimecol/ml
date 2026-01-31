/* ML Project Framework Web Interface - Main JavaScript */

let helpData = null;

document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
  checkFirstTimeUser();
});

function initializeApp() {
  // Check system status
  fetch("/api/status")
    .then((response) => response.json())
    .then((data) => {
      console.log("System status:", data);
      updateStatusIndicator(true);
    })
    .catch((error) => {
      console.error("Status check failed:", error);
      updateStatusIndicator(false);
    });

  // Load help data
  fetch("/api/help")
    .then(r => r.json())
    .then(data => {
      helpData = data.help;
    });
}

function checkFirstTimeUser() {
  if (!localStorage.getItem('visited')) {
    localStorage.setItem('visited', 'true');
    showWelcomeModal();
  }
}

function showWelcomeModal() {
  const modal = document.getElementById('welcome-modal');
  if (modal) {
    modal.classList.add('show');
    modal.style.display = 'block';
  }
}

function closeWelcomeModal() {
  const modal = document.getElementById('welcome-modal');
  if (modal) {
    modal.classList.remove('show');
    modal.style.display = 'none';
  }
  
  if (document.getElementById('dont-show-welcome').checked) {
    localStorage.setItem('dont-show-welcome', 'true');
  }
}

function showSampleDataTutorial() {
  closeWelcomeModal();
  showNotification('Navigate to Pipeline tab and click "Generate Sample Data" button', 'info');
  setTimeout(() => {
    window.location.href = '/pipeline';
  }, 1000);
}

function showWelcomeModal() {
  if (localStorage.getItem('dont-show-welcome')) return;
  const modal = document.getElementById('welcome-modal');
  if (modal) {
    modal.classList.add('show');
    modal.style.display = 'block';
  }
}

function showHelp(topic) {
  if (!helpData) {
    showNotification('Help data not loaded yet', 'warning');
    return;
  }

  const modal = document.getElementById('help-modal');
  const body = document.getElementById('help-body');

  let content = '';

  if (topic === 'algorithm' && helpData.algorithms) {
    content = '<div class="help-section">';
    for (const [key, algo] of Object.entries(helpData.algorithms)) {
      content += `
        <div class="help-item">
          <strong>${algo.name}</strong>
          <p>${algo.description}</p>
          <p><em>Best for: ${algo.best_for.join(', ')}</em></p>
          <p>Pros: ${algo.pros.join(', ')}</p>
          <p>Tip: ${algo.pro_tip}</p>
        </div>
      `;
    }
    content += '</div>';
  } else if (topic === 'scaling' && helpData.scaling_methods) {
    content = '<div class="help-section">';
    for (const [key, method] of Object.entries(helpData.scaling_methods)) {
      content += `
        <div class="help-item">
          <strong>${method.name}</strong>
          <p>${method.description}</p>
          <p><em>When: ${method.when_to_use}</em></p>
          <p>Tip: ${method.pro_tip}</p>
        </div>
      `;
    }
    content += '</div>';
  } else if (topic === 'problem_type') {
    content = `
      <div class="help-item">
        <strong>Classification</strong>
        <p>Use when your target is a category (yes/no, spam/not spam, type A/B/C)</p>
      </div>
      <div class="help-item">
        <strong>Regression</strong>
        <p>Use when your target is a number (price, temperature, age, score)</p>
      </div>
    `;
  } else if (topic === 'data_upload') {
    content = `
      <div class="help-item">
        <strong>File Format</strong>
        <p>Supported formats: CSV, Excel (.xlsx, .xls), Parquet</p>
      </div>
      <div class="help-item">
        <strong>Data Requirements</strong>
        <p>- Must have numeric features</p>
        <p>- Must have a target column (default: 'target')</p>
        <p>- Max file size: 16 MB</p>
      </div>
    `;
  } else if (topic === 'data_preview') {
    content = `
      <div class="help-item">
        <strong>What You See</strong>
        <p>- Total number of rows and columns</p>
        <p>- All column names</p>
        <p>- First 10 rows of your data</p>
        <p>- Missing value counts</p>
      </div>
      <div class="help-item">
        <strong>What It Means</strong>
        <p>Use this to verify your data uploaded correctly</p>
      </div>
    `;
  } else if (topic === 'configuration') {
    content = `
      <div class="help-item">
        <strong>Configuration Tips</strong>
        <p>- Start with default settings</p>
        <p>- Choose algorithm based on your problem</p>
        <p>- Adjust hyperparameters for fine-tuning</p>
        <p>- Test with sample data first</p>
      </div>
    `;
  } else {
    content = '<p>Help information not available for this topic</p>';
  }

  body.innerHTML = content;
  modal.classList.add('show');
  modal.style.display = 'block';

  // Close on background click
  modal.onclick = function(e) {
    if (e.target === modal) {
      closeHelpModal();
    }
  };
}

function closeHelpModal() {
  const modal = document.getElementById('help-modal');
  if (modal) {
    modal.classList.remove('show');
    modal.style.display = 'none';
  }
}

function updateStatusIndicator(isOnline) {
  const statusEl = document.getElementById("status-indicator");
  if (statusEl) {
    if (isOnline) {
      statusEl.textContent = "● Online";
      statusEl.className = "status-online";
    } else {
      statusEl.textContent = "● Offline";
      statusEl.className = "status-offline";
    }
  }
}

// API Helper Function
function apiCall(endpoint, method = "GET", body = null) {
  const options = {
    method: method,
    headers: {
      "Content-Type": "application/json",
    },
  };

  if (body && method !== "GET") {
    options.body = JSON.stringify(body);
  }

  return fetch(endpoint, options)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .catch((error) => {
      console.error("API call error:", error);
      throw error;
    });
}

// Notification Helper
function showNotification(message, type = "info") {
  const notification = document.createElement("div");
  notification.className = `notification ${type}`;
  notification.textContent = message;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.opacity = "0";
    notification.style.transform = "translateX(400px)";
    notification.style.transition = "all 0.3s ease";
    setTimeout(() => {
      if (document.body.contains(notification)) {
        document.body.removeChild(notification);
      }
    }, 300);
  }, 4000);
}

// Format number with fixed decimals
function formatNumber(num, decimals = 4) {
  if (typeof num !== "number") return num;
  return num.toFixed(decimals);
}

// Format timestamp
function formatTimestamp(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

// Chart.js initialization helper (for future use)
function initChart(ctx, config) {
  return new Chart(ctx, {
    type: config.type || "bar",
    data: config.data,
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
        title: {
          display: !!config.title,
          text: config.title,
        },
      },
      scales: config.scales || {},
    },
  });
}

// Debounce helper
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Export data as JSON
function exportAsJSON(data, filename) {
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || `export_${new Date().getTime()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Export data as CSV
function exportAsCSV(data, filename) {
  if (!Array.isArray(data) || data.length === 0) {
    alert("No data to export");
    return;
  }

  const headers = Object.keys(data[0]);
  const csv = [
    headers.join(","),
    ...data.map((row) =>
      headers
        .map((header) => {
          const value = row[header];
          if (typeof value === "string" && value.includes(",")) {
            return `"${value}"`;
          }
          return value;
        })
        .join(","),
    ),
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || `export_${new Date().getTime()}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Local storage helpers
const storage = {
  set: (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (e) {
      console.error("Storage set failed:", e);
      return false;
    }
  },
  get: (key) => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch (e) {
      console.error("Storage get failed:", e);
      return null;
    }
  },
  remove: (key) => {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (e) {
      console.error("Storage remove failed:", e);
      return false;
    }
  },
  clear: () => {
    try {
      localStorage.clear();
      return true;
    } catch (e) {
      console.error("Storage clear failed:", e);
      return false;
    }
  },
};

// Polyfill for older browsers
if (!Object.entries) {
  Object.entries = function (obj) {
    var ownProps = Object.keys(obj),
      i = ownProps.length,
      resArray = new Array(i);
    while (i--) resArray[i] = [ownProps[i], obj[ownProps[i]]];
    return resArray;
  };
}

console.log("ML Project Framework Web Interface v1.0.0 loaded");
