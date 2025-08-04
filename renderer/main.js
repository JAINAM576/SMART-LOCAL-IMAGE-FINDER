// Mock electron API for testing

const { ipcRenderer } = require("electron");

let allImages = [];
let filteredImages = [];
let foldersName=new Set();
let favorites = new Set();
let selectedImages = new Set();
let currentSearchType = "caption";
let contextMenuTarget = null;
let currentView = "grid";
let isSemanticSearch = false;

// Search History Management
let searchHistory = [];
const MAX_HISTORY_ITEMS = 10;

// Load search history from storage
function loadSearchHistory() {
  try {
    const stored = localStorage.getItem("semanticSearchHistory");
    if (stored) {
      searchHistory = JSON.parse(stored);
    }
  } catch (e) {
    console.warn("Could not load search history");
    searchHistory = [];
  }
}

// Save search history to storage
function saveSearchHistory() {
  try {
    localStorage.setItem(
      "semanticSearchHistory",
      JSON.stringify(searchHistory)
    );
  } catch (e) {
    console.warn("Could not save search history");
  }
}

// Add search to history
function addToSearchHistory(query, resultCount = 0) {
  if (!query || query.trim().length < 2) return;

  const normalizedQuery = query.trim().toLowerCase();

  // Remove existing entry if it exists
  searchHistory = searchHistory.filter(
    (item) => item.query.toLowerCase() !== normalizedQuery
  );

  // Add new entry at the beginning
  searchHistory.unshift({
    query: query.trim(),
    timestamp: Date.now(),
    resultCount: resultCount,
  });

  // Keep only the most recent items
  if (searchHistory.length > MAX_HISTORY_ITEMS) {
    searchHistory = searchHistory.slice(0, MAX_HISTORY_ITEMS);
  }

  saveSearchHistory();
  updateSearchHistoryDropdown();
}

// Remove item from search history
function removeFromSearchHistory(index) {
  searchHistory.splice(index, 1);
  saveSearchHistory();
  updateSearchHistoryDropdown();
}

// Clear all search history
function clearAllSearchHistory() {
  searchHistory = [];
  saveSearchHistory();
  updateSearchHistoryDropdown();
  hideSearchHistoryDropdown();
}

// Format time for display
function formatTimeAgo(timestamp) {
  const now = Date.now();
  const diff = now - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}

// Update search history dropdown content
function updateSearchHistoryDropdown() {
  const container = document.getElementById("searchHistoryItems");

  if (searchHistory.length === 0) {
    container.innerHTML =
      '<div class="search-history-empty">No recent searches</div>';
    return;
  }

  container.innerHTML = searchHistory
    .map(
      (item, index) => `
                <div class="search-history-item" onclick="selectSearchHistoryItem('${item.query.replace(
                  /'/g,
                  "\\'"
                )}')">
                    <div class="search-history-item-icon">
                        <i class="bi bi-clock-history"></i>
                    </div>
                    <div class="search-history-item-content">
                        <div class="search-history-item-query" title="${
                          item.query
                        }">
                            ${item.query}
                        </div>
                        <div class="search-history-item-meta">
                            <span class="search-history-item-time">${formatTimeAgo(
                              item.timestamp
                            )}</span>
                            <span class="search-history-item-results">${
                              item.resultCount
                            } results</span>
                        </div>
                    </div>
                    <button class="search-history-item-remove" onclick="event.stopPropagation(); removeFromSearchHistory(${index})" title="Remove">
                        <i class="bi bi-x"></i>
                    </button>
                </div>
            `
    )
    .join("");
}

// Select item from search history
function selectSearchHistoryItem(query) {
  document.getElementById("semanticInput").value = query;
  hideSearchHistoryDropdown();
  performSemanticSearchFromButton();
}

// Show search history dropdown
function showSearchHistoryDropdown() {
  const dropdown = document.getElementById("searchHistoryDropdown");
  updateSearchHistoryDropdown();
  dropdown.classList.add("show");
}

// Hide search history dropdown
function hideSearchHistoryDropdown() {
  const dropdown = document.getElementById("searchHistoryDropdown");
  dropdown.classList.remove("show");
}

// Setup search history event listeners
function setupSearchHistoryListeners() {
  const semanticInput = document.getElementById("semanticInput");
  const clearButton = document.getElementById("clearSearchHistory");

  // Show dropdown on focus
  semanticInput.addEventListener("focus", () => {
    if (searchHistory.length > 0) {
      showSearchHistoryDropdown();
    }
  });

  // Hide dropdown on blur (with delay to allow clicks)
  semanticInput.addEventListener("blur", () => {
    setTimeout(() => {
      hideSearchHistoryDropdown();
    }, 150);
  });

  // Show dropdown when clicking on input
  semanticInput.addEventListener("click", () => {
    if (searchHistory.length > 0) {
      showSearchHistoryDropdown();
    }
  });

  // Clear all history
  clearButton.addEventListener("click", (e) => {
    e.stopPropagation();
    clearAllSearchHistory();
  });

  // Hide dropdown when clicking outside
  document.addEventListener("click", (e) => {
    const dropdown = document.getElementById("searchHistoryDropdown");
    const inputGroup = document.querySelector(".semantic-search-input-group");

    if (!inputGroup.contains(e.target)) {
      hideSearchHistoryDropdown();
    }
  });

  // Handle keyboard navigation
  semanticInput.addEventListener("keydown", (e) => {
    const dropdown = document.getElementById("searchHistoryDropdown");
    const items = dropdown.querySelectorAll(".search-history-item");

    if (e.key === "ArrowDown" && dropdown.classList.contains("show")) {
      e.preventDefault();
      if (items.length > 0) {
        items[0].focus();
      }
    } else if (e.key === "Escape") {
      hideSearchHistoryDropdown();
    }
  });
}

// Initialize app
document.addEventListener("DOMContentLoaded", () => {
  loadFavorites();
  loadSearchHistory();
  setupEventListeners();
  loadExistingImages();
});

// Load favorites from storage
function loadFavorites() {
  try {
    const stored = localStorage.getItem("imageFavorites");
    if (stored) {
      favorites = new Set(JSON.parse(stored));
    }
  } catch (e) {
    console.warn("Could not load favorites");
  }
}

// Save favorites to storage
function saveFavorites() {
  try {
    localStorage.setItem("imageFavorites", JSON.stringify([...favorites]));
  } catch (e) {
    console.warn("Could not save favorites");
  }
}

// Show loading overlay
function showLoadingOverlay(text = "Loading...") {
  const overlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");
  loadingText.textContent = text;
  overlay.classList.remove("hidden");
}

// Hide loading overlay
function hideLoadingOverlay() {
  const overlay = document.getElementById("loadingOverlay");
  overlay.classList.add("hidden");
}

// Setup event listeners
function setupEventListeners() {
  // Sidebar controls
  document
    .getElementById("sidebarToggle")
    .addEventListener("click", toggleSidebar);
  document
    .getElementById("sidebarExpand")
    .addEventListener("click", expandSidebar);

  // Toolbar buttons
  document
    .getElementById("addFolderBtn")
    .addEventListener("click", ()=>selectNewFolder("folder"));

  document
    .getElementById("addFilesBtn")
    .addEventListener("click", ()=>selectNewFolder("files"));
  document
    .getElementById("refreshBtn")
    .addEventListener("click", refreshImages);
  document
    .getElementById("semanticSearchBtn")
    .addEventListener("click", focusSemanticSearch);

  // View controls (both toolbar and content header)
  document
    .getElementById("viewGridBtn")
    .addEventListener("click", () => setView("grid"));
  document
    .getElementById("viewListBtn")
    .addEventListener("click", () => setView("list"));
  document
    .getElementById("gridViewBtn")
    .addEventListener("click", () => setView("grid"));
  document
    .getElementById("listViewBtn")
    .addEventListener("click", () => setView("list"));

  // Welcome screen buttons
  document
    .getElementById("welcomeAddFolder")
    .addEventListener("click",()=> selectNewFolder("folder"));
  document
    .getElementById("welcomeSemanticSearch")
    .addEventListener("click", showSemanticSearch);
  document
    .getElementById("welcomeSearch")
    .addEventListener("click", showSearchInterface);

  // Search and filters
  document
    .getElementById("searchInput")
    .addEventListener("input", debounce(performSearch, 300));
  document
    .getElementById("semanticSearchButton")
    .addEventListener("click", performSemanticSearchFromButton);

  // Search type tabs
  document.querySelectorAll(".search-tab").forEach((tab) => {
    tab.addEventListener("click", (e) => {
      document
        .querySelectorAll(".search-tab")
        .forEach((t) => t.classList.remove("active"));
      e.target.closest(".search-tab").classList.add("active");
      currentSearchType = e.target.closest(".search-tab").dataset.type;
      updateSearchPlaceholder();
      performSearch();
    });
  });

  // Filter controls - Fixed favorites filter bug
  document.getElementById("favoritesOnly").addEventListener("change", () => {
    // Reset to all images first, then apply all filters
    filteredImages = [...allImages];
    applyFilters();
  });
  document
    .getElementById("fileTypeFilter")
    .addEventListener("change", applyFilters);
     document
    .getElementById("foldernameselector")
    .addEventListener("change", applyFilters);
  document
    .getElementById("sortBy")
    .addEventListener("change", sortAndDisplayImages);
  document
    .getElementById("sortOrder")
    .addEventListener("change", sortAndDisplayImages);

  // Advanced filters
  ["widthOp", "widthVal", "heightOp", "heightVal", "sizeOp", "sizeVal"].forEach(
    (id) => {
      document.getElementById(id).addEventListener("change", applyFilters);
      document
        .getElementById(id)
        .addEventListener("input", debounce(applyFilters, 300));
    }
  );

  // Context menu
  document.addEventListener("contextmenu", handleContextMenu);
  document.addEventListener("click", hideContextMenu);

  // Keyboard shortcuts
  document.addEventListener("keydown", handleKeyboard);

  // Image details modal
  document
    .getElementById("imageDetailsClose")
    .addEventListener("click", hideImageDetails);
  document
    .getElementById("imageDetailsModal")
    .addEventListener("click", (e) => {
      if (e.target.id === "imageDetailsModal") {
        hideImageDetails();
      }
    });

  // Add search history listeners
  setupSearchHistoryListeners();
}

// Window controls
function closeApp() {
  if (window.electronAPI) {
    window.electronAPI.closeApp();
  }
}

function minimizeApp() {
  if (window.electronAPI) {
    window.electronAPI.minimizeApp();
  }
}

function maximizeApp() {
  if (window.electronAPI) {
    window.electronAPI.maximizeApp();
  }
}

// Sidebar controls
function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("sidebarToggle");
  const icon = toggle.querySelector("i");

  sidebar.classList.toggle("collapsed");

  if (sidebar.classList.contains("collapsed")) {
    icon.className = "bi bi-chevron-right";
  } else {
    icon.className = "bi bi-chevron-left";
  }
}

function expandSidebar() {
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("sidebarToggle");
  const icon = toggle.querySelector("i");

  sidebar.classList.remove("collapsed");
  icon.className = "bi bi-chevron-left";
}

// Semantic search functions
function focusSemanticSearch() {
  showSearchInterface();
  expandSidebar();
  document.getElementById("semanticInput").focus();
}

function showSemanticSearch() {
  showSearchInterface();
  focusSemanticSearch();
}

function setSemanticQuery(query) {
  document.getElementById("semanticInput").value = query;
  performSemanticSearchFromButton();
}

function performSemanticSearchFromInput() {
  const query = document.getElementById("semanticInput").value.trim();
  if (query.length > 2) {
    performSemanticSearch(query);
  }
}

function performSemanticSearchFromButton() {
  const query = document.getElementById("semanticInput").value.trim();
  if (query) {
    performSemanticSearch(query);
  } else {
    setStatus("Please enter a search query", "warning");
  }
}

// View controls
function setView(viewType) {
  currentView = viewType;
  const grid = document.getElementById("imageGrid");

  // Update toolbar buttons
  document
    .getElementById("viewGridBtn")
    .classList.toggle("active", viewType === "grid");
  document
    .getElementById("viewListBtn")
    .classList.toggle("active", viewType === "list");

  // Update content header buttons
  document
    .getElementById("gridViewBtn")
    .classList.toggle("active", viewType === "grid");
  document
    .getElementById("listViewBtn")
    .classList.toggle("active", viewType === "list");

  // Update grid class
  if (viewType === "list") {
    grid.classList.add("list-view");
  } else {
    grid.classList.remove("list-view");
  }

  // Re-render images with new view
  displayImages();
}

// Toggle filter sections
function toggleFilterSection(header) {
  const section = header.parentElement;
  section.classList.toggle("collapsed");
}

document.getElementById("stopBtn").addEventListener("click",()=>{
   ipcRenderer.send('process-stop');
  console.log('Stop signal sent to main process');
})

// Update search placeholder
function updateSearchPlaceholder() {
  const input = document.getElementById("searchInput");
  const placeholders = {
    caption: "Search by image captions...",
    ocr: "Search text in images...",
  };
  input.placeholder = placeholders[currentSearchType];
}

// Select new folder
async function selectNewFolder(type) {
    const modal = document.getElementById("folderModal");
    const input = document.getElementById("folderInput");
    const datalist = document.getElementById("folderSuggestions");


    datalist.innerHTML = ""; // clear old
    foldersName.forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        datalist.appendChild(option);
    });

    modal.classList.remove("hidden");
    input.value = "";
    input.focus();

    document.getElementById("folderOkBtn").onclick =async  () => {
        const selectedFolder = input.value.trim();
        modal.classList.add("hidden");
        if(selectedFolder){
         
  try {
    ipcRenderer.removeAllListeners("image-process");
    setStatus("Selecting folder...", "info");
    showProcessing(true);
    // showLoadingOverlay("Processing folder...");
    ipcRenderer.on("image-process", (error, processed, total) => {
      document.getElementById("data").innerText = `${processed}/${total}`;
    });

    const result = await ipcRenderer.invoke("select-and-process-folder",type,selectedFolder);

    if (result.success) {
      setStatus(`Successfully processed ${result.count} images`, "success");
      await loadExistingImages();
      showSearchInterface();
    } else {
      setStatus("No folder selected", "warning");
      showProcessing(false)
    }
    document.getElementById("data").innerText = "";
  } catch (error) {
    setStatus("Error processing folder: " + error.message, "error");
  } finally {
    document.getElementById("data").innerText = "";
    hideLoadingOverlay();
    showProcessing(false);
  }

        }
    };

    document.getElementById("folderCancelBtn").onclick = () => {
        modal.classList.add("hidden");
    };
}




// Refresh images
async function refreshImages() {
  try {
    setStatus("Refreshing images...", "info");
    showLoadingOverlay("Refreshing images...");
    await loadExistingImages();
    setStatus("Images refreshed successfully", "success");
  } catch (error) {
    setStatus("Error refreshing images: " + error.message, "error");
  } finally {
    hideLoadingOverlay();
  }
}

// Load existing images

async function loadExistingImages() {
  try {
    allImages = await ipcRenderer.invoke("get-images-db");
    filteredImages = [...allImages];
    filteredImages.forEach((img)=>{
      foldersName.add(img.folder);

    })
document.getElementById('foldernameselector').innerHTML='<option value="">All Types</option>';
   Array.from(foldersName).forEach((name)=>{

     document.getElementById('foldernameselector').innerHTML+=`<option value=${name}>${name}</option>`
    }) 
    
    updateImageCount();

    if (allImages.length > 0) {
      showSearchInterface();
      sortAndDisplayImages();
    }

    setStatus(`Loaded ${allImages.length} images`, "success");
  } catch (error) {
    setStatus("Error loading images: " + error.message, "error");
  }
}

// Show search interface
function showSearchInterface() {
  document.getElementById("welcomeScreen").classList.add("hidden");
  document.getElementById("contentHeader").classList.remove("hidden");
  document.getElementById("imageContainer").classList.remove("hidden");
}

// Perform search
async function performSearch() {
  const query = document.getElementById("searchInput").value.trim();

  if (!query) {
    filteredImages = [...allImages];
    isSemanticSearch = false;
    updateSemanticResultsInfo();
    applyFilters();
    return;
  }

  applyTextSearch(query);
}

// Apply text search
function applyTextSearch(query) {
  const lowerQuery = query.toLowerCase();
  isSemanticSearch = false;
  updateSemanticResultsInfo();

  filteredImages = allImages.filter((image) => {
    if (currentSearchType === "caption") {
      return image.caption && image.caption.toLowerCase().includes(lowerQuery);
    } else if (currentSearchType === "ocr") {
      return image.text && image.text.toLowerCase().includes(lowerQuery);
    }
    return true;
  });

  applyFilters();
}

// Perform semantic search
async function performSemanticSearch(query) {
  try {
    setStatus("AI is searching for similar images...", "info");
    showSemanticProcessing(true);

    // Grabe The K Value from selector
    const chooenKValue = document.getElementById("kSelector").value;

    // Disable the button during search

    const button = document.getElementById("semanticSearchButton");
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML =
      '<div class="loading-spinner semantic-spinner"></div> Searching...';

    const results = await ipcRenderer.invoke(
      "get-similar-images",
      query,
      Number(chooenKValue)
    );

    if (results && results.length > 0) {
      const resultPaths = new Set(results.map((r) => r[0]));
      filteredImages = allImages.filter((img) => resultPaths.has(img.path));
      isSemanticSearch = true;
      updateSemanticResultsInfo();
      applyFilters();
      setStatus(
        `Found ${results.length} semantically similar images`,
        "success"
      );
      addToSearchHistory(query, results.length);
    } else {
      filteredImages = [];
      isSemanticSearch = false;
      updateSemanticResultsInfo();
      displayImages();
      setStatus("No similar images found. Try different keywords.", "warning");
      addToSearchHistory(query, 0);
    }

    // Re-enable the button
    button.disabled = false;
    button.innerHTML = originalText;
  } catch (error) {
    setStatus("Error performing AI search: " + error.message, "error");
    const button = document.getElementById("semanticSearchButton");
    button.disabled = false;
    button.innerHTML = '<i class="bi bi-stars"></i> Find Similar Images';
  } finally {
    showSemanticProcessing(false);
  }
}

// Update semantic results info
function updateSemanticResultsInfo() {
  const info = document.getElementById("semanticResultsInfo");
  if (isSemanticSearch) {
    info.classList.remove("hidden");
  } else {
    info.classList.add("hidden");
  }
}

// Apply filters - Fixed to work properly with favorites
function applyFilters() {
  const fileType = document.getElementById("fileTypeFilter").value;
  const folderName = document.getElementById("foldernameselector").value;

  const favoritesOnly = document.getElementById("favoritesOnly").checked;

  // Advanced filters
  const widthOp = document.getElementById("widthOp").value;
  const widthVal = parseInt(document.getElementById("widthVal").value);
  const heightOp = document.getElementById("heightOp").value;
  const heightVal = parseInt(document.getElementById("heightVal").value);
  const sizeOp = document.getElementById("sizeOp").value;
  const sizeVal = parseFloat(document.getElementById("sizeVal").value);


  console.log(filteredImages,"in apply")
  // Start with current filtered images (from search results)
  let imagesToFilter = isSemanticSearch ? filteredImages : [...allImages];

  // Apply text search if there's a query
  const query = document.getElementById("searchInput").value.trim();
  if (query && !isSemanticSearch) {
    const lowerQuery = query.toLowerCase();
    imagesToFilter = allImages.filter((image) => {
      if (currentSearchType === "caption") {
        return (
          image.caption && image.caption.toLowerCase().includes(lowerQuery)
        );
      } else if (currentSearchType === "ocr") {
        return image.text && image.text.toLowerCase().includes(lowerQuery);
      }
      return true;
    });
  }

  filteredImages = imagesToFilter.filter((image) => {
    // File type filter
    if (
      fileType &&
      image.filetype &&
      image.filetype.toLowerCase() !== fileType.toLowerCase()
    ) {
      return false;
    }
    if(folderName && image.folder && image.folder.toLowerCase() !== folderName.toLowerCase()){
      return false;
    }

    // Favorites filter
    if (favoritesOnly && !favorites.has(image.path)) {
      return false;
    }

    // Resolution filters
    if (widthOp && !isNaN(widthVal)) {
      if (!compareValues(image.width, widthOp, widthVal)) return false;
    }
    if (heightOp && !isNaN(heightVal)) {
      if (!compareValues(image.height, heightOp, heightVal)) return false;
    }

    // Size filter
    if (sizeOp && !isNaN(sizeVal)) {
      if (!compareValues(image.file_size, sizeOp, sizeVal)) return false;
    }

    return true;
  });

  sortAndDisplayImages();
}

// Compare values for filters
function compareValues(value, operator, target) {
  switch (operator) {
    case "=":
      return value === target;
    case ">":
      return value > target;
    case "<":
      return value < target;
    default:
      return true;
  }
}

// Sort and display images
function sortAndDisplayImages() {
  const sortBy = document.getElementById("sortBy").value;
  const sortOrder = document.getElementById("sortOrder").value;

  filteredImages.sort((a, b) => {
    let aVal = a[sortBy];
    let bVal = b[sortBy];

    if (sortBy.includes("Date") || sortBy === "timestamp") {
      aVal = new Date(aVal);
      bVal = new Date(bVal);
    }

    const comparison = aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
    return sortOrder === "desc" ? -comparison : comparison;
  });

  displayImages();
}

// Display images
function displayImages() {
  const grid = document.getElementById("imageGrid");
  const count = document.getElementById("resultsCount");

  count.textContent = filteredImages.length;

  if (filteredImages.length === 0) {
    grid.innerHTML = `
                    <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-secondary);">
                        <i class="bi bi-search" style="font-size: 48px; margin-bottom: 16px; display: block;"></i>
                        <h3 style="margin-bottom: 8px; color: var(--text-primary);">No images found</h3>
                        <p>Try adjusting your search criteria or use AI semantic search</p>
                    </div>
                `;
    return;
  }

  grid.innerHTML = "";

  filteredImages.forEach((image, index) => {
    const card = document.createElement("div");
    card.className = `image-card ${currentView === "list" ? "list-view" : ""} ${
      isSemanticSearch ? "semantic-result" : ""
    }`;
    card.dataset.path = image.path;
    card.dataset.index = index;

    const isFavorite = favorites.has(image.path);
    const fileName = image.path.split(/[\\/]/).pop();
    const fileSize =
      typeof image.file_size === "number"
        ? image.file_size.toFixed(1) + " MB"
        : "Unknown";

    card.innerHTML = `
                    <img src="file://${
                      image.path
                    }" alt="${fileName}" class="image-thumbnail" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjM2MzYzNjIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSIgZm9udC1zaXplPSIxMiIgZmlsbD0iIzk2OTY5NiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4='">
                    ${
                      isFavorite
                        ? '<div class="favorite-indicator"><i class="bi bi-heart-fill"></i></div>'
                        : ""
                    }
                    ${
                      isSemanticSearch
                        ? '<div class="semantic-indicator">AI</div>'
                        : ""
                    }
                    <div class="image-info">
                        <div class="image-name" title="${fileName}">${fileName}</div>
                        <div class="image-meta">
                            <span>${image.width || 0}×${
      image.height || 0
    }</span>
                            <span>${fileSize}</span>
                        </div>
                    </div>
                `;

    // Add click handler
    card.addEventListener("click", (e) => {
      if (e.ctrlKey || e.metaKey) {
        toggleSelection(card);
      } else {
        clearSelection();
        selectImage(card);
      }
    });

    // Add double-click handler to show details
    card.addEventListener("dblclick", () => {
      showImageDetails(image);
    });

    grid.appendChild(card);
  });

  updateImageCount();
}

// Show image details modal
function showImageDetails(image = null) {
  if (!image && contextMenuTarget) {
    const index = parseInt(contextMenuTarget.dataset.index);
    image = filteredImages[index];
  }

  if (!image) return;

  const modal = document.getElementById("imageDetailsModal");
  const title = document.getElementById("imageDetailsTitle");
  const preview = document.getElementById("imageDetailsPreview");
  const info = document.getElementById("imageDetailsInfo");

  const fileName = image.path.split(/[\\/]/).pop();
  title.textContent = fileName;
  preview.src = `file://${image.path}`;

  // Format dates
  const formatDate = (dateStr) => {
    if (!dateStr) return "Unknown";
    return new Date(dateStr).toLocaleString();
  };

  // Format file size
  const formatFileSize = (size) => {
    if (typeof size !== "number") return "Unknown";
    if (size < 1) return `${(size * 1024).toFixed(0)} KB`;
    return `${size.toFixed(2)} MB`;
  };

  info.innerHTML = `
                <div class="image-details-section">
                    <div class="image-details-section-title">File Information</div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">File Name</div>
                        <div class="image-details-field-value">${fileName}</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">File Path</div>
                        <div class="image-details-field-value">${
                          image.path
                        }</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">File Type</div>
                        <div class="image-details-field-value">${
                          image.filetype?.toUpperCase() || "Unknown"
                        }</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">File Size</div>
                        <div class="image-details-field-value">${formatFileSize(
                          image.file_size
                        )}</div>
                    </div>
                </div>
                
                <div class="image-details-section">
                    <div class="image-details-section-title">Image Properties</div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Dimensions</div>
                        <div class="image-details-field-value">${
                          image.width || 0
                        } × ${image.height || 0} pixels</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Aspect Ratio</div>
                        <div class="image-details-field-value">${
                          image.width && image.height
                            ? (image.width / image.height).toFixed(2)
                            : "Unknown"
                        }</div>
                    </div>
                </div>
                
                <div class="image-details-section">
                    <div class="image-details-section-title">AI Analysis</div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Caption</div>
                        <div class="image-details-field-value">${
                          image.caption || "No caption available"
                        }</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Extracted Text (OCR)</div>
                        <div class="image-details-field-value multiline">${
                          image.text || "No text detected"
                        }</div>
                    </div>
                </div>
                
                <div class="image-details-section">
                    <div class="image-details-section-title">Timestamps</div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Date Created</div>
                        <div class="image-details-field-value">${formatDate(
                          image.DateCreated
                        )}</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Date Modified</div>
                        <div class="image-details-field-value">${formatDate(
                          image.DateModified
                        )}</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Date Accessed</div>
                        <div class="image-details-field-value">${formatDate(
                          image.DateAccessed
                        )}</div>
                    </div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">Added to Database</div>
                        <div class="image-details-field-value">${formatDate(
                          image.timestamp
                        )}</div>
                    </div>
                </div>
                
                <div class="image-details-section">
                    <div class="image-details-section-title">Technical</div>
                    <div class="image-details-field">
                        <div class="image-details-field-label">File Hash</div>
                        <div class="image-details-field-value">${
                          image.hex || "Not available"
                        }</div>
                    </div>
                </div>
            `;

  modal.classList.remove("hidden");
  hideContextMenu();
}

// Hide image details modal
function hideImageDetails() {
  const modal = document.getElementById("imageDetailsModal");
  modal.classList.add("hidden");
}

// Image selection
function selectImage(card) {
  card.classList.add("selected");
  selectedImages.add(card.dataset.path);
  updateSelectionCount();
}

function toggleSelection(card) {
  if (card.classList.contains("selected")) {
    card.classList.remove("selected");
    selectedImages.delete(card.dataset.path);
  } else {
    card.classList.add("selected");
    selectedImages.add(card.dataset.path);
  }
  updateSelectionCount();
}

function clearSelection() {
  document.querySelectorAll(".image-card.selected").forEach((card) => {
    card.classList.remove("selected");
  });
  selectedImages.clear();
  updateSelectionCount();
}

function updateSelectionCount() {
  const countEl = document.getElementById("selectedCount");
  const count = selectedImages.size;

  if (count > 0) {
    countEl.style.display = "flex";
    countEl.querySelector("span").textContent = `${count} selected`;
  } else {
    countEl.style.display = "none";
  }
}

// Context menu
function handleContextMenu(e) {
  const card = e.target.closest(".image-card");
  if (card) {
    e.preventDefault();
    contextMenuTarget = card;
    showContextMenu(e.clientX, e.clientY);
  }
}

function showContextMenu(x, y) {
  const menu = document.getElementById("contextMenu");
  menu.classList.remove("hidden");
  menu.style.left = x + "px";
  menu.style.top = y + "px";
}

function hideContextMenu() {
  document.getElementById("contextMenu").classList.add("hidden");
  contextMenuTarget = null;
}

// Context menu actions
function openImage(path = null) {
  const imagePath = path || contextMenuTarget?.dataset.path;
  if (imagePath && window.electronAPI) {
    window.electronAPI.openImage(imagePath);
  }
  hideContextMenu();
}

function toggleFavorite() {
  if (!contextMenuTarget) return;

  const path = contextMenuTarget.dataset.path;
  if (favorites.has(path)) {
    favorites.delete(path);
  } else {
    favorites.add(path);
  }

  saveFavorites();
  displayImages(); // Re-render to update favorite indicators
  hideContextMenu();
}

function copyPath() {
  if (!contextMenuTarget) return;

  const path = contextMenuTarget.dataset.path;
  if (navigator.clipboard) {
    navigator.clipboard.writeText(path);
    setStatus("Path copied to clipboard", "success");
  }
  hideContextMenu();
}

function showInExplorer() {
  if (!contextMenuTarget) return;

  const path = contextMenuTarget.dataset.path;
  if (window.electronAPI) {
    window.electronAPI.showInExplorer(path);
  }
  hideContextMenu();
}

async function deleteImage() {
  if (!contextMenuTarget) return;
console.log(selectedImages,"selected");
  const path = contextMenuTarget.dataset.path;
 showProcessing(true);
  if (confirm("Are you sure you want to delete this image?")) {
    
       await ipcRenderer.invoke("remove-image-from-db",Array.from(selectedImages));
     
      showProcessing(false);
    setStatus("Image Deleted Successfully ", "warning");
  }
  hideContextMenu();
}

// Keyboard shortcuts
function handleKeyboard(e) {
  if (e.ctrlKey || e.metaKey) {
    switch (e.key) {
      case "a":
        e.preventDefault();
        selectAllImages();
        break;
      case "f":
        e.preventDefault();
        document.getElementById("searchInput").focus();
        break;
      case "k":
        e.preventDefault();
        document.getElementById("semanticInput").focus();
        break;
    }
  }

  if (e.key === "Escape") {
    clearSelection();
    hideContextMenu();
    hideImageDetails();
    hideSearchHistoryDropdown();
  }
}

function selectAllImages() {
  document.querySelectorAll(".image-card").forEach((card) => {
    card.classList.add("selected");
    selectedImages.add(card.dataset.path);
  });
  updateSelectionCount();
}

// Utility functions
function updateImageCount() {
  document.getElementById(
    "imageCount"
  ).textContent = `${filteredImages.length} images`;
}

function setStatus(message, type = "info") {
  const statusEl = document.getElementById("statusMessage");
  const icons = {
    info: "bi-info-circle",
    success: "bi-check-circle",
    warning: "bi-exclamation-triangle",
    error: "bi-x-circle",
  };

  statusEl.innerHTML = `<i class="bi ${icons[type]}"></i> ${message}`;
}

function showProcessing(show) {
  const processingEl = document.getElementById("processingStatus");
  processingEl.style.display = show ? "flex" : "none";
}

function showSemanticProcessing(show) {
  const processingEl = document.getElementById("semanticProcessingStatus");
  processingEl.style.display = show ? "flex" : "none";
}

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

// Initialize search placeholder
updateSearchPlaceholder();

const downloadingModelInfo = document.getElementById("loaderelement");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const successState = document.getElementById("successState");
const successStateModule = document.getElementById("successStateModule");

const progressBar = document.getElementById("progressBar");
const progressPercentage = document.getElementById("progressPercentage");
const modelName = document.getElementById("modelName");
const statusText = document.getElementById("statusText");
const errorMessage = document.getElementById("errorMessage");
const retryBtn = document.getElementById("retryBtn");
const closeBtn = document.getElementById("closeBtn");

let currentProgress = 0;
let downloadStartTime = 0;

let currenteventname = "load-model";

function showLoader() {
  downloadingModelInfo.classList.remove("element-none");
  showLoadingState();
}

function hideLoader() {
  downloadingModelInfo.classList.add("element-none");
}

function showLoadingState() {
  loadingState.classList.remove("element-none");
  errorState.classList.add("element-none");
  successState.classList.add("element-none");
  successStateModule.classList.add("element-none");

  resetProgress();
}

function showErrorState(error = null) {
  loadingState.classList.add("element-none");
  errorState.classList.remove("element-none");
  successState.classList.add("element-none");
  successStateModule.classList.add("element-none");

  if (error) {
    errorMessage.textContent =
      error.message || "An unexpected error occurred during model download.";
  }
}

function showSuccessState() {
  loadingState.classList.add("element-none");
  errorState.classList.add("element-none");
  successState.classList.remove("element-none");

  // Auto-close after 2 seconds
  setTimeout(() => {
    hideLoader();
  }, 2000);
}
function showsuccessStateModule() {
  loadingState.classList.add("element-none");
  errorState.classList.add("element-none");
  successStateModule.classList.remove("element-none");

  // Auto-close after 2 seconds
  setTimeout(() => {
    hideLoader();
  }, 2000);
}

function resetProgress() {
  currentProgress = 0;
  progressBar.style.width = "0%";
  progressPercentage.textContent = "0%";
  statusText.textContent = "Initializing...";
  downloadStartTime = Date.now();
}

function updateProgress(progress) {
  currentProgress = progress;
  progressBar.style.width = `${progress}%`;
  progressPercentage.textContent = `${progress}%`;

  // Calculate download speed and ETA
  const elapsed = (Date.now() - downloadStartTime) / 1000;
  const rate = progress / elapsed;
  const eta = rate > 0 ? (100 - progress) / rate : 0;

  if (progress > 0 && eta > 0) {
    statusText.textContent = `ETA: ${Math.round(eta)}s`;
  } else {
    statusText.textContent = "Downloading...";
  }
}

function updateModelName(name) {
  modelName.textContent = `Model: ${name}`;
}

async function load_model(force) {
  try {
    showLoader();

    let downloadInProgress = false;
    let downloadCompleted = false;

    // Set up event listeners for progress updates
    ipcRenderer.removeAllListeners("model-download-progress");
    ipcRenderer.removeAllListeners("model-download-name");
    ipcRenderer.removeAllListeners("model-download-error");
    ipcRenderer.removeAllListeners("model-download-complete");

    ipcRenderer.on("model-download-progress", (event, progress) => {
      downloadInProgress = true;
      const numericProgress = Math.round(parseFloat(progress));
      updateProgress(numericProgress);
    });

    ipcRenderer.on("model-download-name", (event, name) => {
      updateModelName(name);
    });

    ipcRenderer.on("model-download-error", (event, error) => {
      showErrorState({ message: error });
    });

    ipcRenderer.on("model-download-complete", (event, result) => {
      downloadCompleted = true;
        showSuccessState();
  
    });

    // Start the download - but don't immediately handle the result
    const resultPromise = ipcRenderer.invoke('load-model',force);

    // Wait a bit to see if progress events start coming
    await new Promise((resolve) => setTimeout(resolve, 5000));

    // If no progress events have started, then handle the immediate result
    if (!downloadInProgress) {
      try {
        const result = await resultPromise;
        console.log("Immediate result:", result);

        if (result && result.success !== false) {
            showSuccessState();
       
        } else {
          throw new Error(
            result?.error || "Model loading failed - no progress detected"
          );
        }
      } catch (immediateError) {
        console.error("Immediate loading error:", immediateError);
        showErrorState(immediateError);
      }
    } else {
      // Progress events are coming, wait for completion
      try {
        const result = await resultPromise;
        console.log("Final result after progress:", result);

        // Only show success if we haven't already handled completion
        if (!downloadCompleted) {
          if (result && result.success !== false) {
        
              showSuccessState();
          
          } else {
            throw new Error(
              result?.error || "Model loading failed after progress"
            );
          }
        }
      } catch (progressError) {
        console.error("Progress loading error:", progressError);
        // Only show error if we haven't already handled it via events
        if (!downloadCompleted) {
          showErrorState(progressError);
        }
      }
    }
  } catch (error) {
    console.error("Model loading setup error:", error);
    showErrorState(error);
  }
}
async function handleRetry(currenteventname) {
  if(currenteventname=="load-model"){

    await load_model();
  }else{
    await load_module();
  }
}
// Event listeners
retryBtn.removeEventListener("click",async ()=> handleRetry("load-model"));
retryBtn.addEventListener("click",async ()=> handleRetry("load-model"));

closeBtn.addEventListener("click", () => {
  hideLoader();
});
function addToTerminal(text) {
  const terminalBody = document.getElementById("terminal-body");
  const terminal = document.getElementById("terminal");

  // Show terminal if hidden
  if (terminal.classList.contains("hidden")) {
    terminal.classList.remove("hidden");
  }

  // Remove cursor from last line
  const cursor = terminalBody.querySelector(".cursor");
  if (cursor) {
    cursor.remove();
  }

  // Add new line with text
  const newLine = document.createElement("div");
  newLine.className = "terminal-line";
  newLine.textContent = text;
  terminalBody.appendChild(newLine);

  // Add cursor to new line
  const cursorSpan = document.createElement("span");
  cursorSpan.className = "cursor";
  newLine.appendChild(cursorSpan);

  // Auto-scroll to bottom
  terminalBody.scrollTop = terminalBody.scrollHeight;
}
function closeTerminal() {
  const terminal = document.getElementById("terminal");
  const terminalBody = document.getElementById("terminal-body");

  // Hide terminal
  terminal.classList.add("hidden");

  // Clear all content except initial prompt
  terminalBody.innerHTML = `
                <div class="terminal-line">
                    <span class="prompt">$</span> Starting module installation...<span class="cursor"></span>
                </div>
            `;
}

async function load_module(force) {
  let downloadInProgress = false;
  let downloadCompleted = false;

  ipcRenderer.removeAllListeners("terminal-log");
  ipcRenderer.removeAllListeners("module-download-error");
  ipcRenderer.removeAllListeners("module-completed");

  ipcRenderer.on("terminal-log", (event, progress) => {
    downloadInProgress = true;
    addToTerminal(progress);
    console.log(progress, "jjj");
  });

  ipcRenderer.on("module-download-error", (event, error) => {
    closeTerminal();
    retryBtn.removeEventListener("click",async ()=> handleRetry("load-module"));
    retryBtn.addEventListener("click",async()=> handleRetry("load-module"));
    showErrorState(error);
  });

  ipcRenderer.on("module-completed", (event, result) => {
    downloadCompleted = true;
    showsuccessStateModule();
    closeTerminal();
  });

  const resultPromise = ipcRenderer.invoke("load-module",force);
  const result = await resultPromise;
  console.log("Immediate result:", result);
}


// TO FORCEFULLY DOWNLOAD MODEL AND MODULES

let loadModelElement=document.getElementById("loadModel");
let loadModuleElement=document.getElementById("loadModule");

loadModelElement.addEventListener("change",async (e)=>{
  if(loadModelElement.checked){
await load_model(true)
  }
  loadModelElement.checked=false;
})
loadModuleElement.addEventListener("change",async (e)=>{
  if(loadModuleElement.checked){
await load_module(true)
  }
  loadModuleElement.checked=false;

})


async function loadAllModelsInSequence() {
  try {
    await load_model();
    currenteventname = "load-module";
    console.log("NOW SECOND MODEL RUNNING");
    await load_module();
    console.log("DONE");
  } catch (error) {
    console.error("Error while loading models:", error);
  }
}

loadAllModelsInSequence();
