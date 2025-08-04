const fs = require("fs");
const path = require("path");
const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const { spawn } = require("child_process");
const Tesseract = require("tesseract.js");
const sizeOf = require("image-size");
const exifr = require("exifr");
const crypto = require("crypto");

let db;
const HASH_DB_PATH = app.isPackaged
  ? path.join(process.resourcesPath, 'assests', 'processed_hashes.json') 
  : path.join(__dirname, 'assests', 'processed_hashes.json');     

let pythonScriptPath;

if (app.isPackaged) {
  pythonScriptPath = path.join(
    process.resourcesPath,
    "python",
    "Portable Python-3.10.5 x64",
    "App",
    "Python",
    "python.exe"
  );
} else {
  pythonScriptPath = path.join(
    __dirname,
    "python",
    "Portable Python-3.10.5 x64",
    "App",
    "Python",
    "python.exe"
  );
}

function returnPath(pythonfile) {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "python", pythonfile);
  }
  return path.join(__dirname, "python", pythonfile);
}

function loadHashDB() {
  if (!fs.existsSync(HASH_DB_PATH)) return {};
  return JSON.parse(fs.readFileSync(HASH_DB_PATH));
}

function saveHashDB(db) {
  fs.writeFileSync(HASH_DB_PATH, JSON.stringify(db, null, 2));
}

function isAlreadyProcessed(hash, db) {
  return db[hash] === true;
}

const initializeDatabase = async () => {
  try {
    const { Low } = await import("lowdb");
    const { JSONFile } = await import("lowdb/node");

   const imagesDbPath = app.isPackaged
  ? path.join(process.resourcesPath, 'assests', 'images.json')
  : path.join(__dirname, 'assests', 'images.json');

const adapter = new JSONFile(imagesDbPath);
    db = new Low(adapter, { images: [] });

    await db.read();

    if (!db.data) {
      db.data = { images: [] };
      await db.write(); // Save the default structure
    }

    console.log("Database initialized successfully");
  } catch (error) {
    console.error("Failed to initialize database:", error);
  }
};

let needToLoad=true;
let needToLoadModules=true;

const checkModelsSize = async () => {
  try {
    const { Low } = await import("lowdb");
    const { JSONFile } = await import("lowdb/node");

       const imagesDbPath = app.isPackaged
  ? path.join(process.resourcesPath, 'assests', 'log.json')
  : path.join(__dirname, 'assests', 'log.json');

const adapter = new JSONFile(imagesDbPath);
   let checkModel = new Low(adapter,{});

    await checkModel.read();

   
    if(Object.keys(checkModel.data).length>=5){
        needToLoad=false;
    }

    console.log("Total tracked files:", Object.keys(checkModel.data).length);
  } catch (error) {
    console.error("Failed to initialize database:", error);
  }
};
const checkModulesSize = async () => {
  try {
    const { Low } = await import("lowdb");
    const { JSONFile } = await import("lowdb/node");

  
        const imagesDbPath = app.isPackaged
  ? path.join(process.resourcesPath, 'assests', 'logmodules.json')
  : path.join(__dirname, 'assests', 'logmodules.json');

const adapter = new JSONFile(imagesDbPath);
    let checkModule = new Low(adapter,{});

    await checkModule.read();

 
    if(Object.keys(checkModule.data).length>=9){
        needToLoadModules=false;
    }

    console.log("Total tracked files:", Object.keys(checkModule.data).length);
  } catch (error) {
    console.error("Failed to initialize database:", error);
  }
};
function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile("renderer/index.html");
}

app.whenReady().then(async () => {
  console.log("App is ready, initializing database...");
  await initializeDatabase();
  await checkModelsSize();
  await checkModulesSize();
  console.log("Database ready, creating window...");
  createWindow();
});
let processes={}

app.on('before-quit', () => {
    killAllProcesses();
});

function killAllProcesses() {
    for (const key in processes) {
        if (processes[key]) {
            processes[key].kill();
        }
    }
}
app.on("window-all-closed", () => {
  app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

function scanImages(folderPath) {
  const extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"];
  const imagePaths = [];

  function walk(dir) {
    try {
      const files = fs.readdirSync(dir);
      for (let file of files) {
        const fullPath = path.join(dir, file);
        try {
          const stat = fs.statSync(fullPath);
          if (stat.isDirectory()) {
            walk(fullPath);
          } else if (extensions.includes(path.extname(file).toLowerCase())) {
            imagePaths.push(fullPath);
          }
        } catch (error) {
          console.warn(`Skipping file ${fullPath}: ${error.message}`);
        }
      }
    } catch (error) {
      console.warn(`Cannot read directory ${dir}: ${error.message}`);
    }
  }

  walk(folderPath);
  return imagePaths;
}

function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return 0;
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const mb = bytes / (k * k);
  return parseFloat(mb.toFixed(dm));
}

function getImageUniqueId(fileBuffer) {
  const hashSum = crypto.createHash("sha256");
  hashSum.update(fileBuffer);
  return hashSum.digest("hex");
}

// NEW: Combined select and process folder handler
ipcMain.handle("select-and-process-folder", async (event,type,selectedFolder) => {
  let properties=type=="folder"?["openDirectory"]:["openFile", "multiSelections"]
  try {
    const result = await dialog.showOpenDialog({
      properties:properties,
      title: "SELECT IMAGE(S) OR FOLDER(S) TO TRACK:",
      filters: [
        { name: "Images", extensions: ["jpg", "jpeg", "png", "bmp", "webp"] },
      ],
    });

    if (result.canceled || result.filePaths.length === 0) {
      return { success: false, count: 0 };
    }

    let imagePaths = [];

    for (const selectedPath of result.filePaths) {
      const stats = fs.statSync(selectedPath);

      if (stats.isDirectory()) {
        // Scan folder for images
        imagePaths.push(...scanImages(selectedPath));
      } else {
        // Single image file
        imagePaths.push(selectedPath);
      }
    }

    if (imagePaths.length === 0) {
      return {
        success: false,
        count: 0,
        message: "No images found in selected path(s)",
      };
    }

    // ---- Processing Images ----
    let processedCount = 0;
    const hashDB = loadHashDB();
    let shouldStop = false;

    ipcMain.on("process-stop", () => {
      console.log("Stop requested by user.");
      shouldStop = true;
    });

    for (const imagePath of imagePaths) {
      if (shouldStop) {
        console.log("Processing stopped.");
        break;
      }

      try {
        const buffer = fs.readFileSync(imagePath);
        const hex = getImageUniqueId(buffer);

        // Skip if already processed
        if (isAlreadyProcessed(hex, hashDB)) {
          console.log(`Skipping already processed: ${path.basename(imagePath)}`);
          processedCount++;
          event.sender.send("image-process", processedCount + 1, imagePaths.length);
          continue;
        }

        await processImage(imagePath, buffer, hex, hashDB,selectedFolder);
        processedCount++;
        event.sender.send("image-process", processedCount + 1, imagePaths.length);

        // Mark as processed
        hashDB[hex] = true;
        saveHashDB(hashDB);
      } catch (error) {
        console.error(`Error processing ${imagePath}:`, error);
      }
    }

    return { success: true, count: processedCount };
  } catch (error) {
    console.error("Error in select-and-process:", error);
    return { success: false, count: 0, error: error.message };
  }
});

// UPDATED: Refactored image processing into separate function
async function processImage(imagePath, buffer, hex, hashDB,selectedFolder) {
  try {
    // Get basic info
    const dimensions = sizeOf.imageSize(buffer);
    const fileStats = fs.statSync(imagePath);
    const fileSizeInBytes = fileStats.size;
    const fileType = path.extname(imagePath).slice(1).toLowerCase();

    let metaData = {
      Width: dimensions.width,
      Height: dimensions.height,
      file_size: formatBytes(fileSizeInBytes),
      FileType: fileType,
    };

    // Get EXIF data
    try {
      const exifData = await exifr.parse(imagePath);
      if (exifData) {
        metaData.CapturedDate = exifData?.DateTimeOriginal;
        metaData.CameraModel = exifData?.Model;
        metaData.CameraMake = exifData?.Make;
        metaData.FNumber = exifData?.FNumber;
        metaData.GPS = exifData?.GPSLatitude;
      }
    } catch (exifError) {
      console.warn(`EXIF parsing failed for ${imagePath}:`, exifError.message);
    }

    // OCR processing
    let ocrText = "";
    try {
      const result = await Tesseract.recognize(imagePath, "eng");
      ocrText = result.data.text || "";
    } catch (ocrError) {
      console.warn(`OCR failed for ${imagePath}:`, ocrError.message);
    }

    // Get AI caption
    let caption = "No caption generated";
    try {
      const scriptPath = returnPath("main.py");
      console.log(pythonScriptPath, "come in caption");
      caption = await runImageCaptioning(scriptPath, imagePath);
    } catch (captionError) {
      console.warn(
        `Caption generation failed for ${imagePath}:`,
        captionError.message
      );
    }

    // Save to database
    await db.read();

    if (!db.data) {
      db.data = { images: [] };
    }
    if (!db.data.images) {
      db.data.images = [];
    }

    db.data.images.push({
      path: imagePath,
      caption: caption,
      text: ocrText,
      timestamp: new Date().toISOString(),
      width: metaData.Width,
      height: metaData.Height,
      file_size: metaData.file_size,
      filetype: metaData.FileType,
      captureddate: metaData.CapturedDate,
      cameramodel: metaData.CameraModel,
      cameramake: metaData.CameraMake,
      fnumber: metaData.FNumber,
      gps: metaData.GPS,
      DateCreated: fileStats.birthtime.toISOString(),
      DateModified: fileStats.mtime.toISOString(),
      DateAccessed: fileStats.atime.toISOString(),
      hex: hex,
      folder:selectedFolder
    });

    await db.write();
    console.log(`Successfully processed: ${path.basename(imagePath)}`);
  } catch (error) {
    console.error(`Error processing image ${imagePath}:`, error);
    throw error;
  }
}

// LEGACY: Keep for backward compatibility
ipcMain.handle("select-folder", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
    title: "SELECT YOUR IMAGE FOLDER:",
  });
  if (result.canceled) return [];
  return scanImages(result.filePaths[0]);
});

ipcMain.handle("caption-image", async (event, imagePath) => {
  try {
    const buffer = fs.readFileSync(imagePath);
    const hex = getImageUniqueId(buffer);
    const hashDB = loadHashDB();

    if (isAlreadyProcessed(hex, hashDB)) {
      console.log("Image was already processed");
      return "Already processed";
    }

    await processImage(imagePath, buffer, hex, hashDB);

    // Mark as processed
    hashDB[hex] = true;
    saveHashDB(hashDB);

    return "Processed successfully";
  } catch (error) {
    console.error("Image processing error:", error);
    throw error;
  }
});

async function runImageCaptioning(scriptPath, imagePath) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn(pythonScriptPath, [scriptPath, imagePath], {
      cwd: path.dirname(scriptPath),
    });
processes[0]=pythonProcess
    let stdout = "";
    let stderr = "";

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        const caption = parseCaption(stdout);
        resolve(caption);
      } else {
        reject(
          new Error(`Python script failed with code ${code}. Error: ${stderr}`)
        );
      }
    });

    pythonProcess.on("error", (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });

    // Increased timeout
  });
}

function parseCaption(output) {
  try {
    const lines = output.split("\n");

    for (const line of lines) {
      if (line.includes("Caption:") || line.includes("Generated caption:")) {
        const captionMatch = line.match(
          /Caption:\s*(.+)|Generated caption:\s*'(.+)'/
        );
        if (captionMatch) {
          return captionMatch[1] || captionMatch[2] || "No caption generated";
        }
      }
    }

    const nonEmptyLines = lines.filter((line) => line.trim() !== "");
    if (nonEmptyLines.length > 0) {
      const lastLine = nonEmptyLines[nonEmptyLines.length - 1];
      if (
        !lastLine.includes("✓") &&
        !lastLine.includes("Error:") &&
        !lastLine.includes("WARNING:")
      ) {
        return lastLine.trim();
      }
    }

    return "No caption generated";
  } catch (error) {
    console.error("Error parsing caption:", error);
    return "Error parsing caption";
  }
}

ipcMain.handle("get-images-db", async () => {
  try {
    await db.read();

    if (!db.data) {
      db.data = { images: [] };
      await db.write();
    }
    if (!db.data.images) {
      db.data.images = [];
      await db.write();
    }

    return db.data.images;
  } catch (error) {
    console.error("Database read error:", error);
    return [];
  }
});

// FIXED: Similarity search handler
ipcMain.handle("get-similar-images", async (event, query, k = 10) => {
  console.log(k, "k ");
  try {
    const scriptPath = returnPath("similarity-search-image.py");
    const results = await findSimilarImages(scriptPath, query, k);
    return results;
  } catch (error) {
    console.error("Similarity search error:", error);
    throw error;
  }
});

async function findSimilarImages(scriptPath, query, k) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn(
      pythonScriptPath,
      [scriptPath, query, k.toString()],
      {
        cwd: path.dirname(scriptPath),
      }
    );
    processes[1]=pythonProcess

    let stdout = "";
    let stderr = "";

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          console.log("Similarity search results:", result);
          resolve(result);
        } catch (parseError) {
          console.error("Error parsing similarity search results:", parseError);
          reject(new Error("Failed to parse similarity search results"));
        }
      } else {
        reject(
          new Error(`Python script failed with code ${code}. Error: ${stderr}`)
        );
      }
    });

    pythonProcess.on("error", (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });

    setTimeout(() => {
      pythonProcess.kill();
      reject(new Error("Similarity search process timed out"));
    }, 30000); // Increased timeout for similarity search
  });
}

// NEW: Additional utility handlers for the frontend

// Get image statistics
ipcMain.handle("get-image-stats", async () => {
  try {
    await db.read();
    const images = db.data?.images || [];

    const stats = {
      total: images.length,
      byType: {},
      totalSize: 0,
      avgSize: 0,
      dateRange: {
        oldest: null,
        newest: null,
      },
    };

    if (images.length === 0) return stats;

    images.forEach((img) => {
      // Count by file type
      const type = img.filetype?.toLowerCase() || "unknown";
      stats.byType[type] = (stats.byType[type] || 0) + 1;

      // Total size
      stats.totalSize += img.file_size || 0;

      // Date range
      const created = new Date(img.DateCreated);
      if (
        !stats.dateRange.oldest ||
        created < new Date(stats.dateRange.oldest)
      ) {
        stats.dateRange.oldest = img.DateCreated;
      }
      if (
        !stats.dateRange.newest ||
        created > new Date(stats.dateRange.newest)
      ) {
        stats.dateRange.newest = img.DateCreated;
      }
    });

    stats.avgSize = stats.totalSize / images.length;

    return stats;
  } catch (error) {
    console.error("Error getting image stats:", error);
    return { total: 0, byType: {}, totalSize: 0, avgSize: 0, dateRange: {} };
  }
});


// DELETE IMAGES FROM DB AND HASHED DB SUPPORTS BOTH BULK / SINGLE
ipcMain.handle("remove-image-from-db", async (event, imagePaths) => {
  try {
    await db.read();

    if (!db.data?.images) return false;

    // Normalize paths to array
    const paths = Array.isArray(imagePaths) ? imagePaths : [imagePaths];

    // Load hash database
    const hashDB = loadHashDB();

    const hexValuesToDelete = [];
    const remainingImages = [];

    for (const img of db.data.images) {
      if (paths.includes(img.path)) {
        if (img.hex) hexValuesToDelete.push(img.hex);
      } else {
        remainingImages.push(img);
      }
    }

    db.data.images = remainingImages;

    // Remove from hashDB
    for (const hex of hexValuesToDelete) {
      if (hashDB[hex]) {
        delete hashDB[hex];
      }
    }

    const deletionHappened = hexValuesToDelete.length > 0;

    if (deletionHappened) {
      await db.write();      
      saveHashDB(hashDB);     
    }

    return deletionHappened;
  } catch (error) {
    console.error("Error removing image from database:", error);
    return false;
  }
});



// NEW: Check if folder is being tracked
ipcMain.handle("check-folder-tracking", async (event, folderPath) => {
  try {
    await db.read();
    const images = db.data?.images || [];

    const trackedImages = images.filter((img) =>
      img.path.startsWith(folderPath)
    );

    return {
      isTracked: trackedImages.length > 0,
      imageCount: trackedImages.length,
      lastUpdated:
        trackedImages.length > 0
          ? Math.max(
              ...trackedImages.map((img) => new Date(img.timestamp).getTime())
            )
          : null,
    };
  } catch (error) {
    console.error("Error checking folder tracking:", error);
    return { isTracked: false, imageCount: 0, lastUpdated: null };
  }
});

//load model initially
ipcMain.handle("load-model", async (event, force) => {
  console.log("STARTS", needToLoad, force);
  const scriptPath = returnPath("loadmodelinitially.py");

  try {
    return new Promise((resolve, reject) => {
      if (!needToLoad && force==undefined) {
        console.log("Model Download completed successfully");
        event.sender.send("model-download-complete", { success: true });
        resolve({ success: true });
        return;
      }

      const pythonProcess = spawn(pythonScriptPath, [scriptPath], {
        cwd: path.dirname(scriptPath),
      });
      processes[2] = pythonProcess;

      let stderr = "";

      pythonProcess.stdout.on("data", (data) => {
        let stdout ="";
        stdout+= data.toString();
        if (stdout.includes("PROGRESS:") && stdout.includes("|NAME:")) {
          const [progressPart, namePart] = stdout.split("|NAME:");
          const percentage = progressPart.replace("PROGRESS:", "").trim();
          const name = namePart.trim();

          console.log(`Progress: ${percentage}%, Model: ${name}`);
          event.sender.send("model-download-progress", percentage);
          event.sender.send("model-download-name", name);
        }

        if (stdout.includes("ALL_DOWNLOADS_COMPLETED")) {
          console.log("Download completed successfully");
          event.sender.send("model-download-complete", { success: true });
          resolve({ success: true });
        }
      });

      pythonProcess.stderr.on("data", (data) => {
        stderr += data.toString();
        console.error("Python stderr:", data.toString());
      });

      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          const errorMessage = `Python process exited with code ${code}: ${stderr}`;
          event.sender.send("model-download-error", errorMessage);
          reject(new Error(errorMessage));
        } 
      });

      pythonProcess.on("error", (error) => {
        console.error("Python process error:", error);
        const errorMessage = `Failed to start Python process: ${error.message}`;
        event.sender.send("model-download-error", errorMessage);
        reject(new Error(errorMessage));
      });
    });
  } catch (error) {
    console.error("Error in load-model handler:", error);
    event.sender.send("model-download-error", error.message);
    return { success: false, error: error.message };
  }
});
//load model initially
ipcMain.handle("load-module", async (event,force) => {
  console.log("STARTS",needToLoadModules,force);

  // process.resourcesPath
  const scriptPath = returnPath("loadModules.py");

  try {
    // RETURN the Promise - this was missing!
    return new Promise((resolve, reject) => {
          if(!needToLoadModules && force==undefined)
  {
        console.log("Module Download completed successfully");
          event.sender.send("module-completed", { success: true });
          resolve({ success: true });
          return ;
  }
      const pythonProcess = spawn(pythonScriptPath, [scriptPath], {
        cwd: path.dirname(scriptPath),
      });
processes[4]=pythonProcess
      let stderr = "";

      pythonProcess.stdout.on("data", (data) => {
        let stdoutBuffer = "";
        stdoutBuffer += data.toString();
        console.log(stdoutBuffer, "snsns");

        // Process each line separately to handle multiple progress updates

        event.sender.send("terminal-log", stdoutBuffer);
        if (stdoutBuffer.includes("ALL_DOWNLOADS_COMPLETED")) {
          console.log("Download completed successfully");
          event.sender.send("module-completed", { success: true });
          resolve({ success: true });
        }
      });

      pythonProcess.stderr.on("data", (data) => {
        stderr += data.toString();
      });
      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          const errorMessage = `Python process exited with code ${code}: ${stderr}`;
          event.sender.send("module-download-error", errorMessage);
          reject(new Error(errorMessage));
        }
      });

      pythonProcess.on("error", (error) => {
        console.error("Python process error:", error);
        const errorMessage = `Failed to start Python process: ${error.message}`;
      });
    });
  } catch (error) {
    console.error("Error in load-module handler:", error);
    event.sender.send("module-download-error", error.message);
    return { success: false, error: error.message };
  }
});

console.log("Backend initialization complete. Ready to process images!");
