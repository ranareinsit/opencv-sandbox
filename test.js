const addon = require('./build/Release/opencv_addon.node');
const path = require('path');
const fs = require('fs');
const sharp = require('sharp');

// Function to resize and save an image
function resize_and_save(inputPath, width, height, outputPath) {
  return sharp(inputPath)
    .resize(width, height, { fit: 'cover' }) // Use 'cover' because we want to keep the aspect ratio
    .toFile(outputPath);
}

// Function to crop, resize, and save an image
function crop_resize_and_save(inputPath, cropLeft, cropTop, cropWidth, cropHeight, resizeWidth, resizeHeight, outputPath) {
  return sharp(inputPath)
    .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
    .resize(resizeWidth, resizeHeight, { fit: 'fill' }) // Use 'fill' to stretch cropped area to target dimensions
    .toFile(outputPath);
}

// Helper function to calculate Intersection over Union (IoU)
function calculateIoU(box1, box2) {
  const xA = Math.max(box1.x, box2.x);
  const yA = Math.max(box1.y, box2.y);
  const xB = Math.min(box1.x + box1.width, box2.x + box2.width);
  const yB = Math.min(box1.y + box1.height, box2.y + box2.height);

  const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);

  const box1Area = box1.width * box1.height;
  const box2Area = box2.width * box2.height;
  const unionArea = box1Area + box2Area - intersectionArea;

  // Avoid division by zero
  if (unionArea === 0) {
    return 0;
  }

  return intersectionArea / unionArea;
}

// Function to perform Non-Maximum Suppression
function nonMaxSuppression(boxes, overlapThresh) {
  // Sort the boxes by their confidence scores in descending order
  boxes.sort((a, b) => b.confidence - a.confidence);

  const selectedBoxes = [];

  const overlapThreshDifferentTemplate = 0.5; // Increased IoU threshold for different templates

  while (boxes.length > 0) {
    // Select the box with the highest confidence
    const bestBox = boxes.shift();
    selectedBoxes.push(bestBox);

    // Remove boxes that significantly overlap with the bestBox
    boxes = boxes.filter(currentBox => {
      // Calculate the intersection area
      const iou = calculateIoU(bestBox, currentBox);

      // If the templates are the same, use the strict overlap threshold
      if (bestBox.template === currentBox.template) {
        return iou < overlapThresh;
      } else {
        // If templates are different, use the higher threshold, but also consider confidence ratio
        const confidenceRatio = bestBox.confidence / currentBox.confidence;
        // Suppress the lower-confidence box if it significantly overlaps a much higher-confidence box of a different template
        if (iou > overlapThreshDifferentTemplate && confidenceRatio > 1.5) { // Adjust the confidence ratio as needed
          return false; // Suppress the currentBox
        }
        // Otherwise, keep the box if IoU is below the higher threshold
        return iou < overlapThreshDifferentTemplate;
      }
    });
  }

  return selectedBoxes;
}

async function main() {
  try {
    const mainImagePath = path.resolve(__dirname, './images/target.png');

    // Read and print main image dimensions
    const mainImageMetadata = await sharp(mainImagePath).metadata();
    console.log(`Main image dimensions: ${mainImageMetadata.width}x${mainImageMetadata.height}`);

    const templates_raw = fs.readdirSync("./images/1080");
    const originalTemplateDir = path.resolve(__dirname, "./images"); // Use original directory
    const resizedTemplateDir = path.resolve(__dirname, "./temp_resized_templates");
    const croppedResizedTemplateDir = path.resolve(__dirname, "./temp_cropped_resized_templates");

    // Ensure the temporary directories exist and are empty
    if (fs.existsSync(resizedTemplateDir)) {
      fs.rmSync(resizedTemplateDir, { recursive: true, force: true });
    }
    fs.mkdirSync(resizedTemplateDir);

    if (fs.existsSync(croppedResizedTemplateDir)) {
      fs.rmSync(croppedResizedTemplateDir, { recursive: true, force: true });
    }
    fs.mkdirSync(croppedResizedTemplateDir);

    const resizedTemplates = [];

    console.log("Resizing templates...");
    // Resize templates and save to temporary directory for the first and second pass
    for (const templateName of templates_raw) {
      const originalTemplatePath = path.resolve(originalTemplateDir + "/1080", templateName); // Use 1080 subdirectory
      const resizedTemplatePath = path.resolve(resizedTemplateDir, templateName);
      // Resize to the determined size of icons in the grid
      await resize_and_save(originalTemplatePath, 50, 35, resizedTemplatePath);
      resizedTemplates.push(resizedTemplatePath);
    }
    console.log("Finished resizing templates.");

    let p90 = 0
    let p80 = 0
    let p70 = 0
    let p60 = 0

    console.log(`templates (resized for first/second pass):`, resizedTemplates.length);

    // --- First Pass ---
    console.log("\nRunning first pass...");
    console.time('templateMatchingFirstPass');
    const method = 5; // TM_CCOEFF_NORMED
    const threshold = 0.833; // good for learn.png
    // const threshold = 0.7;

    const firstPassTemplateMatchResults = addon.findTemplates(
      mainImagePath,
      resizedTemplates,
      method,
      threshold
    );
    console.timeEnd('templateMatchingFirstPass');

    // Store results from the first pass and identify not found templates
    const firstPassResults = [];
    const notFoundTemplatesRawFirstPass = [];

    console.log('Template Max Confidences (first pass):');
    firstPassTemplateMatchResults.results.forEach((templateResult, index) => {
      const originalTemplateName = templates_raw[index];

      if (templateResult.maxConfidence < 0.88) {
        console.log(`  ${originalTemplateName}: ${templateResult.maxConfidence}`);

      }

      if (templateResult.matches && templateResult.matches.length > 0) {
        templateResult.matches.forEach(match => {
          match.template = originalTemplateName; // Add original template name
          firstPassResults.push(match);
        });
      } else {
        notFoundTemplatesRawFirstPass.push(originalTemplateName);
      }
    });

    console.log('Found in first pass:', firstPassResults.length);
    console.log('Not found in first pass:', notFoundTemplatesRawFirstPass.length);

    const finalNotFoundTemplatesRaw = [...notFoundTemplatesRawFirstPass];

    // Combine results from all passes (using filtered third pass results)
    const allResults = [...firstPassResults];
    console.log('Total results before NMS:', allResults.length);

    // Apply Non-Maximum Suppression to combined results
    const iouThreshold = 0.2; // IoU threshold for NMS
    const nmsResults = nonMaxSuppression(allResults, iouThreshold);

    console.log('Total found rectangles after NMS:', nmsResults.length);

    // Validate and prepare overlays
    const overlays = nmsResults.map(({ confidence, x, y, width, height, template, method }, i) => {
      // Validate coordinates
      if ([x, y, width, height].some(Number.isNaN)) {
        console.warn('Invalid rectangle coordinates:', { x, y, width, height });
        return null;
      }

      // Include method name in the overlay text for debugging
      const displayTemplateName = template.replace('_png', '').slice(0, 5);
      const overlayText = `${displayTemplateName}`;

      const color = confidence > 0.9 ? "#38E6FF" : confidence > 0.8 ? "#8F0CB6" : confidence > 0.7 ? "#C03D0C" : "#E1625B"
      const rect_width = confidence > 0.9 ? 3 : confidence > 0.8 ? 2 : confidence > 0.7 ? 2 : 3

      if (confidence >= 0.9) p90++
      if (confidence >= 0.8 && confidence < 0.9) p80++
      if (confidence >= 0.7 && confidence < 0.8) p70++
      if (confidence <= 0.6) p60++

      return {
        input: Buffer.from(`
        <svg 
          width="${width}"
          height="${height}"
        >
          <rect
            x="0"
            y="0"
            width="${width}"
            height="${height}"
            stroke="${color}"
            stroke-width="${rect_width}"
            fill="none"
          />
          <text
            x="1%"
            y="17%"
            font-family="monospace"
            font-size="16"
            fill="red"
            text-anchor="start"
            dominant-baseline="middle"
          >
            ${overlayText}
          </text>
        </svg>
      `),
        top: Math.floor(y),
        left: Math.floor(x),
      }
    }).filter(overlay => overlay !== null); // Filter out invalid overlays

    console.log(`Drawing ${overlays.length} valid rectangles`);

    // Composite rectangles onto the image
    await sharp(mainImagePath)
      .composite(overlays)
      .toFile('output_with_rectangles.png');

    console.log('Image with rectangles saved to output_with_rectangles.png');
    console.log(`Confidence level 90>=:${p90}  <80>=:${p80}  <70>=:${p70}  60<=:${p60}`);

    // Save JSON results
    fs.writeFileSync('results.json', JSON.stringify(
      nmsResults, // Save NMS results
      null, 2
    ));

    console.log(`not found: `,
      finalNotFoundTemplatesRaw.length
    );

  } catch (e) {
    console.error('Error:', e);
  } finally {
    // Clean up the temporary resized templates directory
    const resizedTemplateDir = path.resolve(__dirname, "./temp_resized_templates");
    if (fs.existsSync(resizedTemplateDir)) {
      fs.rmSync(resizedTemplateDir, { recursive: true, force: true });
      console.log("Cleaned up temporary resized templates.");
    }
    const croppedResizedTemplateDir = path.resolve(__dirname, "./temp_cropped_resized_templates");
    if (fs.existsSync(croppedResizedTemplateDir)) {
      fs.rmSync(croppedResizedTemplateDir, { recursive: true, force: true });
      console.log("Cleaned up temporary cropped and resized templates.");
    }
  }
}

main();