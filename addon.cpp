#include <napi.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

struct FeatureMatchResult {
    std::vector<cv::Point2f> corners;
    double confidence;
    int matchesCount;
};

Napi::Object FindFeatures(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    // Validate input
    if (info.Length() < 3 || !info[0].IsString() || !info[1].IsArray() || !info[2].IsNumber()) {
        Napi::TypeError::New(env, "Arguments: (string)sceneImagePath, (array)objectPaths, (number)minHessian")
            .ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }

    // Parse arguments
    std::string sceneImagePath = info[0].As<Napi::String>();
    Napi::Array objectPaths = info[1].As<Napi::Array>();
    double minHessian = info[2].As<Napi::Number>().DoubleValue();
    double ratioThreshold = info.Length() > 3 ? info[3].As<Napi::Number>().DoubleValue() : 0.75;

    // Load scene image
    cv::Mat imgScene = cv::imread(sceneImagePath, cv::IMREAD_GRAYSCALE);
    if (imgScene.empty()) {
        Napi::Error::New(env, "Failed to load scene image").ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }

    // Prepare result object
    Napi::Object result = Napi::Object::New(env);
    Napi::Array matches = Napi::Array::New(env);

    // Create SURF detector
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

    // Process each object image
    for (uint32_t t = 0; t < objectPaths.Length(); t++) {
        Napi::Value val = objectPaths[t];
        if (!val.IsString())
            continue;

        std::string objectPath = val.As<Napi::String>();
        fs::path p(objectPath);
        std::string filename = p.filename().string();

        Napi::Object objectResult = Napi::Object::New(env);

        try {
            // Load object image
            cv::Mat imgObject = cv::imread(objectPath, cv::IMREAD_GRAYSCALE);
            if (imgObject.empty()) {
                objectResult.Set("error", "Failed to load object image");
                matches[t] = objectResult;
                continue;
            }

            //-- Step 1: Detect the keypoints and compute descriptors
            std::vector<cv::KeyPoint> keypointsObject, keypointsScene;
            cv::Mat descriptorsObject, descriptorsScene;
            detector->detectAndCompute(imgObject, cv::noArray(), keypointsObject, descriptorsObject);
            detector->detectAndCompute(imgScene, cv::noArray(), keypointsScene, descriptorsScene);

            // Skip if not enough keypoints
            if (keypointsObject.empty() || keypointsScene.empty()) {
                objectResult.Set("matchesCount", 0);
                matches[t] = objectResult;
                continue;
            }

            //-- Step 2: Matching descriptor vectors with a FLANN based matcher
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(descriptorsObject, descriptorsScene, knnMatches, 2);

            //-- Filter matches using the Lowe's ratio test
            std::vector<cv::DMatch> goodMatches;
            for (size_t i = 0; i < knnMatches.size(); i++) {
                if (knnMatches[i][0].distance < ratioThreshold * knnMatches[i][1].distance) {
                    goodMatches.push_back(knnMatches[i][0]);
                }
            }

            // Skip if not enough good matches
            if (goodMatches.size() < 10) {
                objectResult.Set("matchesCount", (int)goodMatches.size());
                matches[t] = objectResult;
                continue;
            }

            //-- Localize the object
            std::vector<cv::Point2f> obj;
            std::vector<cv::Point2f> scene;

            for (size_t i = 0; i < goodMatches.size(); i++) {
                //-- Get the keypoints from the good matches
                obj.push_back(keypointsObject[goodMatches[i].queryIdx].pt);
                scene.push_back(keypointsScene[goodMatches[i].trainIdx].pt);
            }

            // Find homography
            cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);

            //-- Get the corners from the object image
            std::vector<cv::Point2f> objCorners(4);
            objCorners[0] = cv::Point2f(0, 0);
            objCorners[1] = cv::Point2f((float)imgObject.cols, 0);
            objCorners[2] = cv::Point2f((float)imgObject.cols, (float)imgObject.rows);
            objCorners[3] = cv::Point2f(0, (float)imgObject.rows);
            
            std::vector<cv::Point2f> sceneCorners(4);
            cv::perspectiveTransform(objCorners, sceneCorners, H);

            // Calculate confidence based on matches count and area
            double confidence = (double)goodMatches.size() / (keypointsObject.size() + keypointsScene.size()) * 2.0;
            confidence = std::min(1.0, confidence); // Cap at 1.0

            // Prepare result
            Napi::Array cornersArray = Napi::Array::New(env, 4);
            for (int i = 0; i < 4; i++) {
                Napi::Object point = Napi::Object::New(env);
                point.Set("x", sceneCorners[i].x);
                point.Set("y", sceneCorners[i].y);
                cornersArray[i] = point;
            }

            objectResult.Set("template", filename);
            objectResult.Set("corners", cornersArray);
            objectResult.Set("confidence", confidence);
            objectResult.Set("matchesCount", (int)goodMatches.size());
        }
        catch (const cv::Exception &e) {
            objectResult.Set("error", e.what());
        }

        matches[t] = objectResult;
    }

    result.Set("matches", matches);
    return result;
}

// Added new function to find multiple template matches
Napi::Object FindTemplates(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    // Validate input
    if (info.Length() < 4 || !info[0].IsString() || !info[1].IsArray() || !info[2].IsNumber() || !info[3].IsNumber()) {
        Napi::TypeError::New(env, "Arguments: (string)sceneImagePath, (array)objectPaths, (number)method, (number)threshold")
            .ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }

    // Parse arguments
    std::string sceneImagePath = info[0].As<Napi::String>();
    Napi::Array objectPaths = info[1].As<Napi::Array>();
    int method = info[2].As<Napi::Number>().Int32Value();
    double threshold = info[3].As<Napi::Number>().DoubleValue();

    // Load scene image
    cv::Mat imgScene = cv::imread(sceneImagePath, cv::IMREAD_GRAYSCALE);
    if (imgScene.empty()) {
        Napi::Error::New(env, "Failed to load scene image").ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }

    // Prepare result object
    Napi::Object result = Napi::Object::New(env);
    Napi::Array templateResults = Napi::Array::New(env);

    // Process each object image
    for (uint32_t t = 0; t < objectPaths.Length(); t++) {
        Napi::Value val = objectPaths[t];
        if (!val.IsString())
            continue;

        std::string objectPath = val.As<Napi::String>();
        fs::path p(objectPath);
        std::string filename = p.filename().string();

        Napi::Object singleTemplateResult = Napi::Object::New(env);
        singleTemplateResult.Set("template", filename);
        Napi::Array matchesArray = Napi::Array::New(env);
        int matchIndex = 0;

        try {
            // Load object image
            cv::Mat imgObject = cv::imread(objectPath, cv::IMREAD_GRAYSCALE);
            if (imgObject.empty()) {
                singleTemplateResult.Set("error", "Failed to load object image");
                templateResults[t] = singleTemplateResult;
                continue;
            }

            // Ensure scene is larger than template
            if (imgScene.cols < imgObject.cols || imgScene.rows < imgObject.rows) {
                 singleTemplateResult.Set("error", "Scene image is smaller than template image");
                 templateResults[t] = singleTemplateResult;
                 continue;
            }

            // Perform template matching
            cv::Mat result_map;
            cv::matchTemplate(imgScene, imgObject, result_map, method);

            // Find locations above the threshold
            // Also find the global maximum confidence for this template
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(result_map, &minVal, &maxVal, &minLoc, &maxLoc);

            singleTemplateResult.Set("maxConfidence", maxVal); // Add max confidence to result

            for (int i = 0; i < result_map.rows; i++) {
                for (int j = 0; j < result_map.cols; j++) {
                    double confidence = result_map.at<float>(i, j);
                    bool isMatch = false;
                    if (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED) {
                        isMatch = confidence <= threshold;
                    } else {
                        isMatch = confidence >= threshold;
                    }

                    if (isMatch) {
                        Napi::Object match = Napi::Object::New(env);
                        match.Set("x", j);
                        match.Set("y", i);
                        match.Set("width", imgObject.cols);
                        match.Set("height", imgObject.rows);
                        match.Set("confidence", confidence);
                        matchesArray[matchIndex++] = match;
                    }
                }
            }
             singleTemplateResult.Set("matches", matchesArray);
        }
        catch (const cv::Exception &e) {
            singleTemplateResult.Set("error", e.what());
        }
        templateResults[t] = singleTemplateResult;
    }

    result.Set("results", templateResults);
    return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("findFeatures", Napi::Function::New(env, FindFeatures));
    // Added new export for findTemplates
    exports.Set("findTemplates", Napi::Function::New(env, FindTemplates));
    return exports;
}

NODE_API_MODULE(feature_addon, Init)