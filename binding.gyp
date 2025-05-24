{
  "targets": [
    {
      "target_name": "opencv_addon",
      "sources": ["addon.cpp"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "./opencv/sources/include",
        "./opencv/build/include",
        "./opencv/build/opencv2",
        "./opencv/build/modules/core/include",
        "./opencv/build/modules/features2d/include",
        "./opencv/build/modules/flann/include",
        "./opencv/build/modules/imgproc/include",
        "./opencv/build/modules/ml/include",
        "./opencv_contrib-4.11.0/modules/xfeatures2d/include",
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "conditions": [
        ["OS=='win'", {
          "libraries": [
            "./opencv/build/lib/Release/opencv_world4110.lib"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "AdditionalIncludeDirectories": [
              ],
              "RuntimeLibrary": "MultiThreadedDLL"
            },
            "VCLinkerTool": {
              "AdditionalLibraryDirectories": [
              ]
            }
          },
          "copies": [
            {
              "destination": "<(module_root_dir)/build/Release",
              "files": [
                "./opencv/build/bin/Release/opencv_world4110.dll",
                "./opencv/build/bin/Release/opencv_videoio_ffmpeg4110_64.dll"
              ]
            }
          ]
        }]
      ]
    }
  ]
}