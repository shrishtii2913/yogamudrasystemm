/*
 * =============================================================================
 * main.cpp — Yoga Mudra Detection System (C++ / OpenCV Component)
 * =============================================================================
 * ACADEMIC PROJECT: Hybrid C++ & Python Yoga Mudra Detection System
 *
 * Component Role:
 *   Reads mudra detection results written by the Python process to a shared
 *   JSON file, then uses OpenCV to:
 *     1. Display a rich real-time feedback overlay on the live camera feed
 *     2. Draw a confidence bar graph
 *     3. Log every detected pose change to logs/cpp_log.csv
 *     4. Write a summary table to logs/session_summary.txt on exit
 *
 * Libraries Used:
 *   - OpenCV 4.x : Camera capture, drawing primitives, window management
 *   - nlohmann/json (header-only) : Parse the IPC JSON from Python
 *   - Standard C++ : File I/O, chrono, string manipulation
 *
 * IPC Mechanism:
 *   Python writes  → shared/output.json
 *   C++ reads      ← shared/output.json
 *   The file is re-read every frame. A timestamp field lets C++ detect
 *   staleness (Python might have stalled).
 * =============================================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include "json.hpp"          // nlohmann/json — single-header, bundled in repo

using namespace std;
using namespace cv;
using json = nlohmann::json;
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
static const string JSON_PATH    = "shared/output.json";
static const string CSV_LOG      = "logs/cpp_log.csv";
static const string SUMMARY_PATH = "logs/session_summary.txt";

// OpenCV colour constants (BGR)
static const Scalar COL_GREEN   (  0, 220,  80);
static const Scalar COL_YELLOW  (  0, 220, 220);
static const Scalar COL_CYAN    (220, 200,   0);
static const Scalar COL_RED     ( 50,  50, 220);
static const Scalar COL_WHITE   (240, 240, 240);
static const Scalar COL_DARK    ( 25,  25,  25);
static const Scalar COL_ACCENT  (180, 120,   0);   // amber
static const Scalar COL_PANEL   ( 18,  18,  30);   // near-black panel

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY: current wall-clock as a formatted string (for logging)
// ─────────────────────────────────────────────────────────────────────────────
string nowString() {
    auto t  = chrono::system_clock::now();
    time_t tt = chrono::system_clock::to_time_t(t);
    tm local{};
#ifdef _WIN32
    localtime_s(&local, &tt);
#else
    localtime_r(&tt, &local);
#endif
    ostringstream ss;
    ss << put_time(&local, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY: draw a filled rounded rectangle (OpenCV lacks a native one)
// ─────────────────────────────────────────────────────────────────────────────
void drawRoundedRect(Mat& img, Rect rect, Scalar color, int radius, int thickness = -1) {
    int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
    // Filled corners with circles then rectangles
    if (thickness < 0) {
        rectangle(img, Point(x + radius, y),        Point(x + w - radius, y + h), color, -1);
        rectangle(img, Point(x, y + radius),         Point(x + w, y + h - radius), color, -1);
        circle(img, Point(x + radius,     y + radius),     radius, color, -1);
        circle(img, Point(x + w - radius, y + radius),     radius, color, -1);
        circle(img, Point(x + radius,     y + h - radius), radius, color, -1);
        circle(img, Point(x + w - radius, y + h - radius), radius, color, -1);
    } else {
        rectangle(img, Point(x + radius, y),         Point(x + w - radius, y + h), color, thickness);
        rectangle(img, Point(x, y + radius),          Point(x + w, y + h - radius), color, thickness);
        circle(img, Point(x + radius,     y + radius),     radius, color, thickness);
        circle(img, Point(x + w - radius, y + radius),     radius, color, thickness);
        circle(img, Point(x + radius,     y + h - radius), radius, color, thickness);
        circle(img, Point(x + w - radius, y + h - radius), radius, color, thickness);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OVERLAY DRAWING: left-side info panel
//   Draws pose name, confidence bar, description, FPS badge
// ─────────────────────────────────────────────────────────────────────────────
void drawInfoPanel(Mat& frame, const string& pose, float confidence,
                   const string& description, float fps, bool handDetected,
                   const string& feedback) {

    int fW = frame.cols, fH = frame.rows;

    // Semi-transparent dark panel on the left
    Mat overlay = frame.clone();
    drawRoundedRect(overlay, Rect(8, 8, 300, 220), COL_PANEL, 12);
    addWeighted(overlay, 0.82, frame, 0.18, 0, frame);

    // ── Pose name ──────────────────────────────────────────────────────────
    Scalar poseColor = handDetected ? COL_GREEN : COL_RED;
    putText(frame, pose,
            Point(20, 48), FONT_HERSHEY_DUPLEX, 0.85, poseColor, 2);

    // ── Confidence bar ─────────────────────────────────────────────────────
    putText(frame, "Confidence", Point(20, 75),
            FONT_HERSHEY_SIMPLEX, 0.48, COL_WHITE, 1);

    int barX = 20, barY = 82, barW = 270, barH = 16;
    // Background track
    rectangle(frame, Point(barX, barY), Point(barX + barW, barY + barH),
              Scalar(60, 60, 60), -1);
    // Filled portion — colour shifts green → yellow → red as confidence drops
    int filled = static_cast<int>(barW * confidence);
    Scalar barColor = confidence > 0.7 ? COL_GREEN
                    : confidence > 0.4 ? COL_YELLOW
                    : COL_RED;
    if (filled > 0)
        rectangle(frame, Point(barX, barY), Point(barX + filled, barY + barH),
                  barColor, -1);
    // Border
    rectangle(frame, Point(barX, barY), Point(barX + barW, barY + barH),
              COL_WHITE, 1);

    // Percentage label
    string confStr = to_string(static_cast<int>(confidence * 100)) + "%";
    putText(frame, confStr, Point(barX + barW + 6, barY + 13),
            FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1);

    // ── Description ────────────────────────────────────────────────────────
    // Word-wrap at ~38 chars
    string desc = description.size() > 38
                ? description.substr(0, 35) + "..."
                : description;
    putText(frame, desc, Point(20, 120),
            FONT_HERSHEY_SIMPLEX, 0.44, COL_CYAN, 1);

    // ── Feedback message ───────────────────────────────────────────────────
    putText(frame, feedback, Point(20, 150),
            FONT_HERSHEY_SIMPLEX, 0.5, COL_YELLOW, 1);

    // ── FPS badge ──────────────────────────────────────────────────────────
    string fpsStr = "FPS: " + to_string(static_cast<int>(fps));
    putText(frame, fpsStr, Point(20, 185),
            FONT_HERSHEY_SIMPLEX, 0.5, COL_ACCENT, 1);

    // ── Hand status dot ────────────────────────────────────────────────────
    Scalar dotColor = handDetected ? COL_GREEN : Scalar(80, 80, 200);
    circle(frame, Point(fW - 20, 20), 9, dotColor, -1);
    putText(frame, handDetected ? "HAND" : "NONE",
            Point(fW - 65, 24), FONT_HERSHEY_SIMPLEX, 0.4, COL_WHITE, 1);

    // ── Watermark ──────────────────────────────────────────────────────────
    putText(frame, "C++ Visualizer | OpenCV",
            Point(fW - 220, fH - 10),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(100, 100, 100), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// FEEDBACK ENGINE
//   Returns a short instructional string based on current confidence and pose.
// ─────────────────────────────────────────────────────────────────────────────
string generateFeedback(const string& pose, float confidence, bool handDetected) {
    if (!handDetected) return ">> Show your hand to the camera";
    if (pose == "No Pose") return ">> Try a mudra gesture";
    if (confidence > 0.85) return ">> Perfect form!";
    if (confidence > 0.60) return ">> Good — hold steady";
    if (confidence > 0.35) return ">> Adjust finger position";
    return ">> Keep refining the pose";
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV LOGGER
//   Writes a new row each time the detected pose changes.
// ─────────────────────────────────────────────────────────────────────────────
class CsvLogger {
public:
    explicit CsvLogger(const string& path) : path_(path) {
        fs::create_directories(fs::path(path).parent_path());
        out_.open(path, ios::app);
        if (!out_.is_open())
            cerr << "[WARN] Cannot open CSV log: " << path << "\n";
        else {
            // Write header only if the file was just created (empty)
            out_.seekp(0, ios::end);
            if (out_.tellp() == 0)
                out_ << "Timestamp,Pose,Confidence,FPS\n";
        }
    }

    void log(const string& pose, float confidence, float fps) {
        if (!out_.is_open()) return;
        out_ << nowString() << ","
             << pose        << ","
             << fixed << setprecision(4) << confidence << ","
             << setprecision(1) << fps << "\n";
        out_.flush();
    }

    ~CsvLogger() { if (out_.is_open()) out_.close(); }

private:
    string   path_;
    ofstream out_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SESSION STATISTICS
//   Accumulates per-pose frame counts for summary on exit.
// ─────────────────────────────────────────────────────────────────────────────
struct SessionStats {
    map<string, int>   poseCounts;    // pose → frames detected
    map<string, float> poseConfSum;   // pose → cumulative confidence
    int                totalFrames = 0;

    void record(const string& pose, float conf) {
        poseCounts[pose]++;
        poseConfSum[pose] += conf;
        totalFrames++;
    }

    void writeSummary(const string& path) const {
        fs::create_directories(fs::path(path).parent_path());
        ofstream f(path);
        if (!f.is_open()) return;

        f << "==============================================\n";
        f << "  Yoga Mudra Session Summary — " << nowString() << "\n";
        f << "==============================================\n";
        f << left << setw(20) << "Pose"
          << setw(10) << "Frames"
          << setw(10) << "Avg Conf"
          << "Share\n";
        f << string(52, '-') << "\n";

        for (const auto& [pose, cnt] : poseCounts) {
            float avgConf = (cnt > 0) ? poseConfSum.at(pose) / cnt : 0.f;
            float share   = totalFrames > 0
                          ? 100.f * cnt / totalFrames : 0.f;
            f << left << setw(20) << pose
              << setw(10) << cnt
              << setw(10) << fixed << setprecision(3) << avgConf
              << fixed << setprecision(1) << share << "%\n";
        }
        f << string(52, '-') << "\n";
        f << "Total frames processed: " << totalFrames << "\n";
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// JSON READER
//   Attempts to parse output.json. Returns false on any parse failure so
//   the display loop keeps the last valid reading rather than crashing.
// ─────────────────────────────────────────────────────────────────────────────
struct DetectionResult {
    string pose         = "No Pose";
    float  confidence   = 0.f;
    string description  = "Waiting for Python...";
    float  fps          = 0.f;
    bool   handDetected = false;
    double timestamp    = 0.0;
};

bool readJson(const string& path, DetectionResult& out) {
    ifstream f(path);
    if (!f.is_open()) return false;
    try {
        json j;
        f >> j;
        out.pose        = j.value("pose",          "No Pose");
        out.confidence  = j.value("confidence",    0.f);
        out.description = j.value("description",   "");
        out.fps         = j.value("fps",           0.f);
        out.handDetected= j.value("hand_detected", false);
        out.timestamp   = j.value("timestamp",     0.0);
        return true;
    } catch (...) {
        return false;  // malformed / partial write — keep previous result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    cout << "=== Yoga Mudra Detection System — C++ Visualizer ===\n";
    cout << "Watching: " << JSON_PATH << "\n";
    cout << "Press ESC to exit.\n\n";

    // ── Camera ────────────────────────────────────────────────────────────
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Cannot open camera. Is it in use by Python?\n"
             << "        The C++ component can still display JSON data\n"
             << "        but needs its own camera handle for the overlay.\n";
        // We'll run in JSON-only display mode
    }
    cap.set(CAP_PROP_FRAME_WIDTH,  640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // ── Subsystems ────────────────────────────────────────────────────────
    CsvLogger    logger(CSV_LOG);
    SessionStats stats;

    DetectionResult current;
    string          lastLoggedPose = "";
    int             frameIndex     = 0;

    // ── Display loop ──────────────────────────────────────────────────────
    while (true) {
        Mat frame;
        if (cap.isOpened()) {
            cap >> frame;
            if (frame.empty()) {
                frame = Mat(480, 640, CV_8UC3, COL_DARK);
            }
        } else {
            // Fallback canvas when camera is unavailable
            frame = Mat(480, 640, CV_8UC3, COL_DARK);
            putText(frame, "Camera unavailable — JSON display mode",
                    Point(50, 240), FONT_HERSHEY_SIMPLEX, 0.6, COL_RED, 1);
        }

        // ── Read IPC JSON from Python ─────────────────────────────────────
        readJson(JSON_PATH, current);   // silently keeps previous on failure

        // ── Log on pose change ────────────────────────────────────────────
        if (current.pose != lastLoggedPose) {
            logger.log(current.pose, current.confidence, current.fps);
            cout << "[" << nowString() << "] Pose changed → "
                 << current.pose
                 << " (conf=" << fixed << setprecision(3) << current.confidence << ")\n";
            lastLoggedPose = current.pose;
        }

        stats.record(current.pose, current.confidence);
        frameIndex++;

        // ── Generate real-time feedback ───────────────────────────────────
        string feedback = generateFeedback(current.pose,
                                           current.confidence,
                                           current.handDetected);

        // ── Draw overlay ──────────────────────────────────────────────────
        drawInfoPanel(frame,
                      current.pose,
                      current.confidence,
                      current.description,
                      current.fps,
                      current.handDetected,
                      feedback);

        imshow("Yoga Mudra — C++ Visualizer", frame);

        // ESC to quit
        if (waitKey(30) == 27) break;
    }

    // ── Teardown ──────────────────────────────────────────────────────────
    stats.writeSummary(SUMMARY_PATH);
    cout << "\n=== Session complete. Summary written to: "
         << SUMMARY_PATH << " ===\n";
    if (cap.isOpened()) cap.release();
    destroyAllWindows();
    return 0;
}
