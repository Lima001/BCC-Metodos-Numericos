#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

// Parameters (standard Lowe values)
const int NUM_INTERVALS = 3;
const float SIGMA = 1.6f;
const float IN_SIGMA = 0.5f;
const float CONTRAST_THRESHOLD = 0.04f;
const float EDGE_THRESHOLD = 10.0f;

// Descriptor parameters
const int DESCRIPTOR_WIDTH = 4;
const int DESCRIPTOR_HIST_BINS = 8;

// Orientation parameters
const float ORI_SIG_FCTR = 1.5f;
const int ORI_HIST_BINS = 36;
const float ORI_PEAK_RATIO = 0.8f;

// Timing macros
#define TIME_SECTION_START(label) auto t_start_##label = high_resolution_clock::now();
#define TIME_SECTION_END(label) \
    auto t_end_##label = high_resolution_clock::now(); \
    cout << #label << " took " << duration_cast<duration<double>>(t_end_##label - t_start_##label).count() << "s\n";


vector<vector<Mat>> buildGaussianPyramid(const Mat& base, int octaves) {
    vector<vector<Mat>> pyr(octaves);
    float k = pow(2.0f, 1.0f / NUM_INTERVALS);
    Mat baseFloat;
    base.convertTo(baseFloat, CV_32F, 1.0 / 255.0);

    for (int o = 0; o < octaves; ++o) {
        pyr[o].resize(NUM_INTERVALS + 3);
        if (o == 0) {
            float sigma0 = sqrt(SIGMA*SIGMA - IN_SIGMA*IN_SIGMA);
            GaussianBlur(baseFloat, pyr[0][0], Size(), sigma0, sigma0);
        } else {
            resize(pyr[o - 1][NUM_INTERVALS], pyr[o][0], Size(), 0.5, 0.5, INTER_NEAREST);
        }
        for (int i = 1; i < NUM_INTERVALS + 3; ++i) {
            float sigma_prev = SIGMA * pow(k, i - 1);
            float sigma_total = SIGMA * pow(k, i);
            float sigma_step = sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev);
            GaussianBlur(pyr[o][i - 1], pyr[o][i], Size(), sigma_step, sigma_step);
        }
    }
    return pyr;
}

vector<vector<Mat>> buildDoGPyramid(const vector<vector<Mat>>& gauss_pyr) {
    vector<vector<Mat>> dog_pyr(gauss_pyr.size());
    for (size_t o = 0; o < gauss_pyr.size(); ++o) {
        dog_pyr[o].resize(NUM_INTERVALS + 2);
        for (int i = 0; i < NUM_INTERVALS + 2; ++i) {
            subtract(gauss_pyr[o][i + 1], gauss_pyr[o][i], dog_pyr[o][i]);
        }
    }
    return dog_pyr;
}

vector<tuple<int,int,int,int>> detectExtrema(const vector<vector<Mat>>& dog_pyr) {
    vector<tuple<int,int,int,int>> extrema;
    for (size_t o = 0; o < dog_pyr.size(); ++o) {
        for (int i = 1; i <= NUM_INTERVALS; ++i) {
            const Mat& curr = dog_pyr[o][i];
            for (int y = 1; y < curr.rows - 1; ++y) {
                for (int x = 1; x < curr.cols - 1; ++x) {
                    float val = curr.at<float>(y, x);
                    if (fabs(val) < 0.5f * CONTRAST_THRESHOLD / NUM_INTERVALS) continue;

                    bool is_max = true, is_min = true;
                    for (int s = -1; s <= 1; ++s) {
                        const Mat& img = dog_pyr[o][i + s];
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (s == 0 && dy == 0 && dx == 0) continue;
                                float neighbor = img.at<float>(y + dy, x + dx);
                                if (val <= neighbor) is_max = false;
                                if (val >= neighbor) is_min = false;
                                if (!is_max && !is_min) break;
                            }
                            if (!is_max && !is_min) break;
                        }
                        if (!is_max && !is_min) break;
                    }
                    if (is_max || is_min) {
                        extrema.emplace_back(o, i, y, x);
                    }
                }
            }
        }
    }
    return extrema;
}

bool localizeKeypoint(const vector<vector<Mat>>& dog_pyr, int& octave, int& interval, int& y, int& x, Point3f& offset) {
    const int max_iter = 5;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (interval < 1 || interval > NUM_INTERVALS || y < 1 || y >= dog_pyr[octave][0].rows - 1 || x < 1 || x >= dog_pyr[octave][0].cols - 1)
            return false;
        const Mat& curr = dog_pyr[octave][interval];
        const Mat& prev = dog_pyr[octave][interval - 1];
        const Mat& next = dog_pyr[octave][interval + 1];

        Vec3f g;
        g[0] = (curr.at<float>(y, x + 1) - curr.at<float>(y, x - 1)) * 0.5f;
        g[1] = (curr.at<float>(y + 1, x) - curr.at<float>(y - 1, x)) * 0.5f;
        g[2] = (next.at<float>(y, x) - prev.at<float>(y, x)) * 0.5f;
        Matx33f H;
        H(0,0) = curr.at<float>(y, x + 1) + curr.at<float>(y, x - 1) - 2 * curr.at<float>(y, x);
        H(1,1) = curr.at<float>(y + 1, x) + curr.at<float>(y - 1, x) - 2 * curr.at<float>(y, x);
        H(2,2) = next.at<float>(y, x) + prev.at<float>(y, x) - 2 * curr.at<float>(y, x);
        H(0,1) = H(1,0) = (curr.at<float>(y + 1, x + 1) - curr.at<float>(y + 1, x - 1) - curr.at<float>(y - 1, x + 1) + curr.at<float>(y - 1, x - 1)) * 0.25f;
        H(0,2) = H(2,0) = (next.at<float>(y, x + 1) - next.at<float>(y, x - 1) - prev.at<float>(y, x + 1) + prev.at<float>(y, x - 1)) * 0.25f;
        H(1,2) = H(2,1) = (next.at<float>(y + 1, x) - next.at<float>(y - 1, x) - prev.at<float>(y + 1, x) + prev.at<float>(y - 1, x)) * 0.25f;

        Vec3f offset_vec;
        solve(H, -g, offset_vec, DECOMP_LU);
        if (fabs(offset_vec[0]) < 0.5f && fabs(offset_vec[1]) < 0.5f && fabs(offset_vec[2]) < 0.5f) {
            offset = Point3f(offset_vec[0], offset_vec[1], offset_vec[2]);
            return true;
        }
        x += cvRound(offset_vec[0]);
        y += cvRound(offset_vec[1]);
        interval += cvRound(offset_vec[2]);
    }
    return false;
}

vector<KeyPoint> filterAndLocalizeKeypoints(const vector<vector<Mat>>& dog_pyr, const vector<tuple<int,int,int,int>>& extrema) {
    vector<KeyPoint> keypoints;
    float img_scale = 1.0f; // This is 1.0 because we doubled the image size at the start

    for (auto [o, i, r, c] : extrema) {
        int octave = o, interval = i, y = r, x = c;
        Point3f offset;
        if (!localizeKeypoint(dog_pyr, octave, interval, y, x, offset))
            continue;

        const Mat& curr_dog = dog_pyr[octave][interval];
        float contrast = curr_dog.at<float>(y, x) + 0.5f * (
            offset.x * (curr_dog.at<float>(y, x+1) - curr_dog.at<float>(y, x-1)) * 0.5f +
            offset.y * (curr_dog.at<float>(y+1, x) - curr_dog.at<float>(y-1, x)) * 0.5f +
            offset.z * (dog_pyr[octave][interval+1].at<float>(y, x) - dog_pyr[octave][interval-1].at<float>(y, x)) * 0.5f
        );
        if (fabs(contrast) < CONTRAST_THRESHOLD / NUM_INTERVALS)
            continue;

        float dxx = curr_dog.at<float>(y, x + 1) + curr_dog.at<float>(y, x - 1) - 2 * curr_dog.at<float>(y, x);
        float dyy = curr_dog.at<float>(y + 1, x) + curr_dog.at<float>(y - 1, x) - 2 * curr_dog.at<float>(y, x);
        float dxy = (curr_dog.at<float>(y + 1, x + 1) - curr_dog.at<float>(y + 1, x - 1) - curr_dog.at<float>(y - 1, x + 1) + curr_dog.at<float>(y - 1, x - 1)) * 0.25f;
        float tr_H = dxx + dyy;
        float det_H = dxx * dyy - dxy * dxy;
        if (det_H <= 0 || (tr_H * tr_H / det_H) >= ((EDGE_THRESHOLD + 1) * (EDGE_THRESHOLD + 1) / EDGE_THRESHOLD))
            continue;

        // Note: final coordinates must be scaled back to the original image size, not the doubled size
        float scale = SIGMA * pow(2.0f, octave + (interval + offset.z) / (float)NUM_INTERVALS);
        float x_final = (x + offset.x) * pow(2.0f, octave) * 0.5f; // Scale by 0.5
        float y_final = (y + offset.y) * pow(2.0f, octave) * 0.5f; // Scale by 0.5

        KeyPoint kp(Point2f(x_final, y_final), scale * 0.5f, -1.f);
        kp.octave = octave + (interval << 8);
        keypoints.push_back(kp);
    }
    return keypoints;
}

void computeGradMagOri(const Mat& img, float x, float y, float& mag, float& angle) {
    if (x < 1.f || x >= img.cols - 1 || y < 1.f || y >= img.rows - 1) {
        mag = 0.f; angle = 0.f;
        return;
    }
    float dx = (img.at<float>((int)y, (int)x+1) - img.at<float>((int)y, (int)x-1)) * 0.5f;
    float dy = (img.at<float>((int)y+1, (int)x) - img.at<float>((int)y-1, (int)x)) * 0.5f;
    mag = sqrt(dx * dx + dy * dy);
    angle = atan2(dy, dx);
    if (angle < 0) angle += 2 * CV_PI;
}

void computeOrientations(const vector<vector<Mat>>& gauss_pyr, vector<KeyPoint>& keypoints) {
    vector<KeyPoint> oriented_keypoints;
    for (const auto& kp : keypoints) {
        int octave = kp.octave & 0xFF;
        int interval = (kp.octave >> 8) & 0xFF;
        const Mat& img = gauss_pyr[octave][interval];
        float scale_factor = 1.0f / (1 << octave);

        // kp.pt is in original image coords. We are working on the doubled image pyramid.
        float kpx = kp.pt.x * 2.0f * scale_factor;
        float kpy = kp.pt.y * 2.0f * scale_factor;
        float sigma = ORI_SIG_FCTR * kp.size * 2.0f * scale_factor;
        int radius = cvRound(3.0f * sigma);

        vector<float> hist(ORI_HIST_BINS, 0.0f);
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                float sample_x = kpx + dx;
                float sample_y = kpy + dy;
                if (dx*dx + dy*dy > radius*radius || sample_x < 1 || sample_y < 1 || sample_x >= img.cols-1 || sample_y >= img.rows-1)
                    continue;
                float mag, angle;
                computeGradMagOri(img, sample_x, sample_y, mag, angle);
                float weight = exp(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
                int bin = cvFloor(ORI_HIST_BINS * angle / (2.0f * CV_PI)) % ORI_HIST_BINS;
                hist[bin] += weight * mag;
            }
        }

        // Smooth the histogram
        vector<float> smoothed_hist(ORI_HIST_BINS);
        for (int i = 0; i < ORI_HIST_BINS; ++i) {
            int prev = (i - 1 + ORI_HIST_BINS) % ORI_HIST_BINS;
            int next = (i + 1) % ORI_HIST_BINS;
            smoothed_hist[i] = (hist[prev] + hist[i] + hist[next]) / 3.0f;
        }

        float max_val = *max_element(smoothed_hist.begin(), smoothed_hist.end());
        for (int i = 0; i < ORI_HIST_BINS; ++i) {
            if (smoothed_hist[i] >= ORI_PEAK_RATIO * max_val) {
                // Parabolic interpolation for peak
                int prev = (i - 1 + ORI_HIST_BINS) % ORI_HIST_BINS;
                int next = (i + 1) % ORI_HIST_BINS;
                float y_prev = smoothed_hist[prev], y_peak = smoothed_hist[i], y_next = smoothed_hist[next];
                float denom = y_prev - 2.0f * y_peak + y_next;
                float offset = (abs(denom) > 1e-6) ? (0.5f * (y_prev - y_next) / denom) : 0.0f;
                
                KeyPoint new_kp = kp;
                new_kp.angle = (i + offset + 0.5f) * (360.f / ORI_HIST_BINS);
                if(new_kp.angle < 0) new_kp.angle += 360.f;
                if(new_kp.angle >= 360.f) new_kp.angle -= 360.f;
                oriented_keypoints.push_back(new_kp);
            }
        }
    }
    keypoints = oriented_keypoints;
}

Mat computeDescriptors(const vector<vector<Mat>>& gauss_pyr, const vector<KeyPoint>& keypoints) {
    Mat descriptors;
    for (const auto& kp : keypoints) {
        int octave = kp.octave & 0xFF;
        const Mat& img = gauss_pyr[octave][kp.octave >> 8];
        float scale_factor = 1.0f / (1 << octave);

        float kpx = kp.pt.x * 2.0f * scale_factor;
        float kpy = kp.pt.y * 2.0f * scale_factor;
        float angle_rad = kp.angle * (CV_PI / 180.f);
        float cos_t = cos(-angle_rad), sin_t = sin(-angle_rad);
        
        float hist_width = 3.0f * kp.size * 2.0f * scale_factor;
        int radius = (int)round(hist_width * sqrt(2.f) * (DESCRIPTOR_WIDTH + 1) * 0.5f);
        
        vector<float> hist(DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * DESCRIPTOR_HIST_BINS, 0.0f);
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                float rotated_x = (cos_t * dx - sin_t * dy) / hist_width;
                float rotated_y = (sin_t * dx + cos_t * dy) / hist_width;
                float r_bin = rotated_y + DESCRIPTOR_WIDTH / 2.0f - 0.5f;
                float c_bin = rotated_x + DESCRIPTOR_WIDTH / 2.0f - 0.5f;
                if (r_bin <= -1 || r_bin >= DESCRIPTOR_WIDTH || c_bin <= -1 || c_bin >= DESCRIPTOR_WIDTH)
                    continue;

                float mag, angle;
                computeGradMagOri(img, kpx + dx, kpy + dy, mag, angle);
                float weight = exp(-(rotated_x * rotated_x + rotated_y * rotated_y) / (0.5f * DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH));
                float angle_diff = angle - angle_rad;
                while (angle_diff < 0) angle_diff += 2 * CV_PI;
                while (angle_diff >= 2 * CV_PI) angle_diff -= 2 * CV_PI;
                float o_bin = angle_diff * DESCRIPTOR_HIST_BINS / (2 * CV_PI);
                
                int r0 = floor(r_bin), c0 = floor(c_bin), o0 = floor(o_bin);
                float dr = r_bin - r0, dc = c_bin - c0, do_ = o_bin - o0;
                for (int ri = 0; ri <= 1; ++ri) {
                    int r = r0 + ri;
                    if (r < 0 || r >= DESCRIPTOR_WIDTH) continue;
                    float wr = (ri == 0) ? 1.f - dr : dr;
                    for (int ci = 0; ci <= 1; ++ci) {
                        int c = c0 + ci;
                        if (c < 0 || c >= DESCRIPTOR_WIDTH) continue;
                        float wc = (ci == 0) ? 1.f - dc : dc;
                        for (int oi = 0; oi <= 1; ++oi) {
                            int o = (o0 + oi) % DESCRIPTOR_HIST_BINS;
                            float wo = (oi == 0) ? 1.f - do_ : do_;
                            hist[(r*DESCRIPTOR_WIDTH + c)*DESCRIPTOR_HIST_BINS + o] += weight * wr * wc * wo * mag;
                        }
                    }
                }
            }
        }
        float norm = 0;
        for (float val : hist) norm += val * val;
        norm = sqrt(norm);
        if (norm > 1e-6) for (float& val : hist) val /= norm;
        for (float& val : hist) val = min(val, 0.2f);
        norm = 0;
        for (float val : hist) norm += val * val;
        norm = sqrt(norm);
        if (norm > 1e-6) for (float& val : hist) val /= norm;
        Mat desc_row(1, DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * DESCRIPTOR_HIST_BINS, CV_32F, hist.data());
        descriptors.push_back(desc_row.clone());
    }
    return descriptors;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./manual_sift <image_path>\n";
        return -1;
    }

    Mat img = imread(argv[1], IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Error loading image\n";
        return -1;
    }

    // Create the base image for the pyramid as per Lowe's paper.
    // Double the image size before pyramid construction.
    Mat base_img;
    resize(img, base_img, Size(img.cols * 2, img.rows * 2), 0, 0, INTER_LINEAR);

    int octaves = static_cast<int>(floor(log2(min(base_img.cols, base_img.rows)))) - 3;
    if (octaves < 1) octaves = 1;

    TIME_SECTION_START(gaussian_pyramid)
    vector<vector<Mat>> gauss_pyr = buildGaussianPyramid(base_img, octaves);
    TIME_SECTION_END(gaussian_pyramid)

    TIME_SECTION_START(dog_pyramid)
    vector<vector<Mat>> dog_pyr = buildDoGPyramid(gauss_pyr);
    TIME_SECTION_END(dog_pyramid)

    TIME_SECTION_START(extrema_detection)
    vector<tuple<int,int,int,int>> extrema = detectExtrema(dog_pyr);
    cout << "Detected initial extrema: " << extrema.size() << endl;
    TIME_SECTION_END(extrema_detection)

    TIME_SECTION_START(keypoint_localization_and_filtering)
    vector<KeyPoint> keypoints = filterAndLocalizeKeypoints(dog_pyr, extrema);
    cout << "Localized & filtered keypoints: " << keypoints.size() << endl;
    TIME_SECTION_END(keypoint_localization_and_filtering)

    TIME_SECTION_START(orientation_assignment)
    computeOrientations(gauss_pyr, keypoints);
    cout << "Keypoints after orientation assignment: " << keypoints.size() << endl;
    TIME_SECTION_END(orientation_assignment)

    TIME_SECTION_START(descriptor_computation)
    Mat descriptors = computeDescriptors(gauss_pyr, keypoints);
    TIME_SECTION_END(descriptor_computation)

    Mat out;
    drawKeypoints(img, keypoints, out, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SIFT Keypoints", out);
    imwrite("sift_result_updated.png", out);
    waitKey();

    cout << "Descriptors computed: " << descriptors.rows << " x " << descriptors.cols << endl;

    return 0;
}
