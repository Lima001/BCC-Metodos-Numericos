/*
    To compile:
    g++ sift.cpp `pkg-config opencv4 --cflags --libs`

    TODO: 
        VERIFICAR NORMALIZAÇÃO DOS PIXELS PARA FICAREM NO INTERVALO [0,1]
        IMPLEMENTAR CONVERSÃO ENTRE ESCALAS DE CORES - RGB  E GRAYSCALE
*/

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include <iostream>

/*
    Constants
*/
const int MAX_REFINEMENT_ITERS = 5;
const float MIN_SIGMA = 0.8;
const float MIN_PIX_DIST = 0.5;
const float SIGMA_IN = 0.5;
const int N_OCT = (int)(std::round(std::log(256) / std::log(2) - 1));
const int N_SPO = 3;
const float C_DOG = 0.015;
const float C_EDGE = 10;

/*
    Auxiliar structs
*/
typedef struct
{
    int i;
    int j;
    int octave;
    int scale;
    double x;
    double y;
    double sigma;
    double extremum_val;
} Keypoint;

typedef struct
{
    int n_octaves;
    int n_scales;
    std::vector<std::vector<cv::Mat>> octaves;
} ScaleSpacePyramid;

double getPixel(const cv::Mat &img, int x, int y)
{
    if (x < 0)
        x = 0;
    if (x >= img.size().width)
        x = img.size().width - 1;
    if (y < 0)
        y = 0;
    if (y >= img.size().height)
        y = img.size().height - 1;

    return img.at<double>(x, y);
}

/*
    Interpolation
*/
double linearInterpolation(double x, double x0, double y0, double x1, double y1)
{
    double u = (x - x0) / (x1 - x0);
    return (1 - u) * y0 + u * y1;
}

double bilinearInterpolation(double x, double y, double x0, double y0, double x1, double y1, double f00, double f01, double f10, double f11)
{
    double r1 = linearInterpolation(x, x0, f00, x1, f10);
    double r2 = linearInterpolation(x, x0, f01, x1, f11);

    return linearInterpolation(y, y0, r1, y1, r2);
}

/*
    Resize
*/
cv::Mat resize(cv::Mat *image, double scale = 1)
{
    cv::Mat out = cv::Mat((int)image->size().height * scale, (int)image->size().width * scale, CV_64FC1);

    double value;

    for (int i = 0; i < out.size().width; i++)
    {
        for (int j = 0; j < out.size().height; j++)
        {

            if (std::fmod(i, scale) == 0 && std::fmod(j, scale) == 0)
                value = image->at<double>(i / scale, j / scale);

            else
            {
                double x1 = std::ceil(i / scale);
                double y1 = std::ceil(j / scale);
                double x0 = std::min(x1 - 1, std::floor(i / scale));
                double y0 = std::min(y1 - 1, std::floor(j / scale));

                if (x1 > image->size().width - 1 || y1 > image->size().height - 1)
                    value = image->at<double>(std::floor(i / scale), std::floor(j / scale));

                else
                {
                    double f00 = image->at<double>(x0, y0);
                    double f01 = image->at<double>(x0, y1);
                    double f10 = image->at<double>(x1, y0);
                    double f11 = image->at<double>(x1, y1);

                    value = bilinearInterpolation(i / scale, j / scale, x0, y0, x1, y1, f00, f01, f10, f11);
                }
            }
            out.at<double>(i, j) = value;
        }
    }
    return out;
}

/*
    Gaussian Blur
*/

double **generateGaussianKernel(int size, double sigma)
{

    assert(size % 2 != 0);

    int radius = (size - 1) / 2;
    double **kernel = new double *[size];

    for (int i = 0; i < size; i++)
        kernel[i] = new double[size];

    double a = 1.0 / (std::sqrt(2 * M_PI) * sigma);
    double b = 2.0 * std::pow(sigma, 2);

    double sum = 0;

    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            kernel[i + radius][j + radius] = a * std::pow(M_E, (-(i * i + j * j) / b));
            sum += kernel[i + radius][j + radius];
        }
    }

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            kernel[i][j] = kernel[i][j] / sum;

    /*for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            std::cout << kernel[i][j] << "; ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;*/
    return kernel;
}

cv::Mat blurImage(cv::Mat *image, double sigma = std::sqrt(2))
{
    //int size = (int)(std::round(sigma*6+1))|1;
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    
    double **kernel = generateGaussianKernel(size, sigma);
    int radius = (size - 1) / 2;

    cv::Mat out = cv::Mat(image->size().height, image->size().width, CV_64FC1);

    double value;
    for (int i = 0; i < out.size().width; i++)
    {
        for (int j = 0; j < out.size().height; j++)
        {
            value = 0;

            for (int ki = -radius; ki <= radius; ki++)
            {
                for (int kj = -radius; kj <= radius; kj++)
                {
                    if (i + ki < 0 || i + ki >= out.size().width || j + kj < 0 || j + kj >= out.size().height)
                        continue;

                    value += image->at<double>(i + ki, j + kj) * kernel[ki + radius][kj + radius];
                }
            }
            out.at<double>(i, j) = value;
        }
    }

    for (int i = 0; i < size; i++)
        delete kernel[i];
    delete[] kernel;

    return out;
}

ScaleSpacePyramid generateGaussianPyramid(cv::Mat *image, int n_octaves, int n_scales, double min_sigma)
{
    double base_sigma = min_sigma / MIN_PIX_DIST;
    cv::Mat base_image = resize(image, 2);
    double sigma_diff = std::sqrt(base_sigma * base_sigma - 1.0);
    base_image = blurImage(&base_image, sigma_diff);

    int images_per_octave = n_scales + 3;

    double k = std::pow(2, 1.0 / n_scales);
    std::vector<double> sigma_vals{base_sigma};
    for (int i = 1; i < images_per_octave; i++)
    {
        double sigma_prev = base_sigma * std::pow(k, i - 1);
        double sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev));
    }

    ScaleSpacePyramid pyramid;
    pyramid.n_octaves = n_octaves;
    pyramid.n_scales = images_per_octave;
    pyramid.octaves = std::vector<std::vector<cv::Mat>>(n_octaves);

    for (int i = 0; i < n_octaves; i++)
    {
        pyramid.octaves[i].reserve(images_per_octave);
        pyramid.octaves[i].push_back(std::move(base_image));

        for (int j = 1; j < (int)sigma_vals.size(); j++)
        {
            cv::Mat prev_image = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(blurImage(&prev_image, sigma_vals[j]));
        }

        cv::Mat next_base_image = pyramid.octaves[i][images_per_octave - 3];
        base_image = resize(&next_base_image, 0.5);
    }

    return pyramid;
}

/*
    DoG
*/
cv::Mat subtractImages(cv::Mat *image1, cv::Mat *image2)
{
    assert(image1->size().height == image2->size().height && image1->size().width == image2->size().width);

    cv::Mat out = cv::Mat(image1->size().height, image1->size().width, CV_64FC1);

    for (int i = 0; i < out.size().width; i++)
    {
        for (int j = 0; j < out.size().height; j++)
        {
            double value = image1->at<double>(i, j) - image2->at<double>(i, j);
            // u_char value = uchar(std::abs((int)image1->at<double>(i,j) - (int)image2->at<double>(i,j)));
            out.at<double>(i, j) = value;
        }
    }
    return out;
}

ScaleSpacePyramid generateDogPyramid(ScaleSpacePyramid *pyramid)
{
    ScaleSpacePyramid dog_pyramid = {
        pyramid->n_octaves,
        pyramid->n_scales - 1,
        std::vector<std::vector<cv::Mat>>(pyramid->n_octaves)
    };

    for (int i = 0; i < dog_pyramid.n_octaves; i++)
    {
        dog_pyramid.octaves[i].reserve(dog_pyramid.n_scales);

        for (int j = 1; j < pyramid->n_scales; j++)
        {
            cv::Mat diff = subtractImages(&pyramid->octaves[i][j], &pyramid->octaves[i][j-1]);
            //cv::Mat diff = pyramid->octaves[i][j] - pyramid->octaves[i][j-1];
            dog_pyramid.octaves[i].push_back(diff);
        }
    }

    /*for (int i = 0; i < dog_pyramid.n_octaves; i++){
        for (int j = 0; j < dog_pyramid.n_scales; j++){
            double max, min;
            cv::minMaxLoc(dog_pyramid.octaves[i][j],&min, &max);
            std::cout << min << "; " << max << std::endl;
            for (int x=0; x<dog_pyramid.octaves[i][j].size().width; x++)
                for (int y=0; y<dog_pyramid.octaves[i][j].size().height; y++)
                    dog_pyramid.octaves[i][j].at<double>(x,y) = 2*(dog_pyramid.octaves[i][j].at<double>(x,y) - min)/(max-min)-1;
            
        }
        std::cout << std::endl;    
    }*/

    return dog_pyramid;
}

/*
    Extremum Values and Keypoints
*/
bool pointIsExtremum(const std::vector<cv::Mat> &octave, int scale, int x, int y)
{
    const cv::Mat &img = octave[scale];
    const cv::Mat &prev = octave[scale - 1];
    const cv::Mat &next = octave[scale + 1];

    bool stop = 0;
    double value = getPixel(img, x, y);

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (x + i < 0 || (x + i) > img.size().width or y + j < 0 || y + j > img.size().height)
                continue;

            double v1 = img.at<double>(x+i, y+j);
            double v2 = prev.at<double>(x+i, y+j);
            double v3 = next.at<double>(x+i, y+j);

            if (std::max({value, v1, v2, v3}) != value && std::min({value, v1, v2, v3}) != value)
            {
                stop = 1;
                break;
            }
        }
    }
    return !stop;
}

std::tuple<double, double, double> fitQuadratic(Keypoint &kp, const std::vector<cv::Mat> &octave, int scale)
{
    const cv::Mat &img = octave[scale];
    const cv::Mat &prev = octave[scale - 1];
    const cv::Mat &next = octave[scale + 1];

    double g1, g2, g3;
    double h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    g1 = (getPixel(next, x, y) - getPixel(prev, x, y)) * 0.5;
    g2 = (getPixel(img, x + 1, y) - getPixel(img, x - 1, y)) * 0.5;
    g3 = (getPixel(img, x, y + 1) - getPixel(img, x, y - 1)) * 0.5;

    h11 = getPixel(next, x, y) + getPixel(prev, x, y) - 2 * getPixel(img, x, y);
    h22 = getPixel(img, x + 1, y) + getPixel(img, x - 1, y) - 2 * getPixel(img, x, y);
    h33 = getPixel(img, x, y + 1) + getPixel(img, x, y - 1) - 2 * getPixel(img, x, y);
    h12 = (getPixel(next, x + 1, y) - getPixel(next, x - 1, y) - getPixel(prev, x + 1, y) + getPixel(prev, x - 1, y)) * 0.25;
    h13 = (getPixel(next, x, y + 1) - getPixel(next, x, y - 1) - getPixel(prev, x, y + 1) + getPixel(prev, x, y - 1)) * 0.25;
    h23 = (getPixel(img, x + 1, y + 1) - getPixel(img, x + 1, y - 1) - getPixel(img, x - 1, y + 1) + getPixel(img, x - 1, y - 1)) * 0.25;

    double hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    double det = h11 * h22 * h33 - h11 * h23 * h23 - h12 * h12 * h33 + 2 * h12 * h13 * h23 - h13 * h13 * h22;
    hinv11 = (h22 * h33 - h23 * h23) / det;
    hinv12 = (h13 * h23 - h12 * h33) / det;
    hinv13 = (h12 * h23 - h13 * h22) / det;
    hinv22 = (h11 * h33 - h13 * h13) / det;
    hinv23 = (h12 * h13 - h11 * h23) / det;
    hinv33 = (h11 * h22 - h12 * h12) / det;

    double offset_s = -hinv11 * g1 - hinv12 * g2 - hinv13 * g3;
    double offset_x = -hinv12 * g1 - hinv22 * g2 - hinv23 * g3;
    double offset_y = -hinv13 * g1 - hinv23 * g3 - hinv33 * g3;

    /*std::cout << "s:" << offset_s << std::endl;
    std::cout << "x:" << offset_x << std::endl;
    std::cout << "y:" << offset_y << std::endl;
    std::cout << std::endl;*/

    double interpolated_extrema_val = getPixel(img, x, y) + 0.5 * (g1 * offset_s + g2 * offset_x + g3 * offset_y);
    kp.extremum_val = interpolated_extrema_val;

    return {offset_s, offset_x, offset_y};
}

bool pointIsOnEdge(const Keypoint &kp, const std::vector<cv::Mat> &octave, double edge_thresh = C_EDGE)
{
    const cv::Mat &img = octave[kp.scale];
    double h11, h12, h22;
    int x = kp.i, y = kp.j;

    h11 = getPixel(img, x + 1, y) + getPixel(img, x - 1, y) - 2 * getPixel(img, x, y);
    h22 = getPixel(img, x, y + 1) + getPixel(img, x, y - 1) - 2 * getPixel(img, x, y);
    h12 = (getPixel(img, x + 1, y + 1) - getPixel(img, x + 1, y - 1) - getPixel(img, x - 1, y + 1) + getPixel(img, x - 1, y - 1)) * 0.25;

    double det_hessian = h11 * h22 - h12 * h12;
    double tr_hessian = h11 + h22;
    double edgeness = tr_hessian * tr_hessian / det_hessian;

    return (edgeness > std::pow(edge_thresh + 1, 2) / edge_thresh);
}

void findInputImgCoords(Keypoint &kp, double offset_s, double offset_x, double offset_y, double sigma_min = MIN_SIGMA, double min_pix_dist = MIN_PIX_DIST, int n_spo = N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s + kp.scale) / n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x + kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y + kp.j);
}

bool refineOrDiscardKeypoint(Keypoint &kp, const std::vector<cv::Mat> &octave, double contrast_thresh, double edge_thresh){
    int k = 0;
    bool kp_is_valid = false;

    while (k++ < MAX_REFINEMENT_ITERS)
    {
        std::tuple<double, double, double> offsets = fitQuadratic(kp, octave, kp.scale);
        double offset_s = std::get<0>(offsets);
        double offset_x = std::get<1>(offsets);
        double offset_y = std::get<2>(offsets);

        double max_offset = std::max({std::abs(offset_s), std::abs(offset_x), std::abs(offset_y)});

        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);

        if (kp.scale >= (int) octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !pointIsOnEdge(kp, octave, edge_thresh)){
            findInputImgCoords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> findKeypoints(const ScaleSpacePyramid &dog_pyramid, double contrast_thresh, double edge_thresh){
    std::vector<Keypoint> keypoints;
    //double threshold = std::floor(0.5 * contrast_thresh / (N_SPO+3) * 255); 
    for (int i=0; i<dog_pyramid.n_octaves; i++){
        const std::vector<cv::Mat> &octave = dog_pyramid.octaves[i];

        for (int j=1; j<dog_pyramid.n_scales-1; j++){
            const cv::Mat &img = octave[j];

            for (int x=1; x < img.size().width-1; x++){
                for (int y=1; y < img.size().height-1; y++){

                    //if (std::abs(getPixel(img, x, y)) < contrast_thresh)
                    //    continue;

                    if (pointIsExtremum(octave, j, x, y)){
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refineOrDiscardKeypoint(kp, octave, contrast_thresh, edge_thresh);
                        if (kp_is_valid)
                            keypoints.push_back(kp);
                    }
                }
            }
        }
    }
    return keypoints;
}

/*
    Other
*/
cv::Mat returnImage(cv::Mat* image){
    cv::Mat output = cv::Mat(image->size().height, image->size().width, 0);

    for (int i = 0; i < output.size().width; i++)
        for (int j = 0; j < output.size().height; j++)
            output.at<uchar>(i,j) = (uchar)std::round((image->at<double>(i,j)*255));
    
    return output;
}

void saveScaleSpacePyramidImages(ScaleSpacePyramid* pyramid, std::string base_path){
    for (int i=0; i<pyramid->n_octaves; i++){
        for (int j=0; j<pyramid->n_scales; j++){
            std::stringstream sstm;
            sstm << base_path << i << j << ".jpg";
            cv::imwrite(sstm.str(), returnImage(&pyramid->octaves[i][j]));
        }
    }
}

cv::Mat convert8UITo64F(cv::Mat* image){
    cv::Mat output = cv::Mat(image->size().height, image->size().width, CV_64FC1);
    
    for (int i = 0; i < output.size().width; i++)
        for (int j = 0; j < output.size().height; j++)
            output.at<double>(i,j) = (double)(image->at<uchar>(i,j))/255.0;

    return output;
}

/*
    Main Function
*/
int main()
{

    cv::Mat input = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat image = convert8UITo64F(&input);
    //cv::imwrite("output.jpg", image);
    
    //cv::Mat out_resize = resize(&image, 2);
    //cv::imwrite("output_resize.jpg", out_resize);

    //cv::Mat out_blur = blurImage(&image, std::sqrt(2));
    //cv::imwrite("output_blur.jpg", out_blur);
    
    ScaleSpacePyramid pyramid = generateGaussianPyramid(&image, N_OCT, N_SPO, MIN_SIGMA);
    //saveScaleSpacePyramidImages(&pyramid, "output");
    
    ScaleSpacePyramid dog = generateDogPyramid(&pyramid);
    saveScaleSpacePyramidImages(&dog, "dog");
    
    std::vector<Keypoint> keypoints = findKeypoints(dog, C_DOG, C_EDGE);
    
    
    std::cout << keypoints.size() << std::endl << std::endl;
    /*for (int i=0; i<keypoints.size(); i++){
        Keypoint kp = keypoints[i];
        std::cout << kp.i << " " << kp.j << std::endl;
        std::cout << kp.octave << " " << kp.scale << std::endl;
        std::cout << kp.x << " " << kp.y << std::endl;
        std::cout << kp.sigma << " " << kp.extremum_val << std::endl;
        std::cout << std::endl;

    }*/
    
    return 0;
}