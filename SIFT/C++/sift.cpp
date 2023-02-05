/*
    Implementação do algoritmo SIFT - Scale-Invariant Feature Transform - inventado por David G. Lowe 

    // TODO: COMENTAR REFERẼNCIAS
    Artigos Utilizados como Referência:

    Outros materiais utilizados como referências:

*/

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

/*
    Definição de Constantes
*/

// Constantes referente ao tamanho da imagem utilizada como entrada para o algoritmo
const int IN_WIDTH = 256;                   
const int IN_HEIGH = 256;

const double MIN_PIX_DIST = 0.5;  

// Constantes utilizadas em relação ao blur gaussiano
const double MIN_SIGMA = 0.8;                                                                           // Sigma mínimo a ser considerado                   
const double SIGMA_IN = 0.5;                                                                            // Sigma da imagem de entrada

// Constantes utilizadas para montar as pirâmides de imagens
const int N_OCT = (int)(std::round(std::log(std::max(IN_HEIGH, IN_WIDTH)) / std::log(2) - 1));              // Número de octetos
const int N_SPO = 3;                                                                                        // Número de escalas por octeto

// Constantes utilizadas para descartar candidatos á ponto característico
const double C_DOG = 0.015;                                                                                 // Threshold contraste
const double C_EDGE = 10;                                                                                   // Threshold aresta/curvatura

// Constante utilizada no refinamento de candidatos á ponto característico
const int MAX_REFINEMENT_ITERS = 5;                                                                         // Limita a quantidade de iterações usadas para refinar/descartar um candidato á ponto característico

// Constantes utilizadas para a computação das orientações e descritor dos pontos característicos
const int MAX_HIST_SMOOTH = 6;                                                                              // Limite de suavização do histograma  
const int N_BINS = 36;                                                                                      // Divisões do histograma (36 -> 10º de variação para cada) de orientações
const double ORI_THRESHOLD = 0.8;                                                                           // Usado para refinar as orientações que serão consideradas relevantes
const double LAMBDA_ORI = 1.5; //TODO: COMENTAR 
const int N_HIST = 4;                                                                                       // Quantidade de histogramas criados durante o cálculo do descritor de um ponto característico
const int N_ORI = 8;                                                                                        // Intervalo de orientações consideradas nos histogramas para cálculo do descritor
const double LAMBDA_DESC = 6;                                                                               // Peso utilizado para definir o tamanho da janela considerada pelo descritor
const double MAG_THRESHOLD = 0.2;                                                                           // Threshold para eliminar magnitudes de gradientes

/*
    Definição de Estruturas auxiliares utilizadas pelo algoritmo
*/

// Estrutura utilizada para representar um Ponto Característico processado pelo algoritmo
typedef struct {
    int i;                                      // Coordenada x Inteira - Imagem I(x,y)
    int j;                                      // Coordenada y Inteira - Imagem I(x,y)
    int octave;                                 // Octeto em que o ponto encontra-se
    int scale;                                  // Escala do octeto em que o ponto encontra-se
    double x;                                   // Coordenada x Real - Precisão subpíxel
    double y;                                   // Coordenada y Real - Precisão subpíxel
    double sigma;                               // Valor de sigma da imagem em que o ponto encontra-se
    double extremum_val;                        // Valor do ponto (calculado com precisão subpíxel)
    std::array<uint8_t, 128> descriptor;        // Descritores do ponto característico
} Keypoint;

// Estrutura utilizada para representar uma estrutura de pirâmides de imagens
typedef struct {
    int n_octaves;                              // Número de octetos da pirâmide
    int n_scales;                               // Número de escalas por octeto
    std::vector<std::vector<cv::Mat>> octaves;  // Vetor bidimensional armazenando as imagens (Matrizes do opencv) em octetcos e escalas
} ScaleSpacePyramid;


/*
    Funções auxiliares - Manipulação de Imagens
*/

// Retorna o valor de um píxel de uma imagem.
// Caso a coordenada esteja além das dimensões da imagem, retorna-se o valor
// correspondente a borda mais próxima
double getPixel(const cv::Mat &img, int x, int y){
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

// Dados os offsets um ponto característico, atualiza a sua posição na imagem
void findInputImgCoords(Keypoint &kp, double offset_s, double offset_x, double offset_y){
    kp.sigma = std::pow(2, kp.octave) * MIN_SIGMA * std::pow(2, (offset_s + kp.scale) / N_SPO);
    kp.x = MIN_PIX_DIST * std::pow(2, kp.octave) * (offset_x + kp.i);
    kp.y = MIN_PIX_DIST * std::pow(2, kp.octave) * (offset_y + kp.j);
}

// Dada uma Matriz (opencv) normalizada, retorna-a em formato hábil para ser convertida em imagem (valores entre 0 e 255)
cv::Mat returnImage(cv::Mat* image){
    cv::Mat output = cv::Mat(image->size().height, image->size().width, 0);

    for (int i = 0; i < output.size().width; i++)
        for (int j = 0; j < output.size().height; j++)
            output.at<uchar>(i,j) = (uchar)std::round((image->at<double>(i,j)*255));
    
    return output;
}

// Salva as imagens contidas em uma pirâmide de imagens
void saveScaleSpacePyramidImages(ScaleSpacePyramid* pyramid, std::string base_path){
    for (int i=0; i<pyramid->n_octaves; i++){
        for (int j=0; j<pyramid->n_scales; j++){
            std::stringstream sstm;
            sstm << base_path << i << j << ".jpg";
            cv::imwrite(sstm.str(), returnImage(&pyramid->octaves[i][j]));
        }
    }
}

// Converte imagens para o formato utilizado pelo algoritmo - valores float 64 bits normalizados
cv::Mat convert8UITo64F(cv::Mat* image){
    cv::Mat output = cv::Mat(image->size().height, image->size().width, CV_64FC1);
    
    for (int i=0; i<output.size().width; i++)
        for (int j=0; j<output.size().height; j++)
            output.at<double>(i,j) = (double)(image->at<uchar>(i,j))/255.0;

    return output;
}


/*
    Funções Auxiliares - Manipulação de Histogramas
*/

// Suavizar histograma
// TODO: MODIFICAR PARA TRABALHAR COM PONTEIROS
void smoothHistogram(double hist[N_BINS]){
    
    // Histograma utilizado para armazenar os valores do histograma conforme suavização
    double tmp_hist[N_BINS];
    
    for (int i=0; i<MAX_HIST_SMOOTH; i++) {
        
        // Suavização do histograma através da média aritmética dos valores
        for (int j=0; j<N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        
        for (int j=0; j<N_BINS; j++)
            hist[j] = tmp_hist[j];
    }
}

// Atualizar o histograma
// TODO: COMENTAR
void updateHistograms(double hist[N_HIST][N_HIST][N_ORI], double x, double y, double contrib, double theta_mn){
    double x_i, y_j;
    
    for (int i = 1; i<=N_HIST; i++) {
        x_i = (i-(1+(double)N_HIST)/2) * 2*LAMBDA_DESC/N_HIST;
        
        if (std::abs(x_i-x) > 2*LAMBDA_DESC/N_HIST)
            continue;
        
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(double)N_HIST)/2) * 2*LAMBDA_DESC/N_HIST;
            if (std::abs(y_j-y) > 2*LAMBDA_DESC/N_HIST)
                continue;
            
            double hist_weight = (1 - N_HIST*0.5/LAMBDA_DESC*std::abs(x_i-x))*(1 - N_HIST*0.5/LAMBDA_DESC*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                double theta_k = 2*M_PI*(k-1)/N_ORI;
                double theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                double bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

// Converter um histograma para vetor/array - utilizado como o descritor de um ponto característico
void histsToVec(double histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec){
    
    int size = N_HIST*N_HIST*N_ORI;
    double *hist = reinterpret_cast<double *>(histograms);

    /*
        Normalização dos valores
    */

    double norm = 0;
    for (int i=0; i<size; i++) {
        norm += hist[i] * hist[i];
    }
    
    norm = std::sqrt(norm);
    
    double norm2 = 0;
    for (int i=0; i<size; i++) {
        hist[i] = std::min(hist[i], MAG_THRESHOLD*norm);
        norm2 += hist[i] * hist[i];
    }
    
    norm2 = std::sqrt(norm2);
    for (int i=0; i<size; i++) {
        double val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}


/*
    Funções para implementar o método de Cholesky para resolução de sistemas lineares
*/
// TODO: ADICIONAR CHOLESKY PARA INVERSÃO DE MATRIZ


/*
    Funções para interpolação numérica
*/

double linearInterpolation(double x, double x0, double y0, double x1, double y1){
    
    double u = (x - x0) / (x1 - x0);
    return (1 - u) * y0 + u * y1;
}

double bilinearInterpolation(double x, double y, double x0, double y0, double x1, double y1, double f00, double f01, double f10, double f11){
    
    double r1 = linearInterpolation(x, x0, f00, x1, f10);
    double r2 = linearInterpolation(x, x0, f01, x1, f11);

    return linearInterpolation(y, y0, r1, y1, r2);
}


/*
    Função para redimensionar uma imagem (representação matricial do opencv)
*/

cv::Mat resize(cv::Mat *image, double scale=1){

    // Cria uma matriz que conterá os valores da imagem redimensionada. Utiliza-se uma imagem de canal único
    // que comporta valores float 64 bits pela precisão, e pela imagem ser manipulada em escala de cinza. 
    cv::Mat out = cv::Mat((int)image->size().height * scale, (int)image->size().width * scale, CV_64FC1);

    double value;

    // Geração dos valores da imagem redimensionada
    for (int i=0; i < out.size().width; i++){
        for (int j=0; j < out.size().height; j++){

            // Posição na imagem de saída é a mesma da imagem de entrada
            if (std::fmod(i, scale) == 0 && std::fmod(j, scale) == 0)
                value = image->at<double>(i / scale, j / scale);

            // Posição na imagem de saída está entre os pixels da imagem de entrada
            else {

                // Calcular os valores dos píxels da imagem de entrada que serão utilizados
                // para a interpolação do pixel da imagem de saída redimensionada
                double x1 = std::ceil(i / scale);
                double y1 = std::ceil(j / scale);
                double x0 = std::min(x1 - 1, std::floor(i / scale));
                double y0 = std::min(y1 - 1, std::floor(j / scale));

                // Caso um dos píxels esteja além da borda, replica o valor da borda
                if (x1 > image->size().width - 1 || y1 > image->size().height - 1)
                    value = image->at<double>(std::floor(i / scale), std::floor(j / scale));

                // Caso contrário, realiza-se a interpolação numérica do píxel da imagem de saída
                else {

                    // Recuperar os valores dos píxels utilizados como referência para a interpolação
                    double f00 = image->at<double>(x0, y0);
                    double f01 = image->at<double>(x0, y1);
                    double f10 = image->at<double>(x1, y0);
                    double f11 = image->at<double>(x1, y1);

                    value = bilinearInterpolation(i / scale, j / scale, x0, y0, x1, y1, f00, f01, f10, f11);
                }
            }
            // Escrita do valor interpolado na imagem de saída em sua respectiva posição
            out.at<double>(i, j) = value;
        }
    }
    return out;
}

/*
    Blur Gaussiano
*/

// Função utilizada para gerar o kernel do blur gaussiano
double **generateGaussianKernel(int size, double sigma){

    // Garante que o tamanho do kernel seja ímpar
    assert(size % 2 != 0);

    int radius = (size - 1) / 2;
    double **kernel = new double *[size];

    // Alocação da memória utilizada para o kernel
    for (int i=0; i<size; i++)
        kernel[i] = new double[size];

    // Calculando elementos da fórmula do blur gaussiano
    double a = 1.0 / (2 * M_PI * std::pow(sigma, 2));
    double b = 2.0 * std::pow(sigma, 2);

    // Utilizado para a normalização do kernel
    double sum = 0;

    // Geração do kernel
    for (int i =-radius; i <=radius; i++){
        for (int j=-radius; j <=radius; j++){
            kernel[i + radius][j + radius] = a * std::pow(M_E, (-(i * i + j * j) / b));
            sum += kernel[i + radius][j + radius];
        }
    }

    // Normalização do kerel
    for (int i=0; i<size; i++)
        for (int j=0; j<size; j++)
            kernel[i][j] = kernel[i][j] / sum;

    return kernel;
}

// Função para gerar uma imagem após a aplicação do blur Gaussiano
cv::Mat blurImage(cv::Mat *image, double sigma = std::sqrt(2)){
    
    // Calcula o tamanho do kernel baseado no valor de sigma
    int size = (int)(std::round(sigma*6+1))|1;                      // Código retirado da biblioteca opencv
    //int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    
    // Gera o kernel
    double **kernel = generateGaussianKernel(size, sigma);
    int radius = (size - 1) / 2;

    // Imagem de saída após aplicação do blur
    cv::Mat out = cv::Mat(image->size().height, image->size().width, CV_64FC1);

    double value;
    
    // Aplicação do blur
    for (int i=0; i<out.size().width; i++){
        for (int j=0; j<out.size().height; j++){
            
            value = 0;

            // Convulução do kernel no píxel da posição (i,j)
            for (int ki = -radius; ki <= radius; ki++){
                for (int kj = -radius; kj <= radius; kj++){
                    // Se o píxel necessário para  convulução está além da borda da imagem, apenas ignore e continue a convulção em (i,j)
                    if (i + ki < 0 || i + ki >= out.size().width || j + kj < 0 || j + kj >= out.size().height)
                        continue;

                    value += image->at<double>(i + ki, j + kj) * kernel[ki + radius][kj + radius];
                }
            }
            out.at<double>(i, j) = value;
        }
    }

    // Desaloca a memória dinâmicamente ocupada pelo kernel do blur gaussiano
    for (int i = 0; i < size; i++)
        delete kernel[i];
    delete[] kernel;

    return out;
}


/*
    Função para gerar a pirâmide gaussiana utilizada pelo SIFT
*/

ScaleSpacePyramid generateGaussianPyramid(cv::Mat *image){
    
    // Gera a imagem base da pirâmide - conforme observado no artigo escrito por Lowe,
    // redimensionar a imagem e aplicar o blur permite a detecção de mais pontos característicos
    double base_sigma = MIN_SIGMA / MIN_PIX_DIST;
    double sigma_diff = std::sqrt(base_sigma * base_sigma - 1.0);
    cv::Mat base_image = resize(image, 2);
    base_image = blurImage(&base_image, sigma_diff);

    int images_per_octave = N_SPO + 3;

    // Criação dos valores de sigma utilizados nas diferentes escalas de cada octetos
    double k = std::pow(2, 1.0/N_SPO);
    std::vector<double> sigma_vals{base_sigma};
    for (int i=1; i<images_per_octave; i++){

        double sigma_prev = base_sigma * std::pow(k, i - 1);
        double sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev));
    
    }

    // Criação da pirâmide gaussiana
    ScaleSpacePyramid pyramid;
    pyramid.n_octaves = N_OCT;
    pyramid.n_scales = images_per_octave;
    pyramid.octaves = std::vector<std::vector<cv::Mat>>(N_OCT);

    // Criação das imagens de cada octeto
    for (int i=0; i<N_OCT; i++){

        pyramid.octaves[i].reserve(images_per_octave);
        pyramid.octaves[i].push_back(std::move(base_image));

        for (int j=1; j<(int)sigma_vals.size(); j++){
            // Criação da imagem na escala 'j', do octeto 'i'
            cv::Mat prev_image = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(blurImage(&prev_image, sigma_vals[j]));
        }

        // Criação da imagem base utilizada para gerar o próximo octeto
        cv::Mat next_base_image = pyramid.octaves[i][images_per_octave - 3];
        base_image = resize(&next_base_image, 0.5);
    }

    return pyramid;
}


/*
    Geração da pirâmide com DoG - Difference of Gaussians
*/

// Função que subtrai os valores da imagem2 da imagem1 -> imagem1 - imagem2 
cv::Mat subtractImages(cv::Mat *image1, cv::Mat *image2){

    // Assegura que as imagens possuem o mesmo tamanho
    assert(image1->size().height == image2->size().height && image1->size().width == image2->size().width);

    // Imagem de saída
    cv::Mat out = cv::Mat(image1->size().height, image1->size().width, CV_64FC1);

    // Subtração dos píxels
    for (int i=0; i<out.size().width; i++){
        for (int j=0; j<out.size().height; j++){
            double value = image1->at<double>(i, j) - image2->at<double>(i, j);
            out.at<double>(i, j) = value;
        }
    }
    return out;
}

// Função que gera a pirâmide com as DoG
ScaleSpacePyramid generateDogPyramid(ScaleSpacePyramid *pyramid){

    ScaleSpacePyramid dog_pyramid = {
        pyramid->n_octaves,
        pyramid->n_scales - 1,
        std::vector<std::vector<cv::Mat>>(pyramid->n_octaves)
    };

    // Criação dos octetos e das escalas formadas por DoG
    for (int i=0; i<dog_pyramid.n_octaves; i++){
        dog_pyramid.octaves[i].reserve(dog_pyramid.n_scales);

        for (int j=1; j<pyramid->n_scales; j++){
            cv::Mat diff = subtractImages(&pyramid->octaves[i][j], &pyramid->octaves[i][j-1]);
            dog_pyramid.octaves[i].push_back(diff);
        }
    }

    return dog_pyramid;
}


/*
    Funções para achar os pontos de extremo - candidatos á pontos característicos -
    e refinamento dos pontos encontrados
*/

// Verifica se o ponto é um extremo local
bool pointIsExtremum(const std::vector<cv::Mat> &octave, int scale, int x, int y){

    // Recupera 3 imagens, sendo uma referente a escala atual e as outras sua antecessora e sua sucessora
    const cv::Mat &img = octave[scale];
    const cv::Mat &prev = octave[scale - 1];
    const cv::Mat &next = octave[scale + 1];

    bool stop = 0;
    double value = getPixel(img, x, y);

    // Realiza a verificação na vizinhança do píxel em (x,y)
    for (int i =-1; i<=1; i++){
        for (int j=-1; j<=1; j++){
            // Ignorar se a coordenada a ser usada para comparação está além das bordas da imagem
            if (x + i < 0 || (x + i) > img.size().width or y + j < 0 || y + j > img.size().height)
                continue;

            // Recupera o valor dos píxels em (x+i, y+j) nas 3 escalas
            double v1 = img.at<double>(x+i, y+j);
            double v2 = prev.at<double>(x+i, y+j);
            double v3 = next.at<double>(x+i, y+j);

            // Se o ponto (x,y) não for ou maior ou menor do que seus vizinhos, podemos parar a verificação,
            // pois não é um ponto extremo
            if (std::max({value, v1, v2, v3}) != value && std::min({value, v1, v2, v3}) != value){
                stop = 1;
                break;
            }
        }
    }
    // Se a verificação foi parada, significa que existe algum valor maior/menor que o píxel considerado,
    // logo ele não é um ponto de extremo. Se a verificação chegou até o final, sem interrupções, significa
    // que o ponto é um extremo
    return !stop;
}

// Interpola uma função quadrática (expansão do polinômio de Taylor de 2º grau) no ponto de extremo local para identificar o valor do ponto de extremo 
// com precisão subpíxel
std::tuple<double, double, double> fitQuadratic(Keypoint &kp, const std::vector<cv::Mat> &octave, int scale){

    const cv::Mat &img = octave[scale];
    const cv::Mat &prev = octave[scale - 1];
    const cv::Mat &next = octave[scale + 1];

    double g0, g1, g2;
    double h00, h01, h02, h11, h12, h22;
    int x = kp.i, y = kp.j;

    // Cálculo do gradiente
    g0 = (getPixel(next, x, y) - getPixel(prev, x, y)) * 0.5;
    g1 = (getPixel(img, x + 1, y) - getPixel(img, x - 1, y)) * 0.5;
    g2 = (getPixel(img, x, y + 1) - getPixel(img, x, y - 1)) * 0.5;

    // Cálculo da Matriz Triangular Superior da Hessiana 3x3
    h00 = getPixel(next, x, y) + getPixel(prev, x, y) - 2 * getPixel(img, x, y);
    h01 = (getPixel(next, x + 1, y) - getPixel(next, x - 1, y) - getPixel(prev, x + 1, y) + getPixel(prev, x - 1, y)) * 0.25;
    h02 = (getPixel(next, x, y + 1) - getPixel(next, x, y - 1) - getPixel(prev, x, y + 1) + getPixel(prev, x, y - 1)) * 0.25;
    h11 = getPixel(img, x + 1, y) + getPixel(img, x - 1, y) - 2 * getPixel(img, x, y);
    h12 = (getPixel(img, x + 1, y + 1) - getPixel(img, x + 1, y - 1) - getPixel(img, x - 1, y + 1) + getPixel(img, x - 1, y - 1)) * 0.25;
    h22 = getPixel(img, x, y + 1) + getPixel(img, x, y - 1) - 2 * getPixel(img, x, y);
    
    // Cálculo da inversa da Matriz Gaussiana
    // TODO: APRIMORAR MÉTODO PARA CALCULAR A INVERSÃO DA MATRIZ 
    double hinv00, hinv01, hinv02, hinv11, hinv12, hinv22;
    double det = h00 * h11 * h22 - h00 * h12 * h12 - h01 * h01 * h22 + 2 * h01 * h02 * h12 - h02 * h02 * h11;
    hinv00 = (h11 * h22 - h12 * h12) / det;
    hinv01 = (h02 * h12 - h01 * h22) / det;
    hinv02 = (h01 * h12 - h02 * h11) / det;
    hinv11 = (h00 * h22 - h02 * h02) / det;
    hinv12 = (h01 * h02 - h00 * h12) / det;
    hinv22 = (h00 * h11 - h01 * h01) / det;

    // Cálculo dos offsets - idêntifica o deslocamento com precisão subpíxel do ponto de extremo
    double offset_s = -hinv00 * g0 - hinv01 * g1 - hinv02 * g2;
    double offset_x = -hinv01 * g0 - hinv11 * g1 - hinv12 * g2;
    double offset_y = -hinv02 * g0 - hinv01 * g2 - hinv22 * g2;

    // Interpola o valor do píxel com precisão subpíxel consideradno os offsets encontrados e atualiza as informações
    // do candidato á ponto característico
    double interpolated_extrema_val = getPixel(img, x, y) + 0.5 * (g0 * offset_s + g1 * offset_x + g2 * offset_y);
    kp.extremum_val = interpolated_extrema_val;

    return {offset_s, offset_x, offset_y};
}

// Verifica se um candidato a ponto característico está em uma curvatura principal
bool pointIsOnEdge(const Keypoint &kp, const std::vector<cv::Mat> &octave){
    
    const cv::Mat &img = octave[kp.scale];
    
    double h00, h01, h11;
    int x = kp.i, y = kp.j;

    // Valores da Matriz Hessiana 2x2 utilizados para calcular o determinante e o traço
    h00 = getPixel(img, x + 1, y) + getPixel(img, x - 1, y) - 2 * getPixel(img, x, y);
    h11 = getPixel(img, x, y + 1) + getPixel(img, x, y - 1) - 2 * getPixel(img, x, y);
    h01 = (getPixel(img, x + 1, y + 1) - getPixel(img, x + 1, y - 1) - getPixel(img, x - 1, y + 1) + getPixel(img, x - 1, y - 1)) * 0.25;

    double det_hessian = h00 * h11 - h01 * h01;
    double tr_hessian = h00 + h11;
    double edgeness = tr_hessian * tr_hessian / det_hessian;

    return (edgeness > std::pow(C_EDGE + 1, 2) / C_EDGE);
}

// Função para refinar ou descartar de forma iterativa os candidatos á pontos característicos
bool refineOrDiscardKeypoint(Keypoint &kp, const std::vector<cv::Mat> &octave){
    
    int k = 0;
    bool kp_is_valid = false;

    while (k++ < MAX_REFINEMENT_ITERS){

        // Cálculo dos offsets através da interpolação do polinômio quadrático
        std::tuple<double, double, double> offsets = fitQuadratic(kp, octave, kp.scale);
        double offset_s = std::get<0>(offsets);
        double offset_x = std::get<1>(offsets);
        double offset_y = std::get<2>(offsets);

        double max_offset = std::max({std::abs(offset_s), std::abs(offset_x), std::abs(offset_y)});

        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);

        // Escala obtida está além das escalas geradas em um octeto
        if (kp.scale >= (int) octave.size()-1 || kp.scale < 1)
            break;

        // Verifica o contraste
        bool valid_contrast = std::abs(kp.extremum_val) > C_DOG;
        
        // Validação base do candidato á ponto característico
        if (max_offset < 0.6 && valid_contrast && !pointIsOnEdge(kp, octave)){
            findInputImgCoords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

// Função para achar os pontos característicos
std::vector<Keypoint> findKeypoints(const ScaleSpacePyramid &dog_pyramid){
    std::vector<Keypoint> keypoints;
    
    // Percorre os octetos
    for (int i=0; i<dog_pyramid.n_octaves; i++){
        const std::vector<cv::Mat> &octave = dog_pyramid.octaves[i];

        // Percorre as escalas
        for (int j=1; j<dog_pyramid.n_scales-1; j++){
            const cv::Mat &img = octave[j];

            // Percore os pixels da imagem na escala 'j' do octeto 'i' 
            for (int x=1; x < img.size().width-1; x++){                         // Indícies ignoram as bordas da imagem  
                for (int y=1; y < img.size().height-1; y++){                
                    
                    // Realiza as validações e o refinamento dos candidatos a ponto característico
                    if (pointIsExtremum(octave, j, x, y)){
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refineOrDiscardKeypoint(kp, octave);
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
    Funções necessárias para calcular os descritores dos pontos característicos
*/

// Gera uma pirâmide de imagens, onde os pixels das imagens são substituidos pelo gradiente 
ScaleSpacePyramid generateGradientPyramid(const ScaleSpacePyramid& pyramid){
    
    ScaleSpacePyramid grad_pyramid = {
        pyramid.n_octaves,
        pyramid.n_scales,
        std::vector<std::vector<cv::Mat>>(pyramid.n_octaves)
    };
    
    // Percorrendo os octetos da pirâmide de imagens de entrada
    for (int i=0; i<pyramid.n_octaves; i++) {
        
        grad_pyramid.octaves[i].reserve(grad_pyramid.n_scales);
        
        int width = pyramid.octaves[i][0].size().width;
        int height = pyramid.octaves[i][0].size().height;
    
        // Percorrendo as escalas do octeto
        for (int j=0; j<pyramid.n_scales; j++) {
            
            // Imagem de gradiente
            cv::Mat grad = cv::Mat(height, width, CV_64FC1);
            
            double gx, gy;
    
            for (int x=1; x < width-1; x++) {
                for (int y=1; y < height-1; y++) {
            
                    // Derivada parcial em relação á x
                    gx = (pyramid.octaves[i][j].at<double>(x+1, y) - pyramid.octaves[i][j].at<double>(x-1, y)) * 0.5;
                    grad.at<double>(x,y) = gx;
                    
                    // Derivada parcial em relação á y
                    gy = (pyramid.octaves[i][j].at<double>(x, y+1) - pyramid.octaves[i][j].at<double>(x, y-1)) * 0.5;
                    grad.at<double>(x,y) = gy;
                }
            }
            grad_pyramid.octaves[i].push_back(grad);
        }
    }
    return grad_pyramid;
}

// Calcular as orientações para um ponto característico
std::vector<double> findKeypointOrientations(Keypoint& kp, const ScaleSpacePyramid& grad_pyramid){
    
    double pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const cv::Mat& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // Discartar pontos próximos da borda
    double min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.size().width-kp.x,pix_dist*img_grad.size().height-kp.y});

    if (min_dist_from_border <= std::sqrt(2)*LAMBDA_DESC*kp.sigma)
        return {};

    // Histograma
    double hist[N_BINS] = {};
    int bin;
    
    double gx, gy, grad_norm, weight, theta;
    double patch_sigma = LAMBDA_ORI * kp.sigma;
    double patch_radius = 3 * patch_sigma;
    
    // Limites da região da imagem usados para computar as orientações do ponto característico
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // Formar o histograma com base nas orientações dos pontos próximos ao ponto característico
    for (int x=x_start; x<=x_end; x++) {
        for (int y=y_start; y<=y_end; y++) {
            gx = img_grad.at<double>(x, y);
            gy = img_grad.at<double>(x, y);
            
            // Magnitude do gradiente
            grad_norm = std::sqrt(gx*gx + gy*gy);
            
            // Magnitude
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))/(2*patch_sigma*patch_sigma));
            // Ângulo orientação
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            
            // Preencher o histograma
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;                            // Considera-se os pesos calculados anteriormente
        }
    }

    // Suavização do histograma
    smoothHistogram(hist);

    
    // Extrair as orientações que correspondam entre 80% e 100% do maior valor das orientações
    
    double ori_max = 0;
    std::vector<double> orientations;
    
    // Encontrar maior orientação
    for (int j=0; j<N_BINS; j++) {
        if (hist[j] > ori_max)
            ori_max = hist[j];
    }
    
    // Filtrar as orientações restantes
    for (int j=0; j<N_BINS; j++) {
        if (hist[j] >= ORI_THRESHOLD * ori_max) {
            
            double prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            
            if (prev > hist[j] || next > hist[j])
                continue;
            
            // Interpolação das orientações através de função quadrática + Cálculo do ângulo interpolado
            double theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}


/*
    Computar o descritor de um ponto característico (com base nas suas orientações)
*/

void computeKeypointDescriptor(Keypoint& kp, double theta, const ScaleSpacePyramid& grad_pyramid){
    
    double pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const cv::Mat& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    double histograms[N_HIST][N_HIST][N_ORI] = {0};

    // Variáveis utilizadas para determinar a janela do descritor
    double half_size = std::sqrt(2)*LAMBDA_DESC*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    double cos_t = std::cos(theta), sin_t = std::sin(theta);
    double patch_sigma = LAMBDA_DESC * kp.sigma;

    // Cálculo do histograma para calcular o descritor do ponto característico
    for (int m=x_start; m<=x_end; m++) {
        for (int n=y_start; n<=y_end; n++) {
            
            // Rotação do ponto em relação ao ângulo de orientação do ponto característico
            double x = ((m*pix_dist - kp.x)*cos_t + (n*pix_dist - kp.y)*sin_t) / kp.sigma;
            double y = (-(m*pix_dist - kp.x)*sin_t + (n*pix_dist - kp.y)*cos_t) / kp.sigma;

            
            if (std::max(std::abs(x), std::abs(y)) > LAMBDA_DESC*(N_HIST+1.)/N_HIST)
                continue;

            double gx = img_grad.at<double>(m, n), gy = img_grad.at<double>(m, n);
            double theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            double grad_norm = std::sqrt(gx*gx + gy*gy);
            double weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))/(2*patch_sigma*patch_sigma));
            double contribution = weight * grad_norm;

            updateHistograms(histograms, x, y, contribution, theta_mn);
        }
    }

    // Converter o histograma para o formato de vetor - utilizado como descritor do ponto característico
    histsToVec(histograms, kp.descriptor);
}

/*
    Função Principal - Calcular os pontos característicos e seu descritores
*/

// TODO: COMENTAR
std::vector<Keypoint> findKeypointsAndDescriptors(cv::Mat *img){

    ScaleSpacePyramid gaussian_pyramid = generateGaussianPyramid(img);
    ScaleSpacePyramid dog_pyramid = generateDogPyramid(&gaussian_pyramid);
    std::vector<Keypoint> tmp_kps = findKeypoints(dog_pyramid);
    ScaleSpacePyramid grad_pyramid = generateGradientPyramid(gaussian_pyramid);
    
    std::vector<Keypoint> kps;

    for (Keypoint& kp_tmp : tmp_kps) {
        std::vector<double> orientations = findKeypointOrientations(kp_tmp, grad_pyramid);
        for (double theta:orientations) {
            Keypoint kp = kp_tmp;
            computeKeypointDescriptor(kp, theta, grad_pyramid);
            kps.push_back(kp);
        }
    }

    return kps;
}

int main(){
    cv::Mat input = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image = convert8UITo64F(&input);
    std::vector<Keypoint> keypoints = findKeypointsAndDescriptors(&image);
    std::cout << keypoints.size() << std::endl;
    return 0;
}