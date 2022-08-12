#include <iostream>
#include <array>
#include <chrono>
#include <thread>
#include <vector>
#include <math.h> 
#include <numeric>
#include <functional>
#include "constants.h"
#include "filter_coefficients.h"
#include <algorithm>

using std::cout;
using std::endl;
//using namespace std;


void generate_array_r_prime(double * r_prime) {
    int elements = constants::elements;
    int column_elements = constants::column_elements;
    int row_elements = constants::row_elements;
    double uni_distance = constants::uni_distance;
    double r_a[3] = { constants::r_ax , constants::r_ay, constants::r_az };

    int element_index = 0;

      for (int i = 0; i < row_elements; i++)
      {

        for (int j = 0; j < column_elements; j++)
        { 
          r_prime[element_index] = i*uni_distance + r_a[0];
          r_prime[element_index + elements] = j*uni_distance + r_a[1];

          //std::cout << r_prime[0][element_index];

          element_index +=1;
        }
      }

      for (int i = 0; i < row_elements * column_elements; i++)
      {
        r_prime[i] -= ((double)(row_elements)*uni_distance/2) - uni_distance/2;
        r_prime[i + elements] -= ((double)(column_elements))*uni_distance/2 - uni_distance/2;
      }
}

/* void filter(double * x,int x_length,double a_0, int P) {
    double x_temp[x_length];    //Create an empty array to copy the values of x

    double temp_var;        //Sum variable
    int n;
    int i;

    for (n = 0; n < x_length; n++)
    {
        temp_var = 0;
        x_temp[n] = x[n];       //Get the past values of x and store them in x_temp
        //x[n] = std::inner_product(series1, series1 + n, series2, 0.0);
        for (i = 0; i <= n && i <= P; i++)
        {
            temp_var += filter_coefficients::filt_coeffs[0][i]*x_temp[n-i];
        }
        x[n] = 1/a_0 * temp_var;    //Store value in x, thus overwriting the past values of x, this explains the reason for x_temp
    }
} */

void generate_emulated_data(std::vector<float>& audio_data, double * r_prime) {

    // Get emulation settings
    int sample_max = (int)((constants::t_end - constants::t_start)*constants::f_sampling);
    int elements = constants::elements;
    int sources = constants::sources_N;

    // Generate the frequencies from the sources
    int max_freqs = 0;
    for (int i = 0; i < sources; i++)
    {
        if (max_freqs < constants::source_frequency_N[i])
        {
            max_freqs = constants::source_frequency_N[i];
        }
        
    }

    double frequencies[sources][max_freqs] = {0};

    for (int i = 0; i < sources; i++)
    {
        for (int j = 0; j < constants::source_frequency_N[i]; j++)
        {
            double freq_increment = (constants::source_frequency_span[i][1]-constants::source_frequency_span[i][0])/(constants::source_frequency_N[i] -1);
            frequencies[i][j] = constants::source_frequency_span[i][0] + freq_increment*j;
            //std::cout << "\n";
            //std::cout << frequencies[i][j] ;
        }        
    }
    // Generated source frequencies DONE

    // Generate emulated data
    double t = 0;

    //double r_1[3] = {0};
    double temp_signal_sample = 0;

    double theta = 0;
    double phi = 0;
    double rho = 0;
    double k = 0;

    double rho_sin_theta = 0;
    double cos_phi = 0;
    double sin_phi = 0;
    //double r_2[3] = {0};  

    double norm_factor = 0;
    double phase_offset = 0;

    double element_amplitude = 0;



    // Generate actual data
    for (int mic = 0; mic < elements ; mic++)
    {

        // Pad data with P zeros in the beginning, where P is the fitler order
        for (int j = 0; j < filter_coefficients::filter_order; j++)
        {
            audio_data.push_back(0);
        }

        double r_1[3] = {r_prime[mic],r_prime[mic + elements],r_prime[mic + 2*elements]};

        for (int i = 0; i < sample_max; i++)
        {
            t = (((double)i)/(double)constants::f_sampling);

            temp_signal_sample = 0;

            for (int source = 0; source < sources; source++)
            {
                if (constants::source_t_start[source] <= t && t < constants::source_t_end[source]) {
                    theta = constants::source_theta_deg[source]* constants::pi /180;
                    phi = constants::source_phi_deg[source]* constants::pi /180;
                    rho = constants::source_distance_away[source];
                    for (int freq_ind = 0; freq_ind < constants::source_frequency_N[source]; freq_ind++)
                    {
                        k = 2*constants::pi*frequencies[source][freq_ind]/constants::c;

                        rho_sin_theta = rho*sin(theta);
                        cos_phi = cos(phi);
                        sin_phi = sin(phi);
                        double r_2[3] = {rho_sin_theta*cos_phi,rho_sin_theta*sin_phi,rho*cos(theta) };  

                        norm_factor = sqrt( pow(r_2[0] - r_1[0],2) + pow(r_2[1] - r_1[1],2) + pow(r_2[2] - r_1[2],2) );
                        phase_offset = -k*norm_factor;

                        element_amplitude = 1/norm_factor;

                        temp_signal_sample += element_amplitude*sin(2*constants::pi*frequencies[source][freq_ind]*t + phase_offset);
                    }
                    
                }
            }
            audio_data.push_back(temp_signal_sample);
        }
        
    }
    
}
int weight_index(double frequency) {

    double lambda = constants::c/frequency;

    double lambda_rel = constants::uni_distance/lambda;
    int index;

    if (lambda_rel > 0.1581)
    {
        index = 1;

    } else if (0.156 >= lambda_rel && lambda_rel > 0.0986)
    {
        index = 3;
    } else if (0.0986 >= lambda_rel && lambda_rel > 0.085) {

        index = 5;
    } else if (0.085 >= lambda_rel && lambda_rel > 0.07) {

        index = 6;
    } else {

        index = 7;
    }

    return index;
}

void generate_weight_matrix(int * weight_matrix) {
    int elements = constants::elements;
    int config_modes = constants::available_modes;

    int columns = constants::column_elements;
    int rows = constants::row_elements;

    int element_index;

    for (int mode = 0; mode < config_modes; mode++)
    {
        int row_lim = static_cast<int>((float)(rows)/(float)(mode+1) + 0.99);

        int column_lim = static_cast<int>((float)(columns)/(float)(mode+1) + 0.99);

        int test = (int)(3/4);

        for (int i = 0; i < row_lim; i++)
        {
            for (int j = 0; j < column_lim; j++)
            {
                element_index = ((mode + 1)*(i)) * rows + (mode +1) *(j);
                weight_matrix[elements*mode + element_index] = 1;
                
            }   
        }   
    }
}

void generate_mfilter_coefficients(float * f_mega_coefficients, double * r_prime, int * weight_matrix,
    double theta, double phi) {

    int elements = constants::elements;
    int m_rows = filter_coefficients::f_bands_N * elements;
    int m_columns = filter_coefficients::filter_order +1 +2;

    double x_factor = sin(theta) * cos(phi);
    double y_factor = sin(theta) * sin(phi);

    const double a_0 = 1.0;

    const int P = filter_coefficients::filter_order;

    for (int freq_ind = 0; freq_ind < filter_coefficients::f_bands_N; freq_ind++)
    {

        // Center frequency
        double frequency = filter_coefficients::center_frequencies[freq_ind];

        // Normalized frequency
        double ny = frequency/((double)(constants::f_sampling));

        // Narrow-band wave vector 
        double k = 2*constants::pi * frequency/ constants::c;

        // Weight index 
        int w_index = weight_index(frequency)-1;

        for (int mic_ind = 0; mic_ind < elements; mic_ind++)
        {
            if (weight_matrix[elements*w_index + mic_ind] == 1)
            {                
                // FIlter coefficients for each band 
                filter_coefficients::filt_coeffs[freq_ind];

                // Row index 
                int row_index = freq_ind*elements + mic_ind;
                
                // Phase shift value theta is dependent on the frequency and the location of the element (x,y)
                double phi_0 = -k*(r_prime[mic_ind]*x_factor + r_prime[mic_ind + elements]*y_factor);

                // Calculation coefficients
                double A = sin(phi_0)/(4*constants::pi*ny*a_0);
                double B = cos(phi_0)/a_0;

                // Calculation of the mega filter coefficients!
                f_mega_coefficients[row_index*m_columns + 0] = A* filter_coefficients::filt_coeffs[freq_ind][0];
                f_mega_coefficients[row_index*m_columns + 1] = B*filter_coefficients::filt_coeffs[freq_ind][0] +  A* filter_coefficients::filt_coeffs[freq_ind][1];

                for (int i = 2; i <= P ; i++)
                {
                    f_mega_coefficients[row_index*m_columns + i] = B*filter_coefficients::filt_coeffs[freq_ind][i-1] + A*(filter_coefficients::filt_coeffs[freq_ind][i] - filter_coefficients::filt_coeffs[freq_ind][i-2]);
                }

                f_mega_coefficients[row_index*m_columns + P+1] = (B*filter_coefficients::filt_coeffs[freq_ind][P] - A*filter_coefficients::filt_coeffs[freq_ind][P-1]); 
                f_mega_coefficients[row_index*m_columns + P+2] = - A*filter_coefficients::filt_coeffs[freq_ind][P];    
  
            }   
        }   
    }
}


/*
Cuda Fir Filter for a single mic
*/
__global__ void cuFirFilter(const float *d_x, float *d_filter, float *d_y, const int filterLength, const int d_yLength){
    float sum;
    __shared__ float filt[200];
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < d_yLength*64)
    {   
        for (int l = 0; l < 64; l++)
        {
            for (int k = 0; k < 45; k++)
            {
                
                if(threadIdx.x < filterLength)
                    filt[threadIdx.x] = d_filter[threadIdx.x+filterLength*k];
                __syncthreads();

                for (int j = 0; j < d_yLength && j < filterLength; j++)
                    {
                        sum += filt[j] * d_x[i-j];
                    }

                d_y[i] = sum;
            }
        }
    }
}

/*             if(threadIdx.x < filterLength)
                filt[threadIdx.x] = d_filter[threadIdx.x+filterLength*j];
            __syncthreads();
 */

__global__ void cuFirFilterV2(const float *d_x, float *d_filter, float *d_y, const int filterLength, const int d_yLength){
    float sum;
    __shared__ float filt[202];
    int idx = 2880;
    int elems = 64;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = tid/d_yLength; 
    
    if( tid < d_yLength*elems){

        for(int j = 0; j < 64; j++){
            
            for(int k = 0; k < 45; k++){
                if(threadIdx.x < filterLength){
                    filt[threadIdx.x] = d_filter[threadIdx.x+filterLength*k+45*filterLength*j];
                }
                __syncthreads();
                
                for (int i = 0; i < d_yLength && i < filterLength; i++)
                {
                    sum += filt[i] * d_x[tid-i];
                }
                d_y[tid] = sum;
            }
        }
    }
}


int main() {
    double r_prime[3*(constants::elements)] = {0};      //initiazte r_prime full of 0s
    generate_array_r_prime(r_prime);                    //Generate r_prime     
    std::vector<float> audio_data;
    generate_emulated_data(audio_data,r_prime);
    std::cout << "Audio_matrix size: " << audio_data.size() << std::endl;

    int weight_matrix[constants::elements * constants::available_modes] = {0};
    std::cout << "weight_matrix size: " << constants::elements * constants::available_modes << std::endl;

    generate_weight_matrix(weight_matrix);

    int mega_f_size = filter_coefficients::f_bands_N * constants::elements * (filter_coefficients::filter_order + 1 + 2); 
    std::cout << "mega_f_zise: " << mega_f_size << std::endl;

    float mega_f_coefficients[mega_f_size] = {0};

    auto start3 = std::chrono::high_resolution_clock::now();
    generate_mfilter_coefficients(mega_f_coefficients,r_prime,weight_matrix,0.3,1.32);
    auto end3 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> float_ms3 = end3 - start3;
    std::cout << "Generating Mfilter elapsed time is " << float_ms3.count() << " milliseconds" << std::endl;
    int column_size = filter_coefficients::filter_order + 3;

/*     int dataLength = 32199;
    int totDL = constants::elements * dataLength;
    int coeffs = 202;
    int dataOutLen = 32199;
    
    float *h_filteredData = new float[totDL]{0};

    float *d_data = nullptr;
    cudaMalloc((void **)&d_data, totDL * sizeof(float));

    float *d_filter = nullptr;
    cudaMalloc((void **)&d_filter, mega_f_size * sizeof(float));

    float *d_filteredData = nullptr;
    cudaMalloc((void **)&d_filteredData, totDL * sizeof(float));

    cudaMemcpy(d_data, audio_data.data(), totDL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, mega_f_coefficients, mega_f_size * sizeof(float), cudaMemcpyHostToDevice);  

    int threadsPerBlock = 256;
    int blocksPerGrid = (totDL + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Work started: " << blocksPerGrid*threadsPerBlock << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();

    cuFirFilterV2<<<blocksPerGrid,threadsPerBlock>>>(d_data, d_filter, d_filteredData, coeffs, dataOutLen);
    
    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_filteredData, d_filteredData, totDL * sizeof(float), cudaMemcpyDeviceToHost);

 */
    int dataLength = 32200;
    int coeffs = 200;
    int totDL = constants::elements * dataLength;
    int totCoeffs = coeffs*45;
    int dataOutLen = 32200;
    
    size_t NumberOfElements = sizeof(filter_coefficients::filt_coeffs[0])/sizeof(filter_coefficients::filt_coeffs[0][0]);
    std::cout << "size of : " << NumberOfElements << std::endl;

    float *h_filteredData = new float[totDL];
    float *d_data = nullptr;
    cudaMalloc((void **)&d_data, totDL * sizeof(float));

    float *d_filter = nullptr;
    cudaMalloc((void **)&d_filter, totCoeffs * sizeof(float));

    float *d_filteredData = nullptr;
    cudaMalloc((void **)&d_filteredData, totDL * sizeof(float));

    cudaMemcpy(d_data, audio_data.data(), dataLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter_coefficients::filt_coeffs, totCoeffs * sizeof(float), cudaMemcpyHostToDevice);  

    int threadsPerBlock = 256;
    int blocksPerGrid = (totDL + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Threads started: " << blocksPerGrid*threadsPerBlock << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();

    cuFirFilter<<<blocksPerGrid,threadsPerBlock>>>(d_data, d_filter, d_filteredData, coeffs, dataOutLen);
    
    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_filteredData, d_filteredData, dataOutLen * sizeof(float), cudaMemcpyDeviceToHost);

 

    std::chrono::duration<double, std::milli> float_ms = end - start;
    std::cout << "CuFirFilter elapsed time is " << float_ms.count() << " milliseconds" << std::endl;
    // SINGLE DIRECTION BEAMFORMING 
    // TESTING  
    std::cout << "\n ";
    //std::cout << audio_signal_temp[16000];
    std::cout << "\n ";


    for (int i = 195; i < 206; i++)
    {
        std::cout << "Raw data: "<<  audio_data[i] << " Filtered data: ";
        std::cout << h_filteredData[i];
        std::cout << "\n ";
    }    
    for (int i = 32394; i < 32404; i++)
    {
        std::cout << "Raw data: "<<  audio_data[i] << " Filtered data: ";
        std::cout << h_filteredData[i];
        std::cout << "\n ";
    }
    for (int i = 64593; i < 64603; i++)
    {
        std::cout << "Raw data: "<<  audio_data[i] << " Filtered data: ";
        std::cout << h_filteredData[i];
        std::cout << "\n ";
    }
 
    cudaFree(d_data);
    cudaFree(d_filter);
    cudaFree(d_filteredData);

    delete [] h_filteredData;


    return 0;
}