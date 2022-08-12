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

void filter(double * x,int x_length,double a_0, int P) {
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
}


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
/* 
void AW_listening_improved(double * audio_out, std::vector<double>& audio_data, double * r_prime,double theta_listen,double phi_listen) {
    double theta = theta_listen * constants::pi/180;
    double phi = phi_listen * constants::pi/180;

    double x_factor = sin(theta)*cos(phi);
    double y_factor = sin(theta)*sin(phi);

    int samples = audio_data.size()/constants::elements;

    for (int freq_ind = 0; freq_ind < filter_coefficients::f_bands_N; freq_ind++)
    {
        std::cout << "\n";
        std::cout << freq_ind;

        double b_temp[filter_coefficients::filter_order] = {0};
        for (int i = 0; i < filter_coefficients::filter_order; i++)
        {
            b_temp[i] = filter_coefficients::filt_coeffs[freq_ind][i];
        }
         
        double frequency = filter_coefficients::center_frequencies[freq_ind];
        double k = 2 * constants::pi * frequency/constants::c;
        double ny = frequency/constants::f_sampling;
        double audio_temp[samples] = {0};   

        for (int mic_ind = 0; mic_ind < constants::elements; mic_ind++)
        {
            double audio_signal_temp[samples] = {0};   
            for (int i = 0; i < samples; i++)
            {
                audio_signal_temp[i] = audio_data[mic_ind + constants::elements*i];
            }
            
            //filter(audio_signal_temp,samples,1,b_temp,filter_coefficients::filter_order);

            auto start = std::chrono::high_resolution_clock::now();
            //filter(audio_signal_temp,samples,1,b_temp,filter_coefficients::filter_order);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> float_ms = end - start;

            std::cout << "funcSleep() elapsed time is " << float_ms.count() << " milliseconds" << std::endl;

            for (int i = 0; i < samples; i++)
            {
                audio_temp[i] += audio_signal_temp[i];
            }

        }
        

        for (int i = 0; i < samples; i++)
        {
                audio_out[i] += audio_temp[i];
        }

    }
    
}
//  */
// void filter_improved(double* y, double * x,int x_length,double a_0,const double * b, int P) {
//     int n = 0;
//     int i;
//     int first;
//     int b_first;
//     int last;
//     /*
//     std::cout << "\n ... ";
//     std::cout << x[0];
//     std::cout << "\n ... ";
//     std::cout << b[0];
//     std::cout << "\n ... ";
//     */

//     for (n = P;n < x_length; n++)
//     {       

//         y[n] = 1/a_0 *(std::inner_product(x + n-P+1, x + 1 + n , b , 0.0));
//         //x[n] = 1/a_0 * std::transform_reduce(x_temp + first, x_temp + last+1, b + b_first, 0.0, std::plus<>(), std::multiplies<>());
        
//         /*
//         std::cout << "\nn= ";
//         std::cout << n;
//         std::cout << ", x= ";
//         std::cout << x[n];
//         std::cout << ", y= ";
//         std::cout << y[n];
//         */
        
//     }
// }


/* 
CudaFirFilter. Does 45 sets(bands) of coefficients for a mic.
*/
__global__ void cuFirFilter(const float *d_x, float *d_filter, float *d_y, const int filterLength, const int d_yLength){
    float sum;
    __shared__ float filt[200];
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < d_yLength)
    {     
        for (size_t k = 0; k < 45; k++)
        {
               
            if(threadIdx.x < filterLength)
                filt[threadIdx.x] = d_filter[threadIdx.x+filterLength*k];
           /*  if(i == 0)
                 printf("Y %f", filt[i]);
            */
            __syncthreads();

                for (int j = 0; j < d_yLength && j < filterLength; j++)
                {
                    sum += filt[j] * d_x[i-j];
                }

            d_y[i] = sum;
        }
    }
}


int main() {
    double r_prime[3*(constants::elements)] = {0};      //initiazte r_prime full of 0s
    generate_array_r_prime(r_prime);                    //Generate r_prime     
    std::vector<float> audio_data;
    generate_emulated_data(audio_data,r_prime);
    
    //32000 samples and 200 padded zeros.
    int dataLength = 32200;
    int coeffs = 200;
    int totCoeffs = coeffs*45;
    int dataOutLen = 32200;
    
    size_t NumberOfElements = sizeof(filter_coefficients::filt_coeffs[0])/sizeof(filter_coefficients::filt_coeffs[0][0]);
    std::cout << "size of : " << NumberOfElements << std::endl;

    float *h_filteredData = new float[dataOutLen];
    float *d_data = nullptr;
    cudaMalloc((void **)&d_data, dataLength * sizeof(float));

    float *d_filter = nullptr;
    cudaMalloc((void **)&d_filter, totCoeffs * sizeof(float));

    float *d_filteredData = nullptr;
    cudaMalloc((void **)&d_filteredData, dataOutLen * sizeof(float));

    cudaMemcpy(d_data, audio_data.data(), dataLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter_coefficients::filt_coeffs, totCoeffs * sizeof(float), cudaMemcpyHostToDevice);  

    int threadsPerBlock = 256;
    int blocksPerGrid = (dataOutLen + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Threads started: " << blocksPerGrid*threadsPerBlock << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();

        cuFirFilter<<<blocksPerGrid,threadsPerBlock>>>(d_data, d_filter, d_filteredData, coeffs, dataOutLen);
    
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_filteredData, d_filteredData, dataOutLen * sizeof(float), cudaMemcpyDeviceToHost);
    // SINGLE DIRECTION BEAMFORMING 
/*     std::vector<float> audio_out;
    std::vector<float> audio_signal_temp; 

    for (int i = 0; i < dataLength; i++)
    {
        audio_signal_temp.push_back(audio_data[0 + constants::elements*i]);
    }
 */



    std::chrono::duration<double, std::milli> float_ms = end - start;
    std::cout << "funcSleep() elapsed time is " << float_ms.count() << " milliseconds" << std::endl;
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
    /*
    double test_series[2][2] = {{1,2},{3,4}};
    double newtestomg = (std::inner_product(series1 +2 , series1+3,series2 +2, 0.0));

    std::cout << "\n ";
    std::cout << newtestomg;
    std::cout << "\n ";

    */


    //AW_listening_improved(audio_out2,audio_data,r_prime,0,0);
    /*
    auto start = std::chrono::high_resolution_clock::now();
    AW_listening_improved(audio_out2,audio_data,r_prime,0,0);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> float_ms = end - start;

    std::cout << "funcSleep() elapsed time is " << float_ms.count() << " milliseconds" << std::endl;
    */

    cudaFree(d_data);
    cudaFree(d_filter);
    cudaFree(d_filteredData);

    delete [] h_filteredData;


    return 0;
}
