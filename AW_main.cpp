#include <iostream>
#include <array>
#include <chrono>
#include <thread>
#include <vector>
#include <math.h> 
#include <numeric>
#include <functional>
#include <immintrin.h>
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

void filter(double * y, double * x, int x_length, const double * b, double a_0, const int P) {

    int n;
    int i;

    for (n = 0; n < x_length; n++)
    {
        //x[n] = std::inner_product(series1, series1 + n, series2, 0.0);
        for (i = 0; i <= n && i < P; i++)
        {
            y[n] += b[i]*x[n-i];
        }
    }
}

void generate_emulated_data(std::vector<double>& audio_data, double * r_prime) {

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
            double r_1[3] = {r_prime[mic],r_prime[mic + elements],r_prime[mic + 2*elements]};
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

void test_function(float *x,float * y,float *b) {
    int x_length = 32200;
    int P = 200;

    constexpr auto AVX_FLOAT_COUNT = 8u;

    std::array<float, AVX_FLOAT_COUNT> outStore;


  for (auto n = 200u; n < x_length; ++n) {
    // Set a SIMD register to all zeros;
    // we will use it as an accumulator
    auto outChunk = _mm256_setzero_ps();

    // Note the increment
    for (auto j = 0u; j < 200; j += AVX_FLOAT_COUNT) {

      //int test = (std::inner_product(x + n-P+1, x + 1 + n , b , 0.0));

      // Load the unaligned input signal data into a SIMD register
      auto xChunk = _mm256_loadu_ps(x + n-P+1 + j);
      // Load the unaligned reversed filter coefficients 
      // into a SIMD register
      auto cChunk = _mm256_loadu_ps(b + j);

      // Multiply the both registers element-wise
      auto temp = _mm256_mul_ps(xChunk, cChunk);

      // Element-wise add to the accumulator
      outChunk = _mm256_add_ps(outChunk, temp);
    }

    // Transfer the contents of the accumulator 
    // to the output array
    _mm256_storeu_ps(outStore.data(), outChunk);

    // Sum the partial sums in the accumulator and assign to the output
    y[n] = std::accumulate(outStore.begin(), outStore.end(), 0.f);
  }
}


void filter_improved(double* y, double * x,int x_length,double a_0,const double * b, int P) {
    int n = 0;
    int i;
    int first;
    int b_first;
    int last;
    /*
    std::cout << "\n ... ";
    std::cout << x[0];
    std::cout << "\n ... ";
    std::cout << b[0];
    std::cout << "\n ... ";
    */

    for (n = P;n < x_length; n++)
    {       

        y[n] = 1/a_0 *(std::inner_product(x + n-P+1, x + 1 + n , b , 0.0));
        //x[n] = 1/a_0 * std::transform_reduce(x_temp + first, x_temp + last+1, b + b_first, 0.0, std::plus<>(), std::multiplies<>());
        
        /*
        std::cout << "\nn= ";
        std::cout << n;
        std::cout << ", x= ";
        std::cout << x[n];
        std::cout << ", y= ";
        std::cout << y[n];
        */
        
    }
}

int main() {
    //std::cout << "Hello World!";

    /*/////////////////////////////////////////////////// 


            GENERATE r_prime


    *////////////////////////////////////////////////////

    double r_prime[3*(constants::elements)] = {0};      //initiazte r_prime full of 0s
    generate_array_r_prime(r_prime);                    //Generate r_prime     

    double test[10] = {1,2,3,4,5,6,7,8,9,10};
    int test_length = 10;

    double b[14] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
    int P = 14;

    double a_0 = 0.7;

    //filter(test,test_length,a_0,b,P);

    for (int i = 0; i < test_length; i++)
    {
        //std::cout << "\n";
        //std::cout << test[i];
        
    }
    //std::cout << constants::test_array[4];
    
    //std::cout << r_prime[0];
    //std::cout << filter_coefficients::filt_coeffs[0][0];

    // Generate emulated data
    std::vector<double> audio_data;

    /*/////////////////////////////////////////////////// 


            GENERATE EMULATED DATA


    *////////////////////////////////////////////////////

    generate_emulated_data(audio_data,r_prime);

    
    for (int i = 32399; i < 32403; i++)
    {
        std::cout << "\n ";
        std::cout << audio_data[i];
        std::cout << "\n ";
    }
    
    
    int samples = audio_data.size()/constants::elements;

    // SINGLE DIRECTION BEAMFORMING 
    std::vector<double> audio_out;

    double audio_out2[32000 + 200] = {0};

    double audio_signal_temp[samples] = {0}; 
    double audio_filtered[samples] = {0};

    float audio_signal_temp2[samples] = {0}; 
    float audio_filtered2[samples] = {0};
    float filter_coefficients2[200] = {0};

    for (int i = 0; i < 200; i++)
    {
        filter_coefficients2[i] = (float)filter_coefficients::filt_coeffs[0][i];
    }
    

    //double test_filt_b[6] = {0.5, -0.5, 0.7, 0.2, 0.1, 0.3};
    double test_filt_b[6] = {0.3, 0.1, 0.2, 0.7, -0.5, 0.5};


    for (int i = 0; i < samples; i++)
    {
        audio_signal_temp[i] = audio_data[i];
        audio_signal_temp2[i] = (float)audio_data[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    //filter(audio_signal_temp,samples,1,filter_coefficients::filter_order);
    //filter_improved(audio_filtered,audio_signal_temp,samples,1,filter_coefficients::filt_coeffs[0],200);
    //filter_improved(audio_filtered,audio_signal_temp,samples,1,test_filt_b,6);
    test_function(audio_signal_temp2 , audio_filtered2 , filter_coefficients2);
    //filter(audio_filtered,audio_signal_temp,samples,filter_coefficients::filt_coeffs[0],1,filter_coefficients::filter_order);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> float_ms = end - start;

    std::cout << "funcSleep() elapsed time is " << float_ms.count() << " milliseconds" << std::endl;


    // TESTING 
    double series1[3] = {1,2,1};
    double series2[3] = {10,10,20.1};
    int n = sizeof(series1) / sizeof(double);
    std::cout << "\n ";
    //std::cout << audio_signal_temp[16000];
    std::cout << "\n ";
    

    for (int i = 16199; i < 16206; i++)
    {
        std::cout << "\n ";
        std::cout << audio_filtered2[i];
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

   
    return 0;
}