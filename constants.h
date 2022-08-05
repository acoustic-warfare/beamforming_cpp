#ifndef CONSTANTS_H
#define CONSTANTS_H

// define your own namespace to hold constants
namespace constants
{

    // Array geometry constants
    constexpr int total_elements { 64 };
    constexpr int elements { 64 };
    constexpr int column_elements { 8 };
    constexpr int row_elements { 8 };
    constexpr double uni_distance { 0.02 };
    constexpr double r_ax { 0 };
    constexpr double r_ay { 0 };
    constexpr double r_az  { 0 };
    constexpr int available_modes {7};
    

    // Physical constants
    constexpr double c { 343 };

    // Mathematical constants
    constexpr double pi { 3.14159265359 };

    // Simulation constants
    constexpr int f_sampling { 16000 };
    constexpr double t_start { 0 };
    constexpr double t_end { 2 };


    // Data emulation constants
    constexpr int sources_N { 1 };

        // Sources
        constexpr double source_theta_deg[1] {45};
        constexpr double source_phi_deg[1] {180};

        constexpr double source_distance_away[1] {1};

        constexpr double source_frequency_span[1][2] {{2800,3500}};
        constexpr int source_frequency_N[1] { 40 };

        constexpr double source_t_start[1] { 0 };
        constexpr double source_t_end[1] { 1 };


    // Test constants
    constexpr double test_array[14] {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
    constexpr double test_array2[2][2] {
    {1,2,},
    {3,4}};

}
#endif