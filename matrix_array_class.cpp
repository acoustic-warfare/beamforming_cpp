#include "matrix_array_class.h"
#include <iostream>
#include <array>


Matrix_array::Matrix_array( int row_elements_, int column_elements_, double uni_distance_,double *r_a_ ) {
  elements = row_elements_*column_elements_;
  row_elements = row_elements_;
  column_elements = column_elements_;
  uni_distance = uni_distance_;
  r_a = r_a_;

};

void Matrix_array::create_array() {  // Method/function defined inside the class

      int element_index = 0;

      for (int i = 0; i < row_elements; i++)
      {

        for (int j = 0; j < column_elements; j++)
        { 
          r_prime[0][element_index] = i*uni_distance + r_a[0];
          r_prime[1][element_index] = j*uni_distance + r_a[1];

          //std::cout << r_prime[0][element_index];

          element_index +=1;
        }
      }

      for (int i = 0; i < row_elements * column_elements; i++)
      {
        r_prime[0][i] -= row_elements*uni_distance/2 + uni_distance/2;
        r_prime[1][i] -= column_elements*uni_distance/2 + uni_distance/2;
      }
    };