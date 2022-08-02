#ifndef MATRIX_ARRAY_H
#define MATRIX_ARRAY_H
#include "constants.h"

class Matrix_array 
{      
public:          
    int elements;
    int row_elements;
    int column_elements;
    double uni_distance;
    int myNum;        
    double r_prime [3][constants::elements] = {0}; 
    double *r_a;

    void create_array();
    Matrix_array( int row_elements_, int column_elements_, double uni_distance_,double *r_a_ );
};

#endif 