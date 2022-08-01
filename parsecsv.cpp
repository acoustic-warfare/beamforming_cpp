#include <iostream>
#include <fstream>
#include <string>
#include <vector>


/* std::vector<std::string> getNextLineAndSplitIntoTokens()
{
    
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ','))
    {
        result.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
} */

int main (){
    std::ifstream file;
    file.open("testfile.txt");
    if(!file) { // file couldn't be opened
      std::cout << "Error: file could not be opened" << std::endl;
      return 1;
    }
    std::string str;
    std::vector<int> result;

    while(std::getline(file, str, ','))
    {
        result.push_back(stoi(str));
        //std::cout << "The next number is " << stoi(str) << std::endl;
    }
    
    std::cout << result[1] << std::endl; 

    return 0;
}

