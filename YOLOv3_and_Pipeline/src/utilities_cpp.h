#ifndef UTILITIES_CPP_H
#define UTILITIES_CPP_H
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

char * str_to_char(std::string intermidiate);
const char* str_to_char_noPointer(std::string intermidiate);
std::vector<char*> list_dir(char *path_in);

#endif