#include "utilities_cpp.h"

char * str_to_char(std::string intermidiate)
{
    char* char_return = new char(intermidiate.size());
    intermidiate.copy(char_return, intermidiate.size());
    char_return[intermidiate.size()] = '\0';

    return char_return;
}
const char* str_to_char_noPointer(std::string intermidiate)
{
    char* char_return = new char(intermidiate.size());
    intermidiate.copy(char_return, intermidiate.size());
    char_return[intermidiate.size()] = '\0';

    return char_return;
}

std::vector<char*> list_dir(char *path_in)
{
    FILE *fpipe;
    char c = 0;
    std::string inter_string = path_in;
    std::string intermediate;
    std::vector<char*> files_in;
    std::vector<char*> files_out;
    std::string command = "ls ";
    command.append(path_in);
    const char *test = command.c_str();    
    fpipe = (FILE*)popen(test,"r");

    while(fread(&c, sizeof(c),1,fpipe))
    {  
        
        if(c == '\n' && intermediate.size() > 4)
        {
            char *file_name = str_to_char(intermediate);
            files_in.push_back(file_name);
            intermediate.clear();
            inter_string = path_in;
            
        }
        else{
            intermediate += c; 
        }       
    }

    fclose(fpipe);
    return files_in;
}