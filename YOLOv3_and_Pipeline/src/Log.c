#include "Log.h"

FILE* create_file(char name)
{

    FILE *log = fopen(name,"a");
    return log;
}
void write_to_log(FILE *log, float loss, float avloss, int incl_map, float map)
{
    float map_mod = map;
    if(log != NULL)
    {
        if(incl_map != 0)
        {
            if(map == -1)
            {
                map_mod = 0;
            }
            fprintf(log,"%f %f %f\n",loss,avloss, map_mod);
        }
        else
        {
            fprintf(log,"%f %f\n",loss,avloss);
        }
        
    }
    else
    {
        printf("File is not opened! \n");
    } 
}

void close_log(FILE *log)
{
    fclose(log);
}

void write_to_log_map(FILE *log, float map)
{
    if(log != NULL)
    {
        fprintf(log,"%f \n",map);
    }
    else
    {
        printf("File is not opened! \n");
    }
    
}
