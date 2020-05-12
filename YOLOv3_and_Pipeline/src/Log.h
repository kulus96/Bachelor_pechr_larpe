#ifndef LOG_H
#define LOG_H


#include <stdio.h>
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif
FILE* create_file(char name);
void write_to_log(FILE *log, float loss, float avloss, int incl_map, float map);
void close_log(FILE *log);
void write_to_log_map(FILE *log, float map);
#ifdef __cplusplus
}
#endif

#endif