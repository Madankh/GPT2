/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H
// #include<unistd.h>
#include<string>
#include<stdio.h>
#include<stdlib.h>
#include<sys/stat.h>
// implementation of dirent for windows in dev/unistd.h
#ifndef _WIN32
#include<direct.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck 
 

extern inline File *fopen_check(const char *path, const char *mode,  const char *file)
#endif