/*
Define the gpt-2 Tokenizer
Only support decoding tokens (integers) -> strings
If we wanted to later prompt the model, we'd have to add decoding.
*/
#include<stdint.h>
#include<ctype.h>
#include<assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck