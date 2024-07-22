/*
Define the gpt-2 Tokenizer
Only supports decoding tokens (integers) -> strings
This is all we need for unconditional generation
If we wanted to later prompt the model we'd have to add decoding
Which could be tricky in c because of the regex involved , to loo into later
*/

#include<stdint.h>
#include<ctype.h>
#include<assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <| endoftext|> token id
} Tokenizer;

void safe_printf(const char *piece){
    // the tokens are raw bytes and we only want to print the printable ones
    // many bytes can be various control codes , backspace etc
    if (piece == NULL) {return;}
    if(piece[0] == '\0') {return;}
    // handle individual byte
    // every token is asserted to be at least one byte so doing pices[1] is ok
    if(piece[1] == '\0'){
        unsigned char byte_val = piece[0];
        if(!(isprint(byte_val) || isspace(byte_val))){
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename){
    FILE *file = fopen(filename, "rb");
    if (file == NULL){
        // try to be more helpful as we just added this feature, erase later
        printf("--\n");
        printf("WARNING : Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added");
        printf("Re-run python train_gpt2.py to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }

    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if(version == 1){
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT;
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    }else if (version == 2){
        tokenizer->eot_token = header[3];
    }else{
        fprintf(stderr, "Tokenizer model file %s has bad version : %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    
}