#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*Texts)[];
typedef int (*Vector)[];
typedef int (*Vectors)[];

typedef struct {
    char words[1000];
    int count;
}VocabStore;
char** split_(const char* str, int* numParts) {
    char** parts = NULL;
    char* token;
    int count = 0;
    char* strCopy = strdup(str);
    if (strCopy == NULL) {
        *numParts = 0;
        return NULL;
    }
    token = strtok(strCopy, " ");
    while (token != NULL) {
        parts = realloc(parts, (count + 1) * sizeof(char*));
        if (parts == NULL) {
            *numParts = 0;
            free(strCopy);
            return NULL;
        }

        parts[count] = strdup(token);
        if (parts[count] == NULL) {
            *numParts = 0;
            free(strCopy);
            free(parts);
            return NULL;
        }

        count++;
        token = strtok(NULL, " ");
    }
    free(strCopy);
    *numParts = count;
    return parts;
}
int* c_transform(char* texts[], int vocab_store_size, int text_size,VocabStore vocab_store[]) {
    int* transformed_texts = malloc(text_size * sizeof(int));
    if (transformed_texts == NULL) {
        // Handle memory allocation failure
        return NULL;
    }
    
    for (int i = 0; i < text_size; i++) {
        transformed_texts[i] = -1; // Default value for missing words
        
        for (int j = 0; j < vocab_store_size; j++) {
            if (strcmp(texts[i], vocab_store[j].words) == 0) {
                transformed_texts[i] = vocab_store[j].count;
                break;
            }
        }
    }
    
    return transformed_texts;
}

int c_fit_vectorize(char* texts[], VocabStore vocab_store[]) {
    int vocab_store_size = sizeof(vocab_store) / sizeof(vocab_store[0]);
    int text_size = sizeof(texts) / sizeof(texts[0]);
    
    int* transformed_texts = c_transform(texts, vocab_store_size, text_size,vocab_store);
    
    // Perform other operations on the transformed texts
    
    free(transformed_texts); // Free allocated memory
    
    return 0; // Return success status or other meaningful result
}