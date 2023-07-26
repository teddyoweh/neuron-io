#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HASH_MAP_SIZE 1000

struct HashMap VocabStore;
struct Array{
    char **values;
    int length;
};
 
struct Array array_make(void **elements, int count)
{
    struct Array ret;
    ret.values = malloc(sizeof(void *) * count);
    ret.length = count;

    for (int i = 0; i < count; i++) {
        ret.values[i] = elements[i];
    }

    return ret;
}

struct Node
{
    char* key;
    int value;
    struct Node* next;
};

struct HashMap
{
    struct Node* table[HASH_MAP_SIZE];
};

unsigned int hash(const char* key)
{
    unsigned int hashval = 0;
    for (int i = 0; key[i] != '\0'; i++)
    {
        hashval = key[i] + (hashval << 5) - hashval;
    }
    return hashval % HASH_MAP_SIZE;
}

struct Node* createNode(const char* key, int value)
{
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->key = strdup(key);
    newNode->value = value;
    newNode->next = NULL;
    return newNode;
}

void initializeHashMap(struct HashMap* hashMap)
{
    for (int i = 0; i < HASH_MAP_SIZE; i++)
    {
        hashMap->table[i] = NULL;
    }
}

void hashmap_put(struct HashMap* hashMap, const char* key, int value)
{
    unsigned int index = hash(key);
    struct Node* newNode = createNode(key, value);

    if (hashMap->table[index] == NULL)
    {
        hashMap->table[index] = newNode;
    }
    else
    {
        struct Node* current = hashMap->table[index];
        while (current != NULL)
        {
            if (strcmp(current->key, key) == 0)
            {
                current->value = value; // Update the value if the key already exists
                free(newNode->key);
                free(newNode);
                return;
            }
            current = current->next;
        }
        // Key not found, add a new node to the list
        newNode->next = hashMap->table[index];
        hashMap->table[index] = newNode;
    }
}

int hashmap_get(struct HashMap* hashMap, const char* key)
{
    unsigned int index = hash(key);
    struct Node* current = hashMap->table[index];

    while (current != NULL)
    {
        if (strcmp(current->key, key) == 0)
        {
            return current->value;
        }
        current = current->next;
    }
    return 0;
}

void hashmap_update(struct HashMap* hashMap, const char* key, int value)
{
    hashmap_put(hashMap, key, value); // Call hashmap_put to update or add the key-value pair
}

 
 

#define array(elements...) ({ void *values[] = { elements }; array_make(values, sizeof(values) / sizeof(void *)); })
#define destroy(arr) ({ free(arr.values); })

struct Array split_string(const char *str, const char *delimiter) {
    struct Array arr;
    arr.values = NULL;
    arr.length = 0;

    char *token;
    char *copy = strdup(str);  
    token = strtok(copy, delimiter);

    while (token != NULL) {
        arr.values = realloc(arr.values, sizeof(char *) * (arr.length + 1));
        arr.values[arr.length] = strdup(token);
        arr.length++;
        token = strtok(NULL, delimiter);
    }

    free(copy); 
    return arr;
}
int is_element_in_set(const struct Array *set, const char *element) {
    for (int i = 0; i < set->length; ++i) {
        if (strcmp(set->values[i], element) == 0) {
            return 1;  
        }
    }
    return -1;  
}
void add_element_to_set(struct Array *set, const char *element) {
    if (!is_element_in_set(set, element)) {
        set->values = realloc(set->values, sizeof(char *) * (set->length + 1));
        set->values[set->length] = strdup(element);
        set->length++;
    }
}

void print_set(const struct Array *set) {
    for (int i = 0; i < set->length; ++i) {
        printf("%s\n", set->values[i]);
    }
}

void destroy_set(struct Array *set) {
    for (int i = 0; i < set->length; ++i) {
        free(set->values[i]);
    }
    free(set->values);
    set->length = 0;
}
int c_fit(struct Array arr){
 for (int i=0 ;i<arr.length;++i){
 
    struct Array set;
    set.values = NULL;
    set.length = 0;


    struct Array splitArr = split_string(arr.values[i], " ");
  
    for (int j = 0; j < splitArr.length; ++j) {
        int oldElem = hashmap_get(&VocabStore,splitArr.values[j]);
        hashmap_put(&VocabStore,splitArr.values[j],oldElem+1);

            
    }
//destroy(splitArr);
 }
 return hashmap_get(&VocabStore, "all,") ;
}
 
int main() {
;
    initializeHashMap(&VocabStore);
    struct Array arr= array("im all getting on borderlands and i will murder you all",
  
 "I am coming to the borders and I will kill you all,",
 "im getting on borderlands and i will kill you all,",
 "im coming on borderlands and i will murder you all,",
 "im getting on borderlands 2 and i will murder you me all,",
 "im getting into borderlands and i can murder you all,");
    c_fit(arr);
    printf("Value for 'apple': %d\n", hashmap_get(&VocabStore, "all,"));
    
    return 0;
}
