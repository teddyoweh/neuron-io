

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <errno.h>
#include "include/hashmap_base.h"
#define HASHMAP_SIZE_MIN                32
#define HASHMAP_SIZE_DEFAULT            128
#define HASHMAP_SIZE_MOD(map, val)      ((val) & ((map)->table_size - 1))
#define HASHMAP_PROBE_NEXT(map, index)  HASHMAP_SIZE_MOD(map, (index) + 1)


struct hashmap_entry {
    void *key;
    void *data;
};

static inline size_t hashmap_calc_table_size(const struct hashmap_base *hb, size_t size)
{
    size_t table_size;
    table_size = size + (size / 3);
    if (table_size < hb->table_size_init) {
        table_size = hb->table_size_init;
    } else {
        table_size = 1 << ((sizeof(unsigned long) << 3) - __builtin_clzl(table_size - 1));
    }

    return table_size;
}
static inline size_t hashmap_calc_index(const struct hashmap_base *hb, const void *key)
{
    size_t index = hb->hash(key);
    index = hashmap_hash_default(&index, sizeof(index));

    return HASHMAP_SIZE_MOD(hb, index);
}
static struct hashmap_entry *hashmap_entry_get_populated(const struct hashmap_base *hb,
        const struct hashmap_entry *entry)
{
    if (hb->size > 0) {
        for (; entry < &hb->table[hb->table_size]; ++entry) {
            if (entry->key) {
                return (struct hashmap_entry *)entry;
            }
        }
    }
    return NULL;
}
static struct hashmap_entry *hashmap_entry_find(const struct hashmap_base *hb,
    const void *key, bool find_empty)
{
    size_t i;
    size_t index;
    struct hashmap_entry *entry;

    index = hashmap_calc_index(hb, key);

    
    for (i = 0; i < hb->table_size; ++i) {
        entry = &hb->table[index];
        if (!entry->key) {
            if (find_empty) {
                return entry;
            }
            return NULL;
        }
        if (hb->compare(key, entry->key) == 0) {
            return entry;
        }
        index = HASHMAP_PROBE_NEXT(hb, index);
    }
    return NULL;
}
static void hashmap_entry_remove(struct hashmap_base *hb, struct hashmap_entry *removed_entry)
{
    size_t i;
    size_t index;
    size_t entry_index;
    size_t removed_index = (removed_entry - hb->table);
    struct hashmap_entry *entry;
    if (hb->key_free) {
        hb->key_free(removed_entry->key);
    }
    --hb->size;
    index = HASHMAP_PROBE_NEXT(hb, removed_index);
    for (i = 0; i < hb->size; ++i) {
        entry = &hb->table[index];
        if (!entry->key) {
            break;
        }
        entry_index = hashmap_calc_index(hb, entry->key);
       
        if (HASHMAP_SIZE_MOD(hb, index - entry_index) >
                HASHMAP_SIZE_MOD(hb, removed_index - entry_index)) {
            *removed_entry = *entry;
            removed_index = index;
            removed_entry = entry;
        }
        index = HASHMAP_PROBE_NEXT(hb, index);
    }
   
    memset(removed_entry, 0, sizeof(*removed_entry));
}


static int hashmap_rehash(struct hashmap_base *hb, size_t table_size)
{
    size_t old_size;
    struct hashmap_entry *old_table;
    struct hashmap_entry *new_table;
    struct hashmap_entry *entry;
    struct hashmap_entry *new_entry;

    assert((table_size & (table_size - 1)) == 0);
    assert(table_size >= hb->size);

    new_table = (struct hashmap_entry *)calloc(table_size, sizeof(struct hashmap_entry));
    if (!new_table) {
        return -ENOMEM;
    }
    old_size = hb->table_size;
    old_table = hb->table;
    hb->table_size = table_size;
    hb->table = new_table;

    if (!old_table) {
        return 0;
    }

   
    for (entry = old_table; entry < &old_table[old_size]; ++entry) {
        if (!entry->key) {
            continue;
        }
        new_entry = hashmap_entry_find(hb, entry->key, true);
       
        assert(new_entry != NULL);

       
        *new_entry = *entry;
    }
    free(old_table);
    return 0;
}


static void hashmap_free_keys(struct hashmap_base *hb)
{
    struct hashmap_entry *entry;

    if (!hb->key_free || hb->size == 0) {
        return;
    }
    for (entry = hb->table; entry < &hb->table[hb->table_size]; ++entry) {
        if (entry->key) {
            hb->key_free(entry->key);
        }
    }
}


void hashmap_base_init(struct hashmap_base *hb,
        size_t (*hash_func)(const void *), int (*compare_func)(const void *, const void *))
{
    assert(hash_func != NULL);
    assert(compare_func != NULL);

    memset(hb, 0, sizeof(*hb));

    hb->table_size_init = HASHMAP_SIZE_DEFAULT;
    hb->hash = hash_func;
    hb->compare = compare_func;
}


void hashmap_base_cleanup(struct hashmap_base *hb)
{
    if (!hb) {
        return;
    }
    hashmap_free_keys(hb);
    free(hb->table);
    memset(hb, 0, sizeof(*hb));
}


void hashmap_base_set_key_alloc_funcs(struct hashmap_base *hb,
    void *(*key_dup_func)(const void *),
    void (*key_free_func)(void *))
{
    hb->key_dup = key_dup_func;
    hb->key_free = key_free_func;
}


int hashmap_base_reserve(struct hashmap_base *hb, size_t capacity)
{
    size_t old_size_init;
    int r = 0;

   
    old_size_init = hb->table_size_init;

   
    hb->table_size_init = HASHMAP_SIZE_MIN;
    hb->table_size_init = hashmap_calc_table_size(hb, capacity);

    if (hb->table_size_init > hb->table_size) {
        r = hashmap_rehash(hb, hb->table_size_init);
        if (r < 0) {
            hb->table_size_init = old_size_init;
        }
    }
    return r;
}


int hashmap_base_put(struct hashmap_base *hb, const void *key, void *data)
{
    struct hashmap_entry *entry;
    size_t table_size;
    int r = 0;

    if (!key || !data) {
        return -EINVAL;
    }

   
    table_size = hashmap_calc_table_size(hb, hb->size);
    if (table_size > hb->table_size) {
        r = hashmap_rehash(hb, table_size);
    }

   
    entry = hashmap_entry_find(hb, key, true);
    if (!entry) {
       
        if (r < 0) {
           
            return r;
        }
        return -EADDRNOTAVAIL;
    }

    if (entry->key) {
       
        return -EEXIST;
    }

    if (hb->key_dup) {
       
        entry->key = hb->key_dup(key);
        if (!entry->key) {
            return -ENOMEM;
        }
    } else {
        entry->key = (void *)key;
    }
    entry->data = data;
    ++hb->size;
    return 0;
}


void *hashmap_base_get(const struct hashmap_base *hb, const void *key)
{
    struct hashmap_entry *entry;

    if (!key) {
        return NULL;
    }

    entry = hashmap_entry_find(hb, key, false);
    if (!entry) {
        return NULL;
    }
    return entry->data;
}


void *hashmap_base_remove(struct hashmap_base *hb, const void *key)
{
    struct hashmap_entry *entry;
    void *data;

    if (!key) {
        return NULL;
    }

    entry = hashmap_entry_find(hb, key, false);
    if (!entry) {
        return NULL;
    }
    data = entry->data;
   
    hashmap_entry_remove(hb, entry);
    return data;
}


void hashmap_base_clear(struct hashmap_base *hb)
{
    hashmap_free_keys(hb);
    hb->size = 0;
    memset(hb->table, 0, sizeof(struct hashmap_entry) * hb->table_size);
}


void hashmap_base_reset(struct hashmap_base *hb)
{
    struct hashmap_entry *new_table;

    hashmap_free_keys(hb);
    hb->size = 0;
    if (hb->table_size != hb->table_size_init) {
        new_table = (struct hashmap_entry *)realloc(hb->table,
                sizeof(struct hashmap_entry) * hb->table_size_init);
        if (new_table) {
            hb->table = new_table;
            hb->table_size = hb->table_size_init;
        }
    }
    memset(hb->table, 0, sizeof(struct hashmap_entry) * hb->table_size);
}
struct hashmap_entry *hashmap_base_iter(const struct hashmap_base *hb,
        const struct hashmap_entry *pos)
{
    if (!pos) {
        pos = hb->table;
    }
    return hashmap_entry_get_populated(hb, pos);
}
bool hashmap_base_iter_valid(const struct hashmap_base *hb, const struct hashmap_entry *iter)
{
    return hb && iter && iter->key && iter >= hb->table && iter < &hb->table[hb->table_size];
}
bool hashmap_base_iter_next(const struct hashmap_base *hb, struct hashmap_entry **iter)
{
    if (!*iter) {
        return false;
    }
    return (*iter = hashmap_entry_get_populated(hb, *iter + 1)) != NULL;
}
bool hashmap_base_iter_remove(struct hashmap_base *hb, struct hashmap_entry **iter)
{
    if (!*iter) {
        return false;
    }
    if ((*iter)->key) {
 
        hashmap_entry_remove(hb, *iter);
    }
    return (*iter = hashmap_entry_get_populated(hb, *iter)) != NULL;
}
const void *hashmap_base_iter_get_key(const struct hashmap_entry *iter)
{
    if (!iter) {
        return NULL;
    }
    return (const void *)iter->key;
}
void *hashmap_base_iter_get_data(const struct hashmap_entry *iter)
{
    if (!iter) {
        return NULL;
    }
    return iter->data;
}
int hashmap_base_iter_set_data(struct hashmap_entry *iter, void *data)
{
    if (!iter) {
        return -EFAULT;
    }
    if (!data) {
        return -EINVAL;
    }
    iter->data = data;
    return 0;
}
double hashmap_base_load_factor(const struct hashmap_base *hb)
{
    if (!hb->table_size) {
        return 0;
    }
    return (double)hb->size / hb->table_size;
}
size_t hashmap_base_collisions(const struct hashmap_base *hb, const void *key)
{
    size_t i;
    size_t index;
    struct hashmap_entry *entry;

    if (!key) {
        return 0;
    }

    index = hashmap_calc_index(hb, key);
    for (i = 0; i < hb->table_size; ++i) {
        entry = &hb->table[index];
        if (!entry->key) {
 
            return 0;
        }
        if (hb->compare(key, entry->key) == 0) {
            break;
        }
        index = HASHMAP_PROBE_NEXT(hb, index);
    }

    return i;
}
double hashmap_base_collisions_mean(const struct hashmap_base *hb)
{
    struct hashmap_entry *entry;
    size_t total_collisions = 0;

    if (!hb->size) {
        return 0;
    }
    for (entry = hb->table; entry < &hb->table[hb->table_size]; ++entry) {
        if (!entry->key) {
            continue;
        }

        total_collisions += hashmap_base_collisions(hb, entry->key);
    }
    return (double)total_collisions / hb->size;
}
double hashmap_base_collisions_variance(const struct hashmap_base *hb)
{
    struct hashmap_entry *entry;
    double mean_collisions;
    double variance;
    double total_variance = 0;

    if (!hb->size) {
        return 0;
    }
    mean_collisions = hashmap_base_collisions_mean(hb);
    for (entry = hb->table; entry < &hb->table[hb->table_size]; ++entry) {
        if (!entry->key) {
            continue;
        }
        variance = (double)hashmap_base_collisions(hb, entry->key) - mean_collisions;
        total_variance += variance * variance;
    }
    return total_variance / hb->size;
}
size_t hashmap_hash_default(const void *data, size_t len)
{
    const uint8_t *byte = (const uint8_t *)data;
    size_t hash = 0;

    for (size_t i = 0; i < len; ++i) {
        hash += *byte++;
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}
size_t hashmap_hash_string(const char *key)
{
    size_t hash = 0;

    for (; *key; ++key) {
        hash += *key;
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}
size_t hashmap_hash_string_i(const char *key)
{
    size_t hash = 0;

    for (; *key; ++key) {
        hash += tolower(*key);
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}