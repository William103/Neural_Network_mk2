#include "threads.h"

void *thread_func(void *ID_arg) {
    int ID = *((int*)ID_arg);
}
