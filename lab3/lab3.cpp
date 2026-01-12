#include <cmath>
#include <cfloat>
#include <climits>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>
using namespace std;

struct ProgramParams {
    string schedule = "";
    int chunk = 1;
    string output_file = "default.csv";
    int thread_num = 12;
};

#ifdef _OPENMP
#include <omp.h>

void check_schedule(string param_schedule) {
    if (param_schedule == "") {
        printf("Schedule: default, chunk size = default\n");
        return;
    }
    // const char *env = getenv("OMP_SCHEDULE");
    // if (env) {
    //     printf("Env OMP_SCHEDULE = \"%s\"\n", env);
    // } else {
    //     printf("Env OMP_SCHEDULE not set\n");
    // }
    omp_sched_t kind;
    int chunk;
    omp_get_schedule(&kind, &chunk);
    printf("Schedule: ");
    switch (kind) {
        case omp_sched_static:  printf("static"); break;
        case omp_sched_dynamic: printf("dynamic"); break;
        case omp_sched_guided:  printf("guided"); break;
        case omp_sched_auto:    printf("auto"); break;
        default:                printf("unknown(%d)", (int)kind); break;
    }
    printf(", chunk size = %d\n", chunk);
}

omp_sched_t parse_schedule_kind(const string &s) {
    if (s == "static")  return omp_sched_static;
    if (s == "dynamic") return omp_sched_dynamic;
    if (s == "guided")  return omp_sched_guided;
    if (s == "auto")    return omp_sched_auto;

    fprintf(stderr, "Unknown schedule '%s'. Using default (auto).\n", s.c_str());
    return omp_sched_auto;
}

void apply_schedule(const ProgramParams &params) {
    omp_sched_t kind = parse_schedule_kind(params.schedule);
    omp_set_schedule(kind, params.chunk);
}

#else
void check_schedule(string params) { return; }
void apply_schedule(const ProgramParams &) { return; }
void omp_set_nested(int enable) { return; }
int omp_get_thread_num() { return 0; }
int omp_get_num_procs() { return 2; }
void omp_set_num_threads(int thread_num) { return; }
int omp_get_max_threads() {return 1; }
#endif

ProgramParams parse_args(int argc, char **argv) {
    ProgramParams params;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--schedule") == 0 && i + 1 < argc) {
            params.schedule = argv[++i];
        } else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            params.chunk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            params.output_file = argv[++i];
        } else if (strcmp(argv[i], "--thread_num") == 0 && i + 1 < argc) {
            params.thread_num = atoi(argv[++i]);
        } else {
            printf("Unknown argument: %s\n", argv[i]);
        }
    }
    return params;
}

void append_to_csv(const string &filename, const vector<string> &values)
{
    ofstream file(filename, ios::app); // append mode
    if (!file.is_open()) {
        fprintf(stderr, "Cannot open CSV file '%s'\n", filename.c_str());
        return;
    }

    for (size_t i = 0; i < values.size(); ++i) {
        file << values[i];
        if (i + 1 < values.size())
            file << ",";
    }
    file << "\n";
}

// ---------- Generate ----------
double rand_uniform(unsigned int *seedp, double low, double high) {
    unsigned int r = rand_r(seedp);
    double unit = (double)r / (double)RAND_MAX;
    return low + unit * (high - low);
}

vector<double> generate_vector_random(unsigned int *seedp, size_t size, double low, double high) {
    vector<double> v;
    v.resize(size);
    #pragma omp parallel default(none) shared(seedp, low, high, v, size)
    {
        unsigned int loc_seed = *seedp;
        loc_seed += omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < size; ++i) {
            v[i] = rand_uniform(&loc_seed, low, high);
        }
    }
    return v;
}

vector<double> generate_vector_fixed(unsigned int *seedp, size_t size, double low, double high) {
    vector<double> v;
    v.resize(size);
    #pragma omp parallel default(none) shared(seedp, low, high, v, size)
    {
        #pragma omp for
        for (size_t i = 0; i < size; ++i) {
            unsigned int loc_seed = *seedp + static_cast<unsigned int>(i);
            v[i] = rand_uniform(&loc_seed, low, high);
        }
    }
    return v;
}

// vector<double> generate_vector_seq(unsigned int *seedp, size_t size, double low, double high) {
//     vector<double> v;
//     v.resize(size);
    
//     for (size_t i = 0; i < size; ++i) {
//         v[i] = rand_uniform(seedp, low, high);
//     }
//     return v;
// }


// ---------- Map ----------
void map_M1_sinh_square(vector<double> &M1) {
    #pragma omp parallel for default(none) shared(M1)
    for (size_t i = 0; i < M1.size(); ++i) {
        double s = sinh(M1[i]);
        M1[i] = s * s;
    }
}

void map_M2_tan_abs_with_prev(vector<double> &M2) {
    if (M2.empty()) return;
    vector<double> copy = M2;

    #pragma omp parallel for
    for (size_t i = 0; i < M2.size(); ++i) {
        double prev = (i == 0 ? 0.0 : copy[i - 1]);
        double val = copy[i] + prev;
        M2[i] = fabs(tan(val));
    }
}

// ---------- Merge ----------
void merge_pow(vector<double> &M1, vector<double> &M2) {
    size_t limit = min(M1.size(), M2.size());
    #pragma omp parallel for default(none) shared(M1, M2, limit)
    for (size_t i = 0; i < limit; ++i) {
        M2[i] = pow(M1[i], M2[i]);
    }
}

// ---------- Sort (Selection sort) ----------
void selection_sort_par_second_loop(vector<double> &arr) {
    size_t n = arr.size();
    for (size_t i = 0; i < n - 1; ++i) {
        size_t min_idx = i;
        #pragma omp parallel default(none) shared(i, n, arr, min_idx)
        {
            size_t local_min_idx = i;
            #pragma omp for
            for (size_t j = i + 1; j < n; ++j) {
                if (arr[j] < arr[local_min_idx]) local_min_idx = j;
            }

            #pragma omp critical
            {
                if (arr[local_min_idx] < arr[min_idx]) min_idx = local_min_idx;
            }
        }
        swap(arr[i], arr[min_idx]);
    }
}

void selection_sort_seq(vector<double>& arr, size_t l, size_t r) {
    for (size_t i = l; i < r; ++i) {
        size_t min_idx = i;
        for (size_t j = i + 1; j <= r; ++j) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        swap(arr[i], arr[min_idx]);
    }
}

vector<double> merge_sorted(const vector<double>& a, const vector<double>& b) {
    vector<double> res;
    res.reserve(a.size() + b.size());
    size_t i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j])
            res.push_back(a[i++]);
        else
            res.push_back(b[j++]);
    }
    while (i < a.size()) res.push_back(a[i++]);
    while (j < b.size()) res.push_back(b[j++]);

    return res;
}

void print_arr(vector<double> &arr) {
    cout << "Array: ";
    for (size_t i = 0; i < arr.size(); ++i) {
        cout << arr[i];
        cout << " ";
    }
    cout << "\n\n";
}

// ---------- Reduce ----------
double compute_reduce_sum(const vector<double> &arr, bool use_sort) {
    double minnz = DBL_MAX;
    if (!use_sort) {
        #pragma omp parallel for default(none) shared(arr) reduction(min:minnz)
        for (int i = 0; i < arr.size(); ++i) {
            if (!isfinite(arr[i])) continue;
            if (fabs(arr[i]) > DBL_EPSILON && arr[i] < minnz) minnz = arr[i];
        }
    } else {
        for (int i = 0; i < arr.size(); ++i) {
            if (fabs(arr[i]) > DBL_EPSILON) {
                minnz = arr[i];
                break;
            }
        }
    }

    double sum = 0.0;
    #pragma omp parallel for default(none) shared(arr, minnz) reduction(+:sum)
    for (double x : arr) {
        if (!isfinite(x)) continue;
        double qd = x / minnz;
        long long qi = LLONG_MAX;
        if (qd < static_cast<double>(LLONG_MAX)) {
            qi = static_cast<long long>(qd);
        }
        if (qi % 2 == 0) {
            sum += sin(x);
        }
    }
    return sum;
}

void selection_sort_partition_par(vector<double> &arr) {
    size_t n = arr.size();
    size_t chunks = 12;
    // if (n > 1000) {
    //     chunks = n / 1000;
    // }
    size_t mid = n / chunks;

    #pragma omp parallel for default(none) shared(mid, chunks, n, arr)
    for (int i = 0; i < chunks; i++) {
        size_t l = i * mid;
        size_t r = (i == chunks - 1) ? (n - 1) : (l + mid - 1);
        selection_sort_seq(arr, l, r);
    }

    vector<vector<double>> parts;
    parts.reserve(chunks);

    for (int i = 0; i < chunks; i++) {
        size_t l = i * mid;
        size_t r = (i == chunks - 1) ? (n - 1) : (l + mid - 1);
        parts.emplace_back(arr.begin() + l, arr.begin() + r + 1);
    }

    while (parts.size() > 1) {
        vector<vector<double>> new_parts;
        new_parts.reserve((parts.size() + 1) / 2);

        for (int i = 0; i < parts.size(); i += 2) {
            if (i + 1 < parts.size())
                new_parts.push_back(merge_sorted(parts[i], parts[i+1]));
            else
                new_parts.push_back(std::move(parts[i]));
        }
        parts = std::move(new_parts);
    }

    arr = std::move(parts[0]);
}

double start_algo(long long N_input, double A, bool fixed, bool use_sort) {
    size_t N = static_cast<size_t>(N_input);
    size_t N2 = N / 2;
    vector<double> M1, M2;
    unsigned int seed;
    if (!fixed) {
        seed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now().time_since_epoch()).count();
        M1 = generate_vector_random(&seed, N, 1.0, A);
        M2 = generate_vector_random(&seed, N2, A, 10.0 * A);
    } else {
        seed = 123454321;
        M1 = generate_vector_fixed(&seed, N, 1.0, A);
        M2 = generate_vector_fixed(&seed, N2, A, 10.0 * A);
    }
    
    map_M1_sinh_square(M1);
    map_M2_tan_abs_with_prev(M2);

    merge_pow(M1, M2);

    if (use_sort) {
        selection_sort_partition_par(M2);

        // selection_sort_par_second_loop(M2);
        // selection_sort_seq(M2, 0, M2.size() - 1);
    }

    double X = compute_reduce_sum(M2, use_sort);

    return X;
}

int get_values_from_stdin(long long *N_input, double *A) {
    cout << "Insert N: ";
    if (!(cin >> *N_input)) {
        cerr << "Insert error. Need to insert N.\n";
        return 1;
    }

    cout << "Insert A: ";
    if (!(cin >> *A)) {
        cerr << "Insert error. Need to insert A.\n";
        return 1;
    }

    if (*N_input <= 0) {
        cerr << "Error, insert int N > 0.\n";
        return 1;
    }
    if (*A <= 0.0) {
        cerr << "Error, insert A > 0\n";
        return 1;
    }
    return 0;
}

struct TestCase {
    string name;
    bool fixed;
    bool use_sort;
    vector<int> elems;
};

vector<TestCase> tests = {
    // {"Fixed with sort",        true,  true, { 1000000 } },
    // {"Fixed without sort",        true,  false, { 1000000 } },
    // {"Random with sort",        false, true,     {10000, 200, 500, 1000, 5000, 10000, 50000, 200000, 500000, 1000000, 2000000} },
    // {"Random without sort",     false, false, {10000, 200, 500, 1000, 5000, 10000, 50000, 200000, 500000, 1000000, 20000000, 50000000} },
    {"Fixed with sort",         true,  true,  {10000, 200, 500, 1000, 5000, 10000, 50000, 200000, 500000, 1000000, 2000000} },
    {"Fixed without sort",      true,  false, {10000, 200, 500, 1000, 5000, 10000, 50000, 200000, 500000, 1000000, 20000000, 50000000} }
};

volatile double progress = 0.0;
volatile bool finished = false;

void progress_thread() {
    printf("Progress: %5.2f%%", 0.0);
    fflush(stdout);
    while (!finished) {
        printf("\rProgress: %5.2f%%   ", progress * 100.0);
        fflush(stdout);
        // #pragma omp sleep(1)
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    printf("\n");
}

#define NUM_OF_EXP 1.0
#define A_VALUE 42

// ---------- Main ----------
int main(int argc, char **argv) {
    ProgramParams params = parse_args(argc, argv);
    if (params.schedule != "") {
        apply_schedule(params);
    }
    check_schedule(params.schedule);
    omp_set_num_threads(params.thread_num);
    omp_set_nested(1);

    double X;
#ifdef USE_PROGRESS
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            progress_thread();
        }

        #pragma omp section
        {
        int total_steps = 0;

        for (const auto& t : tests) {
            total_steps += t.elems.size() * NUM_OF_EXP;
        }

        int done = 0;
#endif

        for (const auto& t : tests) {
            cout << t.name << ": \n";
            for (int j = 0; j < t.elems.size(); ++j) {
                double sum_ms = 0.0;
                for (int i = 0; i < NUM_OF_EXP; ++i) {
                    auto start = chrono::steady_clock::now();
                    X = start_algo(t.elems[j], A_VALUE, t.fixed, t.use_sort);
                    auto end = chrono::steady_clock::now();
                    
                    auto us = chrono::duration_cast<chrono::microseconds>(end - start);
                    double ms = us.count() / 1000.0;
                    // cout << ms << " ms \n";
                    sum_ms += ms;
#ifdef USE_PROGRESS
                    done++;
                    progress = (double)done / total_steps;
#endif
                }
                if (j != 0) {
                    #ifdef _OPENMP
                    append_to_csv(params.output_file, {
                        to_string(t.elems[j]),
                        params.schedule != "" ? params.schedule : "default",
                        params.schedule != "" ? to_string(params.chunk) : "default",
                        to_string(params.thread_num),
                        to_string(sum_ms / NUM_OF_EXP),
                        to_string(X)
                    });
                    #else
                    append_to_csv(params.output_file, {
                        to_string(t.elems[j]),
                        to_string(sum_ms / NUM_OF_EXP),
                        to_string(X)
                    });
                    #endif
                    cout << t.elems[j] << " elements; ";
                    cout << sum_ms / NUM_OF_EXP << " average ms; ";
                    cout << "thread_num= " << omp_get_max_threads() << ";";
                    cout << "X= " << X << ";\n";
                }
            }
        }

#ifdef USE_PROGRESS
        finished = true;   
    }
}
#endif


    return 0;
}
