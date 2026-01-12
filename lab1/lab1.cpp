#include <cmath>
#include <cfloat>
#include <climits>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <chrono>
using namespace std;

// ---------- Generate ----------
double rand_uniform(unsigned int *seedp, double low, double high) {
    unsigned int r = rand_r(seedp);
    double unit = (double)r / (double)RAND_MAX;
    return low + unit * (high - low);
}

vector<double> generate_vector(unsigned int *seedp, size_t size, double low, double high) {
    vector<double> v;
    v.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        v[i] = rand_uniform(seedp, low, high);
    }
    return v;
}

// ---------- Map ----------
void map_M1_sinh_square(vector<double> &M1) {

    for (size_t i = 0; i < M1.size(); ++i) {
        double s = sinh(M1[i]);
        M1[i] = s * s;
    }
}

void map_M2_tan_abs_with_prev(vector<double> &M2) {
    if (M2.empty()) return;
    vector<double> copy = M2;
    double prev = 0.0;
    for (size_t i = 0; i < M2.size(); ++i) {
        double val = copy[i] + prev;
        double t = tan(val);
        M2[i] = fabs(t);
        prev = copy[i];
    }
}

// ---------- Merge ----------
void merge_pow(vector<double> &M1, vector<double> &M2) {
    size_t limit = min(M1.size(), M2.size());
    for (size_t i = 0; i < limit; ++i) {
        M2[i] = pow(M1[i], M2[i]);
    }
}

// ---------- Sort (Selection sort) ----------
void selection_sort(vector<double> &arr) {
    size_t n = arr.size();
    for (size_t i = 0; i < n - 1; ++i) {
        size_t min_idx = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (arr[j] < arr[min_idx]) min_idx = j;
        }
        swap(arr[i], arr[min_idx]);
    }
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

double start_algo(long long N_input, double A, bool fixed, bool use_sort) {
    size_t N = static_cast<size_t>(N_input);
    size_t N2 = N / 2;
    unsigned int seed = 123454321;
    if (!fixed) {
        seed = static_cast<unsigned int>(time(nullptr));
    }

    // Stage Generate
    vector<double> M1 = generate_vector(&seed, N, 1.0, A);
    // print_arr(M1);
    vector<double> M2 = generate_vector(&seed, N2, A, 10.0 * A);
    // print_arr(M2);
    
    // Stage Map
    map_M1_sinh_square(M1);
    // print_arr(M1);
    map_M2_tan_abs_with_prev(M2);
    // print_arr(M2);

    // Stage Merge
    merge_pow(M1, M2);
    // print_arr(M2);

    // Stage Sort (selection sort)
    if (use_sort) {
        selection_sort(M2);
    }
    // print_arr(M2);

    // Stage Reduce
    double X = compute_reduce_sum(M2, use_sort);

    return X;
}

struct TestCase {
    std::string name;
    bool fixed;
    bool use_sort;
};

std::vector<TestCase> tests = {
    {"Random with sort",        false, true},
    {"Random without sort",     false, false},
    {"Fixed with sort",         true,  true},
    {"Fixed without sort",      true,  false}
};

// ---------- Main ----------
int main() {
    // ios::sync_with_stdio(false);
    // cin.tie(nullptr);

    // cout << fixed << setprecision(12);

    long long N_input;
    double A;
    // Read N and A from console
    cout << "Insert N: ";
    if (!(cin >> N_input)) {
        cerr << "Insert error. Need to insert N.\n";
        return 1;
    }

    cout << "Insert A: ";
    if (!(cin >> A)) {
        cerr << "Insert error. Need to insert A.\n";
        return 1;
    }

    if (N_input <= 0) {
        cerr << "Error, insert int N > 0.\n";
        return 1;
    }
    if (A <= 0.0) {
        cerr << "Error, insert A > 0\n";
        return 1;
    }
    double X;

    for (const auto& t : tests) {
        cout << t.name << " ";
        
        auto start = chrono::steady_clock::now();
        X = start_algo(N_input, A, t.fixed, t.use_sort);
        auto end = chrono::steady_clock::now();
        
        auto ms = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << ms.count() << " ms ";
        cout << X << "\n";
    }   

    return 0;
}
