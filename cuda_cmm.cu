#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#define PYBIND11

#ifdef PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif

#define ROMBERG_CONVERGENCE 1e-6f
#define ROMBERG_MAX_ITERATIONS 10
#define SEC_STEPS 50
#define SEC_TOL 1e-7f

#define FR_TOL 1e-6f
#define FR_STEPS 100

#define KAPPA_DIVISIONS 60000

#ifdef PYBIND11
#define PROGRESS_BAR if (index == blockDim.x * gridDim.x - 1) \
        { \
            const int num_chars = 20; \
            char a[num_chars + 1]; \
            a[num_chars] = '\0'; \
            for (int j = 0; j < num_chars; j++) \
            { \
                if (j * (n / num_chars) < i) \
                { \
                    a[j] = '#'; \
                } \
                else \
                { \
                    a[j] = ' '; \
                } \
            } \
            printf("Sample Thread current progress [%s]  %10d / %10d\r", a, i, n); \
        }
#else
#define PROGRESS_BAR //;
#endif

class CMM_Sheave {
    protected:
        bool converged, bisection;

        int debug;

        double kappa_array[KAPPA_DIVISIONS][3];
    public:
        float beta_naught, mu, alpha, v_theta_naught, A, delta, theta_c;
        __device__
        bool testSheaveExpansionCenter(float theta_c_naught);
        __device__
        void sheaveExpansionCenter();
        __device__
        float secNumerator(float theta);
        __device__
        float secDenominator(float theta);
        __device__
        void kappaFuncInit();
        __device__
        double rhs(float theta);
        __device__
        double kappaSlope(float theta);

        __device__
        float computeThetaC();

        // Manual Testing Constructor
        __device__
        CMM_Sheave(float beta_naught, float mu, float alpha, float v_theta_naught, float A, float delta, int debug, bool manual)
        : beta_naught{beta_naught}
        , mu{mu}
        , alpha{alpha}
        , v_theta_naught{v_theta_naught}
        , A{A}
        , delta{delta}
        , debug{debug}
        {
            this->converged = true;
            this->theta_c = 0.5752 * this->alpha;
            if (!manual)
            {
                this->sheaveExpansionCenter();
            }
        };

        __device__
        static void frBasedSheave(CMM_Sheave *sheave, float beta_naught, float mu, float alpha, float fr, float A, float delta, int debug);

        // Basic Constructor
        __device__
        CMM_Sheave(float beta_naught, float mu, float alpha, float v_theta_naught, float A, float delta, int debug)
        : CMM_Sheave(beta_naught, mu, alpha, v_theta_naught, A, delta, debug, false)
        {};

        // Non-Debug Constructor
        __device__
        CMM_Sheave(float beta_naught, float mu, float alpha, float v_theta_naught, float A, float delta)
        : CMM_Sheave(beta_naught, mu, alpha, v_theta_naught, A, delta, false)
        {};

        __device__
        CMM_Sheave()
        {};

        __device__
        float psi(float theta);
        __device__
        double kappa(float theta);
        __device__
        float press(float theta);
        __device__
        float vRadial(float theta);
        __device__
        float vTangent(float theta);
        __device__
        float beta(float theta);
        __device__
        float betaS(float theta);

        __device__
        float dimlessClamp();
        __device__
        float forceRatio();


        __device__
        float getThetaC() { return this->theta_c; };

        __device__
        static float kappaToTension(float kappa, float f_naught, float sigma, float omega, float R)
        {
            float inert = sigma * pow(omega, 2) * pow(R, 2);
            float tense = kappa * (f_naught - inert) + inert;
            return tense;
        }

        __device__
        static float sToAxialClamp(float s, float f_naught, float sigma, float omega, float R)
        {
            float inert = sigma * pow(omega, 2) * pow(R, 2);
            float axial = s * (f_naught - inert);
            return axial;
        }

        __device__
        static float axialClampToS(float axial, float f_naught, float sigma, float omega, float R)
        {
            float inert = sigma * pow(omega, 2) * pow(R, 2);
            float s = axial / (f_naught - inert);
            return s;
        }

        __device__
        static float cToTorque(float c, float f_naught, float sigma, float omega, float R)
        {
            float inert = sigma * pow(omega, 2) * pow(R, 2);
            float torque = c * R * (f_naught - inert);
            return torque;
        }

        __device__
        static float torqueToC(float torque, float f_naught, float sigma, float omega, float R)
        {
            float inert = sigma * pow(omega, 2) * pow(R, 2);
            float c = torque / (R * (f_naught - inert));
            return c;
        }

        __device__
        static float beltTensionsToForceRatio(float f_one, float f_two, float sigma, float omega, float R)
        {
            float inert = sigma * pow(omega, 2) * pow(R, 2);
            float force_ratio = (f_two - inert) / (f_one - inert);
            return force_ratio;
        }

        __device__
        static float dLdRp(float r_primary, float r_secondary, float d)
        {
            float del = M_PI + 2 * atan((r_primary - r_secondary) / d);
            del -= 2/d * (r_secondary - r_primary) / (1 + pow(((r_secondary - r_primary)/d), 2));
            del -= 2 * r_primary / sqrt(pow(d, 2) + pow((r_secondary - r_primary), 2));

            return del;
        }

        __device__
        static float dLdRs(float r_primary, float r_secondary, float d)
        {
            return CMM_Sheave::dLdRp(r_secondary, r_primary, d);
        }
};


__device__ float secNumeratorEval(float theta, CMM_Sheave * sheave)
{
    return sheave->press(theta) * sin(theta);
}

__device__ float secDenominatorEval(float theta, CMM_Sheave * sheave)
{
    return sheave->press(theta) * cos(theta);
}

__device__ float dimlessClampEval(float theta, CMM_Sheave * sheave)
{
    return (cos(sheave->beta(theta)) + sheave->mu * sin(sheave->betaS(theta))) * sheave->press(theta);
}

__device__ float trapSum(float (*f)(float, CMM_Sheave*), CMM_Sheave *sheave, float start, float end, int n) {
    float h = (end - start) / n;

    float sum = 0.5f * (f(start, sheave) + f(end, sheave));

    for (int i = 1; i < n; i++) {
        sum += f(start + i * h, sheave);
    }

    return sum * h;
}

__device__ float rombergSum(float (*f)(float, CMM_Sheave*), CMM_Sheave *sheave, float start, float end) {
    float convergence = 1e9;

    float romberg[ROMBERG_MAX_ITERATIONS - 1];
    float last_romberg[ROMBERG_MAX_ITERATIONS - 2];
    int n = 1;
    last_romberg[0] = trapSum(f, sheave, start, end, n);

    int iter = 2;


    while (convergence > ROMBERG_CONVERGENCE && iter < ROMBERG_MAX_ITERATIONS) {
        n *= 2;

        romberg[0] = trapSum(f, sheave, start, end, n);

        for (int k = 1; k < iter; k++) {
            romberg[k] = (pow(4, k) * romberg[k - 1] - last_romberg[k - 1]) / (pow(4, k) - 1);
        }

        convergence = abs((romberg[iter - 1] - last_romberg[iter - 2]) / romberg[iter - 1]);
        // printf("Romberg Iter: %d, Convergence: %f, Value: %f\n", iter, convergence, romberg[iter - 1]);
        iter++;

        for (int i = 0; i < iter - 1; i++) {
            last_romberg[i] = romberg[i];
        }
    }

    // printf("Romberg Iter: %d, Convergence: %f, Value: %f\r", iter, convergence, romberg[iter - 1]);
    return last_romberg[iter - 2];
}

// Takes in log(fr) and inits a sheave that is close enought to that force ratio
// Modified Secant Method
__device__
void CMM_Sheave::frBasedSheave(CMM_Sheave *sheave, float beta_naught, float mu, float alpha, float fr, float A, float delta, int debug)
{
    if (debug)
        printf("FR Target: %f\n", fr);

    sheave->beta_naught = beta_naught;
    sheave->mu = mu;
    sheave->alpha = alpha;
    sheave->A = A;
    sheave->delta = delta;
    sheave->debug = (debug > 2) * (debug - 2);

    float last_vel = 0.0, vel = 0.2, mid;

    sheave->v_theta_naught = vel;
    sheave->sheaveExpansionCenter();
    float current_fr = sheave->forceRatio();

    int iter = 0;
    last_vel = -5e3;
    vel = 5e3;

    if (debug > 1) {
        sheave->v_theta_naught = last_vel;
        sheave->sheaveExpansionCenter();

        printf("Lower Bound FR: %f\n", sheave->forceRatio());

        sheave->v_theta_naught = vel;
        sheave->sheaveExpansionCenter();

        printf("Upper Bound FR: %f\n", sheave->forceRatio());
     }

    while (iter < FR_STEPS && vel - last_vel > 1e-5)
    {
        mid = (last_vel + vel) / 2;
        sheave->v_theta_naught = mid;
        sheave->sheaveExpansionCenter();

        current_fr = sheave->forceRatio();

        if (debug > 1)
            printf("Mid Vel: %f, Current FR: %f\n", mid, current_fr);

        if (current_fr < fr)
            last_vel = mid;
        else
            vel = mid;

        if (debug > 1)
            printf("FR Iter: %d Vel Bounds: [%f, %f]\n", iter, last_vel, vel);
        iter++;
    }

    if (debug)
        printf("FR Iterations: %d, Target FR: %f, Final FR: %f, Vel: %f\n", iter, fr, current_fr, vel);
}

__device__
float CMM_Sheave::computeThetaC()
{
    float y = rombergSum(secNumeratorEval, this, 0, this->alpha);
    float x = rombergSum(secDenominatorEval, this, 0, this->alpha);

    return atan2(y, x);
}

__device__
void CMM_Sheave::sheaveExpansionCenter()
{
    int iter = 0;

    double lower_tc, upper_tc;

    this->kappaFuncInit();
    float theta_c_prime = this->computeThetaC();

    if (theta_c_prime < this->theta_c) {
        lower_tc = 0.0;
        upper_tc = this->theta_c;
    } else {
        lower_tc = this->theta_c;
        upper_tc = this->alpha;
    }


    while (upper_tc - lower_tc > SEC_TOL && iter < SEC_STEPS)
    {
        this->theta_c = (upper_tc + lower_tc) / 2;

        this->kappaFuncInit();

        theta_c_prime = this->computeThetaC();

        if (theta_c_prime > this->theta_c)
            lower_tc = this->theta_c;
        else
            upper_tc = this->theta_c;

        iter++;

        if (this->debug > 1)
        {
            printf("Iteration %d: (%f, %f)\n", iter, lower_tc, upper_tc);
        }
    }

    // printf("SEC Iterations: %d\n", iter);
    // this->converged = iter < SEC_STEPS;
    this->converged = true;
}

__device__
void CMM_Sheave::kappaFuncInit()
{
    // Kappa is euler method nearest neighbor (linear interp between computed points)
    // this->kappa_array[ind] = [theta, rhs, rhs_slope]

    double rhs_current = 0.0;
    double rhs_delta = 0.0;

    double start_slope = 0.0;
    double end_slope = 0.0;
    double avg_slope = 0.0;

    double t = 0.0;

    // Resolution of Kappa in the Theta domain
    double kappa_res = this->alpha / (KAPPA_DIVISIONS - 1.0);

    for (int ind = 0; ind < KAPPA_DIVISIONS; ind++)
    {
        t = ind * kappa_res;

        start_slope = this->kappaSlope(t);

        // Predictor Step
        rhs_delta = kappa_res * start_slope;
        end_slope = this->kappaSlope(t + kappa_res);

        // Corrector Step
        avg_slope = 0.5 * (start_slope + end_slope);
        rhs_delta = avg_slope * kappa_res;

        this->kappa_array[ind][0] = t;
        this->kappa_array[ind][1] = rhs_current;
        this->kappa_array[ind][2] = avg_slope;

        rhs_current += rhs_delta;
    }
}

__device__
double CMM_Sheave::kappaSlope(float theta)
{
    float vRad = this->vRadial(theta);
    float vTan = this->vTangent(theta);
    double vMag = sqrt(vRad * vRad + vTan * vTan);

    double cos_p = vRad / vMag;
    return this->mu * (vTan / vMag) / (sin(this->beta_naught) * sqrt(1 + pow(tan(this->beta_naught)*cos_p, 2)) - this->mu * cos_p);
}

__device__
double CMM_Sheave::rhs(float theta)
{
    int ind = (int) (theta / (this->alpha / (KAPPA_DIVISIONS - 1)));

    double* k = this->kappa_array[ind];

    // std::cout << "RHS DEBUG: " << k[1] << " " << k[2] << " " << theta << " " << k[0] << std::endl;

    return k[1] + k[2] * (theta - k[0]);
}

__device__
double CMM_Sheave::kappa(float theta)
{
    // if (theta - this->alpha > 0.1)
    // {
    //     std::cout << "Error: theta > alpha (" << theta << ", " << this->alpha << ")" << std::endl;
    // }
    // else if (theta < 0.0)
    // {
    //     std::cout << "Error: theta < 0 (" << theta << ")" << std::endl;
    // }

    int ind = (int) (theta / (this->alpha / (KAPPA_DIVISIONS - 1)));

    double* k = this->kappa_array[ind];

    double rhs_theta = k[1] + k[2] * (theta - k[0]);

    return exp(rhs_theta);
}

__device__
float CMM_Sheave::psi(float theta)
{
    float vRad = this->v_theta_naught - this->A*theta - 2*sin(theta/2)*sin(theta/2 - this->theta_c);
    float vTan = this->A - cos(theta - this->theta_c + 0.5*M_PI);

    return atan2(vRad, vTan);
}

// EQ. 37
__device__
float CMM_Sheave::press(float theta)
{
    float vRad = this->vRadial(theta);
    float vTan = this->vTangent(theta);

    float cos_p = vRad / sqrt(vRad * vRad + vTan * vTan);

    float tan_2_beta = pow(tan(this->beta_naught), 2);

    return sqrt(1 + tan_2_beta * pow(cos_p, 2)) / (sin(this->beta_naught) * sqrt(1 + tan_2_beta * pow(cos_p, 2)) - this->mu * cos_p) * this->kappa(theta) * 0.5;
}

// EQ. 35
__device__
float CMM_Sheave::vRadial(float theta)
{
    return this->A - cos(theta - this->theta_c + 0.5*M_PI);
}

// EQ. 36
__device__
float CMM_Sheave::vTangent(float theta)
{
    return this->v_theta_naught - this->A*theta - 2*sin(theta/2)*sin(theta/2 - this->theta_c);
}

// EQ. 17
__device__
float CMM_Sheave::beta(float theta)
{
    return this->beta_naught + 0.5 * this->delta * sin(theta - this->theta_c + 0.5 * M_PI);
}

// EQ. 4
__device__
float CMM_Sheave::betaS(float theta)
{
    float vRad = this->vRadial(theta);
    float vTan = this->vTangent(theta);

    float cos_p = vRad / sqrt(vRad * vRad + vTan * vTan);

    return atan(tan(this->beta(theta)) * cos_p);
}

// EQ. 50
__device__
float CMM_Sheave::dimlessClamp()
{
    return rombergSum(dimlessClampEval, this, 0.0, this->alpha);
}

__device__
float CMM_Sheave::forceRatio()
{
    if (this->converged)
    {
        return this->rhs(this->alpha);
    }
    else
    {
        return nanf("e");
    }
}

__device__
float computeL(float primary, float tau, float cToC)
{
    float secondary = primary / tau;
    float theta_t = asinf((secondary - primary) / cToC);

    float alpha_primary = M_PI - 2 * theta_t;
    float alpha_secondary = M_PI + 2 * theta_t;
    return primary * alpha_primary + secondary * alpha_secondary + 2 * sqrtf(cToC*cToC - (secondary - primary) * (secondary - primary));
}

__device__
float lSlope(float primary, float tau, float cToC)
{
    float secondary = primary / tau;
    float theta_t = asinf((secondary - primary) / cToC);

    float alpha_primary = M_PI - 2 * theta_t;
    float alpha_secondary = M_PI + 2 * theta_t;

    float dratio_dp = (1/tau - 1) / cToC;

    float dtt_dp = 1.0f / sqrtf(1 + powf(((primary * (1/tau - 1)) / cToC), 2)) * dratio_dp;

    float dap_dp = -2.0f * dtt_dp;
    float das_dp = 2.0f * dtt_dp;

    float dsl_dp = 0.5f / sqrtf(cToC*cToC - powf((secondary - primary), 2)) * 2.0f * primary * (1/tau - 1) * (1/tau - 1);

    return alpha_primary + primary * dap_dp + 1.0f / tau * alpha_secondary + secondary * das_dp + 2 * dsl_dp;
}

__device__
float tauToPrimary(float tau, float cToC, float L)
{
    float guess = (L - 2 * cToC) / (1 + 1/tau) / M_PI, length;

    int iter = 0;
    float discrep = 1e9;

    while (fabs(discrep) > 1e-6 && iter < 50)
    {
        length = computeL(guess, tau, cToC);
        discrep = length - L;

        guess -= discrep / lSlope(guess, tau, cToC);
        iter++;
    }

    return guess;
}

__device__
float dLdRp(float r_primary, float r_secondary, float d)
{
    float del = M_PI + 2 * atan((r_primary - r_secondary) / d);
    del -= 2/d * (r_secondary - r_primary) / (1 + pow(((r_secondary - r_primary)/d), 2));
    del -= 2 * r_primary / sqrt(pow(d, 2) + pow((r_secondary - r_primary), 2));

    return del;
}

__device__
float dLdRs(float r_primary, float r_secondary, float d)
{
    return dLdRp(r_secondary, r_primary, d);
}

__global__
void fr(int n, float *vec, float beta_naught, float mu)
{
    const int cols = 4;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        PROGRESS_BAR

        float v_theta_naught = vec[cols * i + 0];
        float A = vec[cols * i + 1];
        float alpha = vec[cols * i + 2];

        CMM_Sheave sheave(beta_naught, mu, alpha, v_theta_naught, A, 0.0f, false, false);

        vec[cols * i + 3] = sheave.forceRatio();
    }
}

__global__
void equilibriumClamp(int n, float *vec, float beta_naught, float mu, float delta)
{
    const int cols = 3;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        PROGRESS_BAR

        float fr = vec[cols * i + 0];
        float alpha = vec[cols * i + 1];

        CMM_Sheave sheave;
        CMM_Sheave::frBasedSheave(&sheave, beta_naught, mu, alpha, fr, 0.0, delta, 0);

        // if (fabsf(sheave.forceRatio() - fr) > 0.2f) {
        //     printf("Fr In: %f alpha: %f Fr Out:%f\n", fr, alpha, sheave.forceRatio());
        // }

        vec[cols * i + 0] = sheave.forceRatio();
        vec[cols * i + 2] = sheave.dimlessClamp();
    }
}

__global__
void cCoefficientCompute(int n, float *vec, float beta_naught, float mu, float delta, float cToC, float L, float sigma)
{
    const int cols = 6;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        PROGRESS_BAR

        float fr = vec[cols * i + 0];
        float A = vec[cols * i + 1];
        float tau = vec[cols * i + 2];

        // printf("\nfr %f A %f tau %f omega %f\n", fr, A, tau, omega);

        float prim_rad = tauToPrimary(tau, cToC, L);
        // printf("Primary Radius: %f\n", prim_rad);

        float theta_t = asinf(((1 / tau - 1) * prim_rad) / cToC);

        float alpha_primary = M_PI - 2 * theta_t;
        float alpha_secondary = M_PI + 2 * theta_t;
        // printf("Alpha Primary: %f Alpha Secondary: %f\n", alpha_primary, alpha_secondary);

        // float taught_side_force =

        // printf("Primary FR: %f Secondary FR: %f\n", fr, -fr);
        float secondary_A = -A * dLdRp(prim_rad, prim_rad / tau, cToC) / dLdRs(prim_rad, prim_rad / -fr, cToC);
        // printf("Primary A: %f Secondary A: %f\n", A, secondary_A);

        CMM_Sheave sheave;
        CMM_Sheave::frBasedSheave(&sheave, beta_naught, mu, alpha_primary, fr, 0.0, delta, 0);
        float prim_eq = sheave.dimlessClamp();

        fr = sheave.forceRatio();

        CMM_Sheave::frBasedSheave(&sheave, beta_naught, mu, alpha_secondary, -fr, 0.0, delta, 0);
        float sec_eq = sheave.dimlessClamp();

        CMM_Sheave::frBasedSheave(&sheave, beta_naught, mu, alpha_primary, fr, A, delta, 0);
        float prim_clamp = sheave.dimlessClamp();

        float prim_v_theta = sheave.vTangent(0.0);
        float prim_v_r = sheave.vRadial(0.0);

        float D = sinf(2 * beta_naught) / (delta * (1 + powf(cosf(beta_naught), 2)));


        CMM_Sheave::frBasedSheave(&sheave, beta_naught, mu, alpha_secondary, -fr, secondary_A, delta, 0);
        float sec_clamp = sheave.dimlessClamp();
        float sec_v_theta = sheave.vTangent(alpha_secondary);
        float sec_v_r = sheave.vRadial(alpha_secondary);

        float tau_effective =  tau * (1 - prim_v_theta / (prim_v_theta - D)) / (1 - sec_v_theta / (sec_v_theta - D)) * sqrtf((1 + prim_v_r*prim_v_r / D / D) / (1 + sec_v_r*sec_v_r / D / D));

        vec[cols * i + 0] = fr;

        vec[cols * i + 3] = logf(prim_clamp / sec_clamp * sec_eq / prim_eq);
        vec[cols * i + 4] = tau_effective;

        vec[cols * i + 5] = prim_clamp;
    }
}

#ifdef PYBIND11
// Take in a n x 4 matrix and write the force ratio to the final column
void pyFR(pybind11::array_t<float> vec, float beta_naught, float mu)
{
    const int cols = 4;

    pybind11::buffer_info ha = vec.request();

    if (ha.ndim != 2 && ha.shape[1] != cols) {
        std::stringstream strstr;
        strstr << "Numpy Array not n x " << cols << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int n = ha.shape[0];
    size_t size_bytes = n*cols*sizeof(float);
    float *gpu_ptr;
    cudaError_t error;

    error = cudaMalloc(&gpu_ptr, size_bytes);

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    float* ptr = reinterpret_cast<float*>(ha.ptr);
    error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    int deviceID;
    cudaDeviceProp props;

    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);

    int num_blocks = props.multiProcessorCount;
    int num_threads = 256;

    fr<<<num_blocks, num_threads>>>(n, gpu_ptr, beta_naught, mu);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream strstr;
        strstr << "run_kernel launch failed" << std::endl;
        strstr << cudaGetErrorString(error);
        throw strstr.str();
    }

    error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    error = cudaFree(gpu_ptr);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

// Take in a n x 3 matrix and write the dimensionless clamp to the final column
void pyEquilibrium(pybind11::array_t<float> vec, float beta_naught, float mu, float delta)
{
    const int cols = 3;
    pybind11::buffer_info ha = vec.request();

    if (ha.ndim != 2 && ha.shape[1] != cols) {
        std::stringstream strstr;
        strstr << "Numpy Array not n x " << cols << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int n = ha.shape[0];
    size_t size_bytes = n*cols*sizeof(float);
    float *gpu_ptr;
    cudaError_t error;

    error = cudaMalloc(&gpu_ptr, size_bytes);

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    float* ptr = reinterpret_cast<float*>(ha.ptr);
    error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    int deviceID;
    cudaDeviceProp props;

    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);

    int num_blocks = props.multiProcessorCount;
    int num_threads = 256;

    equilibriumClamp<<<num_blocks, num_threads>>>(n, gpu_ptr, beta_naught, mu, delta);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream strstr;
        strstr << "run_kernel launch failed" << std::endl;
        strstr << cudaGetErrorString(error);
        throw strstr.str();
    }

    error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    error = cudaFree(gpu_ptr);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

// Take in a n x 6 matrix and write the dimensionless clamp ratio delta and the tau effective to the final columns
void pyCCoefficient(pybind11::array_t<float> vec, float beta_naught, float mu, float delta, float cToC, float L, float sigma)
{
    const int cols = 6;

    pybind11::buffer_info ha = vec.request();

    if (ha.ndim != 2 && ha.shape[1] != cols) {
        std::stringstream strstr;
        strstr << "Numpy Array not n x " << cols << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int n = ha.shape[0];
    size_t size_bytes = n*cols*sizeof(float);
    float *gpu_ptr;
    cudaError_t error;

    error = cudaMalloc(&gpu_ptr, size_bytes);

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    float* ptr = reinterpret_cast<float*>(ha.ptr);
    error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    int deviceID;
    cudaDeviceProp props;

    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);

    int num_blocks = props.multiProcessorCount;
    int num_threads = 256;

    cCoefficientCompute<<<num_blocks, num_threads>>>(n, gpu_ptr, beta_naught, mu, delta, cToC, L, sigma);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream strstr;
        strstr << "run_kernel launch failed" << std::endl;
        strstr << cudaGetErrorString(error);
        throw strstr.str();
    }

    error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    error = cudaFree(gpu_ptr);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

PYBIND11_MODULE(cuda_cmm, m)
{
    m.def("fr", &pyFR);
    m.def("eq_clamp", &pyEquilibrium);
    m.def("c_coefficient", &pyCCoefficient);
}
#endif

#ifndef PYBIND11
int main() {
    const size_t n = 8*64;
    float a[n] = {-3.600783, -0.935484, 3.695991, -69.420, -3.491194, -0.935484, 3.563651, -69.420};

    float *gpu_ptr;
    cudaMalloc(&gpu_ptr, n*sizeof(float));

    cudaMemcpy(gpu_ptr, a, n*sizeof(float), cudaMemcpyHostToDevice);

    fr<<<2, 1>>>(n/4, gpu_ptr, M_PI / 180.0 * 23.0 / 2.0, 0.1);

    cudaDeviceSynchronize();
    cudaMemcpy(a, gpu_ptr, n*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << std::endl;
    std::cout << a[4] << " " << a[5] << " " << a[6] << " " << a[7] << std::endl;

    cudaFree(gpu_ptr);


    /* float b[3] = {-1.4, 2.8274333, 0.0};
    cudaMalloc(&gpu_ptr, 3*sizeof(float));

    cudaMemcpy(gpu_ptr, b, 3*sizeof(float), cudaMemcpyHostToDevice);

    equilibriumClamp<<<1, 1>>>(1, gpu_ptr, M_PI / 180.0 * 23.0 / 2.0, 0.1, 1e-4);

    cudaDeviceSynchronize();
    cudaMemcpy(b, gpu_ptr, 3*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << b[0] << " " << b[1] << " " << b[2] << std::endl;

    cudaFree(gpu_ptr);

    const int ps = 12;
    float c[ps] = {-1.4, 0.5, 0.8, 0.0, -1.0, 0.5, 0.8, 0.0, -0.4, 0.5, 0.8, 0.0};
    cudaMalloc(&gpu_ptr, ps*sizeof(float));

    cudaMemcpy(gpu_ptr, c, ps*sizeof(float), cudaMemcpyHostToDevice);

    cCoefficientCompute<<<1, 1>>>(3, gpu_ptr, M_PI / 180.0 * 23.0 / 2.0, 0.1, 1e-4, 0.3, 2.0, 0.5);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
    cudaMemcpy(c, gpu_ptr, ps*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << " " << c[4] << std::endl;


    cudaFree(gpu_ptr); */

    return 0;
}
#endif
