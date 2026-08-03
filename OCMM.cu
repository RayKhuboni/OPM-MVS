#include "OCMM.h"
#include <cmath>
#include <math.h>
#include <queue>
#include <set>
#include <curand_kernel.h>

#define mul4(v,k) { \
    v->x = v->x * k; \
    v->y = v->y * k; \
    v->z = v->z * k; \
	v->w = v->w * k; \
}

#define vecdiv4(v,k) { \
    v->x = v->x / k; \
    v->y = v->y / k; \
    v->z = v->z / k; \
	v->w = v->w / k; \
}
#define nvecdiv4(v,k) { \
    v.x = v.x / k; \
    v.y = v.y / k; \
    v.z = v.z / k; \
	v.w = v.w / k; \
}

#define vecmul4(v,u) { \
    v->x = v->x * u->x; \
    v->y = v->y * u->y; \
    v->z = v->z * u->y; \
	v->w = v->w * u->w; \
}

struct PropagationPoint {
	int point;
	float cost;

	// Comparator for the priority queue, ensuring the lowest cost comes first
	bool operator>(const PropagationPoint &other) const {
		return cost > other.cost;  // Min-heap based on cost
	}
};

// Device function to get the 3D point from the camera using the depth
__device__ float3 Get3DPointFromCamera(const Camera &camera, int point_idx, float depth) {
	// Extract pixel coordinates from point index
	int x = point_idx % camera.width;  // x-coordinate in the image
	int y = point_idx / camera.width;  // y-coordinate in the image

	// Camera intrinsics: fx, fy, cx, cy
	float fx = camera.K[0];  // Focal length in x
	float fy = camera.K[4];  // Focal length in y
	float cx = camera.K[2];  // Principal point x-coordinate
	float cy = camera.K[5];  // Principal point y-coordinate

	// Convert the 2D pixel coordinates to normalized image coordinates (camera space)
	float normalized_x = (x - cx) / fx;
	float normalized_y = (y - cy) / fy;

	// Compute the 3D coordinates by back-projecting using the depth
	float3 point3D;
	point3D.x = normalized_x * depth;
	point3D.y = normalized_y * depth;
	point3D.z = depth;

	return point3D;
}
__device__ inline float dotProduct(const float4 a, const float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float ComputePointToPlaneDistance(const Camera &camera, int point_idx, const float4 &plane_hypothesis,
	curandState *rand_state, float depth_min, float depth_max) {
	// Extract plane coefficients (A, B, C, D) from the plane hypothesis
	float A = plane_hypothesis.x;
	float B = plane_hypothesis.y;
	float C = plane_hypothesis.z;
	float D = plane_hypothesis.w;

	// Step 1: Generate a random depth for the current pixel (if needed, can be from depth map)
	// Here, depth is randomly chosen between depth_min and depth_max (if applicable).
	float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;

	if (depth <= 0) {
		return FLT_MAX; // Invalid depth
	}

	// Step 2: Compute the 3D coordinates from 2D pixel coordinates and depth using the camera model
	// Assuming Get3DPointFromCamera function can handle the 2D to 3D transformation.
	float3 point3D = Get3DPointFromCamera(camera, point_idx, depth);

	// Step 3: Compute the point-to-plane distance using the formula for signed distance from point to plane
	// point3D is the 3D point (x, y, z), and (A, B, C, D) are the plane coefficients
	float numerator = fabs(A * point3D.x + B * point3D.y + C * point3D.z + D);
	float denominator = sqrtf(A * A + B * B + C * C);

	if (denominator < 1e-6f) {
		return FLT_MAX; // Avoid division by zero (if the plane normal is near zero)
	}

	// Calculate the point-to-plane distance
	float distance = numerator / denominator;
	return distance;
}

__device__ float ComputePointToPlaneDistancewithDepth(const Camera &camera, int point_idx, const float4 &plane_hypothesis,
	float curdepth) {
	// Extract plane coefficients (A, B, C, D) from the plane hypothesis
	float A = plane_hypothesis.x;
	float B = plane_hypothesis.y;
	float C = plane_hypothesis.z;
	float D = plane_hypothesis.w;

	float depth = curdepth;
	// Step 2: Compute the 3D coordinates from 2D pixel coordinates and depth using the camera model
	// Assuming Get3DPointFromCamera function can handle the 2D to 3D transformation.
	float3 point3D = Get3DPointFromCamera(camera, point_idx, depth);

	// Step 3: Compute the point-to-plane distance using the formula for signed distance from point to plane
	// point3D is the 3D point (x, y, z), and (A, B, C, D) are the plane coefficients
	float numerator = fabs(A * point3D.x + B * point3D.y + C * point3D.z + D);
	float denominator = sqrtf(A * A + B * B + C * C);

	if (denominator < 1e-6f) {
		return FLT_MAX; // Avoid division by zero (if the plane normal is near zero)
	}

	// Calculate the point-to-plane distance
	float distance = numerator / denominator;
	return distance;
}


__device__ float4 plane4div(float4 plane, int divisible_const) {
	if (divisible_const == 0) {
		// Handle division by zero
		return plane; // Or use a specific fallback value
	}
	float divisor = static_cast<float>(divisible_const);
	return make_float4(plane.x / divisor, plane.y / divisor, plane.z / divisor, plane.w / divisor);
}

__device__ int Point2Idx(const int2& p, int width)
{
	return p.y * width + p.x;
}

__device__ void sort_small(float *cost_vector, int *cost_position_vector, const int count) {
	for (int i = 1; i < count; i++) {
		float cost_tmp = cost_vector[i];            // Temporary value for cost
		int position_tmp = cost_position_vector[i]; // Temporary value for position
		int j = i;

		// Insertion sort logic: Shift larger elements to the right
		for (; j >= 1 && cost_tmp < cost_vector[j - 1]; j--) {
			cost_vector[j] = cost_vector[j - 1];
			cost_position_vector[j] = cost_position_vector[j - 1];
		}

		// Place the current element in the correct position
		cost_vector[j] = cost_tmp;
		cost_position_vector[j] = position_tmp;
	}
}

__device__  void sort_small(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++) 
	{
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}

__device__ void sort_small_weighted(float *d, float *w, int n)
{
    int j;
    for (int i = 1; i < n; i++) 
	{
        float tmp = d[i];
        float tmp_w = w[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--) 
		{
            d[j] = d[j - 1];
            w[j] = w[j - 1];
        }
        d[j] = tmp;
        w[j] = tmp_w;
    }
}

__device__ int FindMinCostIndex(const float *costs, const int n)
{
    float min_cost = costs[0];
    int min_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] <= min_cost) {
            min_cost = costs[idx];
            min_cost_idx = idx;
        }
    }
    return min_cost_idx;
}

__device__ void FindMinCostIndexes(const float *costs, const int n, int *min_cost_indexes)
{
	// Initialize min_cost_indexes to store the indices of the three minimum costs
	min_cost_indexes[0] = 0;
	min_cost_indexes[1] = 1;
	min_cost_indexes[2] = 2;

	// Initialize min_costs to store the three minimum costs
	float min_costs[3] = { costs[0], costs[1], costs[2] };

	// Iterate through the array of costs to find the three minimum costs and their indices
	for (int idx = 3; idx < n; ++idx) {
		// Compare the current cost with the three minimum costs
		if (costs[idx] < min_costs[0]) {
			min_costs[2] = min_costs[1];
			min_costs[1] = min_costs[0];
			min_costs[0] = costs[idx];
			min_cost_indexes[2] = min_cost_indexes[1];
			min_cost_indexes[1] = min_cost_indexes[0];
			min_cost_indexes[0] = idx;
		}
		else if (costs[idx] < min_costs[1]) {
			min_costs[2] = min_costs[1];
			min_costs[1] = costs[idx];
			min_cost_indexes[2] = min_cost_indexes[1];
			min_cost_indexes[1] = idx;
		}
		else if (costs[idx] < min_costs[2]) {
			min_costs[2] = costs[idx];
			min_cost_indexes[2] = idx;
		}
	}
}

__device__ int binarySearchCDF(const float* cdf, int size, float value) {
	int low = 0, high = size - 1;
	while (low < high) {
		int mid = (low + high) / 2;
		if (cdf[mid] < value)
			low = mid + 1;
		else
			high = mid;
	}
	return low;
}
// Make the CDF robust: sanitize, normalize, clamp tail
__device__ inline void TransformPDFToCDF_Safe(float* probs, int n) {
	if (n <= 0) return;

	float sum = 0.0f;
	// sanitize negatives/NaNs to 0
	for (int i = 0; i < n; ++i) {
		float p = probs[i];
		if (!(p > 0.0f)) p = 0.0f; // also clears NaNs
		probs[i] = p;
		sum += p;
	}

	if (sum == 0.0f) {
		// fallback: uniform CDF
		const float inv = 1.0f / n;
		float cum = 0.0f;
		for (int i = 0; i < n; ++i) { cum += inv; probs[i] = cum; }
		probs[n - 1] = 1.0f; // clamp
		return;
	}

	const float inv_sum = 1.0f / sum;
	float cum = 0.0f;
	for (int i = 0; i < n; ++i) {
		cum += probs[i] * inv_sum;
		probs[i] = cum;
	}
	probs[n - 1] = 1.0f; // clamp tail exactly
}

__device__ inline int binarySearchCDF_UB(const float* cdf, int n, float value) {
	// returns first i with cdf[i] >= value; clamps into [0, n-1]
	if (n <= 0) return -1;
	int low = 0, high = n - 1, ans = n - 1;
	while (low <= high) {
		int mid = (low + high) >> 1;
		// Note: comparisons with NaN are false; cdf must be sanitized
		if (cdf[mid] >= value) { ans = mid; high = mid - 1; }
		else { low = mid + 1; }
	}
	return ans; // always in [0, n-1] if cdf[n-1] >= value
}

__device__  void setBit(unsigned int &input, const unsigned int n)
{
    input |= (unsigned int)(1 << n);
}

__device__  int isSet(unsigned int input, const unsigned int n)
{
    return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result)
{
  result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
  result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
  result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ void NormalizeVec3 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf (normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}
__device__ void NormalizeVec4(float4 *vec)
{
	const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z + vec->w * vec->w;
	const float inverse_sqrt = rsqrtf(normSquared);
	vec->x *= inverse_sqrt;
	vec->y *= inverse_sqrt;
	vec->z *= inverse_sqrt;
	vec->w *= inverse_sqrt;
}
// Compute Plane Distance
__device__ float ComputePlaneDistance(float4 *planeA, float4 *planeB)
{
	return sqrt((planeA->x - planeB->x) * (planeA->x - planeB->x) + (planeA->y - planeB->y) * (planeA->y - planeB->y) + (planeA->z - planeB->z) * (planeA->z - planeB->z));
}

__forceinline__ __device__ int argmin3(float a, float b, float c) {
	return (a <= b && a <= c) ? 0 : (b <= a && b <= c) ? 1 : 2;
}

__forceinline__ __device__ int argmin4(float a, float b, float c, float d) {
	return (a <= b && a <= c && a <= d) ? 0 : (b <= a && b <= c && b <= d) ? 1 : (c <= a && c <= b && c <= d) ? 2 : 3;
}

// Orientation-insensitive angular distance between unit normals
__forceinline__ __device__ float angDist(float3 n1, float3 n2) {
	// ensure unit-length n1,n2 before calling
	float c = fabsf(n1.x*n2.x + n1.y*n2.y + n1.z*n2.z);
	c = fminf(1.f, fmaxf(-1.f, c));
	return acosf(c); // radians
}

// Depth distance (use disparity for stability)
__forceinline__ __device__ float depthDist(float d1, float d2) {
	// guard tiny depths if needed
	float u1 = 1.0f / d1;
	float u2 = 1.0f / d2;
	return fabsf(u1 - u2);
}

// Combined plane distance
__forceinline__ __device__ float planeDist(const float4& A, const float4& B,
	float alpha, float beta) {
	// A=(nx,ny,nz,d), normals must be unit and consistently oriented beforehand
	float3 nA = make_float3(A.x, A.y, A.z);
	float3 nB = make_float3(B.x, B.y, B.z);
	return alpha * angDist(nA, nB) + beta * depthDist(A.w, B.w);
}

// Ensure each candidate normal is unit and oriented (e.g., n.z>0 or n·viewdir>0)
__forceinline__ __device__ float4 fixPlane(float4 P) {
	// normalize normal
	float len = rsqrtf(P.x*P.x + P.y*P.y + P.z*P.z);
	float4 Q = make_float4(P.x*len, P.y*len, P.z*len, P.w);
	// enforce consistent orientation (example: n.z>0)
	if (Q.z < 0.f) { Q.x = -Q.x; Q.y = -Q.y; Q.z = -Q.z; Q.w = -Q.w; }
	return Q;
}

// Compute Test if Equal
__device__ bool float4Equal(float4 a, float4 b) {
	return (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
}
__device__ void TransformPDF(float* probs, const int num_probs)
{
	float prob_sum = 0.0f;
	for (int i = 0; i < num_probs; ++i) {
		prob_sum += probs[i];
	}
	const float inv_prob_sum = 1.0f / prob_sum;

	float cum_prob = 0.0f;
	for (int i = 0; i < num_probs; ++i) {
		const float prob = probs[i] * inv_prob_sum;
		probs[i] = prob;
	}
}
__device__ void TransformPDFToCDF(float* probs, const int num_probs)
{
    float prob_sum = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        prob_sum += probs[i];
    }
    const float inv_prob_sum = 1.0f / prob_sum;

    float cum_prob = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        const float prob = probs[i] * inv_prob_sum;
        cum_prob += prob;
        probs[i] = cum_prob;
    }
}

__device__ void Get3DPoint(const Camera camera, const int2 p, const float depth, float *X)
{
    X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    X[2] = depth;
}

__device__ float4 GetViewDirection(const Camera camera, const int2 p, const float depth)
{
    float X[3];
    Get3DPoint(camera, p, depth, X);
    float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

    float4 view_direction;
    view_direction.x = X[0] / norm;
    view_direction.y = X[1] / norm;
    view_direction.z = X[2] / norm;
    view_direction.w = 0;
    return view_direction;
}

__device__ float GetDistance2Origin(const Camera camera, const int2 p, const float depth, const float4 normal)
{
    float X[3];
    Get3DPoint(camera, p, depth, X);
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

__device__ float ComputeDepthfromPlaneHypothesis(const Camera camera, const float4 plane_hypothesis, const int2 p)
{
	return -plane_hypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * plane_hypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y + camera.K[0] * plane_hypothesis.z);
}

__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, curandState *rand_state, const float depth)
{
    float4 normal;
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s = 2.0f;
    while (s >= 1.0f) {
		q1 = 2.0f * curand_uniform(rand_state) - 1.0f;
		q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrt(1.0f - s);
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
    normal.w = 0;

    float4 view_direction = GetViewDirection(camera, p, depth);
    float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
    if (dot_product > 0.0f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    NormalizeVec3(&normal);
    return normal;
}

__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, curandState *rand_state, const float perturbation)
{
    float4 view_direction = GetViewDirection(camera, p, 1.0f);

    const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
    R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
    R[3] = cos_a2 * sin_a3;
    R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
    R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
    R[6] = -sin_a2;
    R[7] = cos_a2 * sin_a1;
    R[8] = cos_a1 * cos_a2;

    float4 normal_perturbed;
    Mat33DotVec3(R, normal, &normal_perturbed);

    if (Vec3DotVec3(normal_perturbed, view_direction) >= 0.0f) {
        normal_perturbed = normal;
    }

    NormalizeVec3(&normal_perturbed);
    return normal_perturbed;
}
__device__ float4 AddNormals(float4 normal1, float4 normal2)
{
	return make_float4(normal1.x + normal2.x, normal1.y + normal2.y, normal1.z + normal2.z, 0.0f);
}

__device__ float GetDepthFromPlane(float4 plane)
{
	// Assuming the depth is represented as the negative of the distance from the plane to the origin
	return -plane.w / (plane.x + plane.y + plane.z);
}

__device__ float4 NormalizeVec3(float4 normal)
{
	float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	return make_float4(normal.x / length, normal.y / length, normal.z / length, 0.0f);
}

__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float depth_min, const float depth_max)
{
    float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
	//Random plane hypothesis based on the camera parameters and random depth
    float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, depth);
    plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
    return plane_hypothesis;
}

__device__ float4 GeneratePertubedPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float perturbation, const float4 plane_hypothesis_now, const float depth_now, const float depth_min, const float depth_max)
{
    float depth_perturbed = depth_now;

    float dist_perturbed = plane_hypothesis_now.w;
    const float dist_min_perturbed = (1 - perturbation) * dist_perturbed;
    const float dist_max_perturbed = (1 + perturbation) * dist_perturbed;
    float4 plane_hypothesis_temp = plane_hypothesis_now;
    do {
        dist_perturbed = curand_uniform(rand_state) * (dist_max_perturbed - dist_min_perturbed) + dist_min_perturbed;
        plane_hypothesis_temp.w = dist_perturbed;
        depth_perturbed = ComputeDepthfromPlaneHypothesis(camera, plane_hypothesis_temp, p);

    } while (depth_perturbed < depth_min && depth_perturbed > depth_max);

    float4 plane_hypothesis = GeneratePerturbedNormal(camera, p, plane_hypothesis_now, rand_state, perturbation * M_PI);
    plane_hypothesis.w = dist_perturbed;
    return plane_hypothesis;
}
__device__ void ComputeHomography(const Camera &ref_camera, const Camera &src_camera, const float4 plane_hypothesis, float *H)
{
	////////////////////////////////////////////////////////////////////////////
	// 1) Camera centers in world coords:
	//    C = -R^T t    (equivalently done with row-major R in your code).
	////////////////////////////////////////////////////////////////////////////
	float ref_C[3], src_C[3];
	{
		ref_C[0] = -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]);
		ref_C[1] = -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]);
		ref_C[2] = -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2]);

		src_C[0] = -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]);
		src_C[1] = -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]);
		src_C[2] = -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2]);
	}
	////////////////////////////////////////////////////////////////////////////
	// 2) R_relative = R_j R_i^T
	////////////////////////////////////////////////////////////////////////////
	float R_relative[9];
	{
		R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] * ref_camera.R[2];
		R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] * ref_camera.R[5];
		R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] * ref_camera.R[8];

		R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] * ref_camera.R[2];
		R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] * ref_camera.R[5];
		R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] * ref_camera.R[8];

		R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] * ref_camera.R[2];
		R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] * ref_camera.R[5];
		R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] * ref_camera.R[8];
	}
	////////////////////////////////////////////////////////////////////////////
	// 3) t_relative = R_j (C_i - C_j)
	//    (This is the crucial piece: we must multiply (C_i - C_j) by R_j).
	////////////////////////////////////////////////////////////////////////////
	float c_relative[3] = { ref_C[0] - src_C[0], ref_C[1] - src_C[1], ref_C[2] - src_C[2]};
	float t_relative[3];
	{
		t_relative[0] = src_camera.R[0] * c_relative[0] + src_camera.R[1] * c_relative[1] + src_camera.R[2] * c_relative[2];
		t_relative[1] = src_camera.R[3] * c_relative[0] + src_camera.R[4] * c_relative[1] + src_camera.R[5] * c_relative[2];
		t_relative[2] = src_camera.R[6] * c_relative[0] + src_camera.R[7] * c_relative[1] +	src_camera.R[8] * c_relative[2];
	}
	////////////////////////////////////////////////////////////////////////////
	// 4) Insert plane normal n and offset d from plane_hypothesis.
	////////////////////////////////////////////////////////////////////////////
	float n[3] = { plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z };
	float d = plane_hypothesis.w;  // plane offset
	////////////////////////////////////////////////////////////////////////////
	// 5) Then the homography matrix is:
	//
	//     H = K_j [ R_j R_i^T + (t_relative n^T) / (n^T d K_i^-1 p) ] K_i^-1
	//
	//  In the code below, we produce that in the same style as your original.
	////////////////////////////////////////////////////////////////////////////
	// --- 5.1) R_j R_i^T part, but we will incorporate the second term by subtracting
	//           (similar to the original code’s approach) or adding it.
	//           If the plane eq is n^T X + d = 0, you want alpha = 1 / (n^T d K_i^-1 p).
	////////////////////////////////////////////////////////////////////////////
	// For a pixel p, you'll typically do something like:
	//    alpha = 1 / (n^T (d*K_i^-1 p))
	// but if you're using plane_hypothesis in some direct param, adapt as needed.
	float alpha = 1.f / d;  // Simplified if you are skipping the p part in this snippet
	// Construct M = R_j R_i^T - t_relative*(n^T/d)
	// or  M = R_j R_i^T + alpha*(t_relative*n^T).
	////////////////////////////////////////////////////////////////////////////
	float M[9];
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			// Original code used a minus sign, but check the sign convention in your plane eq.
			M[3 * r + c] = R_relative[3 * r + c] - alpha * t_relative[r] * n[c];
		}
	}
	////////////////////////////////////////////////////////////////////////////
	// 5.2) Multiply M by K_i^-1 on the right, then multiply by K_j on the left.
	//      This is your usual “tmp then final H” routine.
	////////////////////////////////////////////////////////////////////////////
	float tmp[9];
	{
		// K_i = [ fx  0   cx ]
		//       [ 0   fy  cy ]
		//       [ 0   0   1  ]
		// So dividing each column by fx,fy and subtracting center terms is M*K_i^-1:
		tmp[0] = M[0] / ref_camera.K[0];
		tmp[1] = M[1] / ref_camera.K[4];
		tmp[2] = M[2] - (M[0] * ref_camera.K[2] / ref_camera.K[0] + M[1] * ref_camera.K[5] / ref_camera.K[4]);

		tmp[3] = M[3] / ref_camera.K[0];
		tmp[4] = M[4] / ref_camera.K[4];
		tmp[5] = M[5] - (M[3] * ref_camera.K[2] / ref_camera.K[0] + M[4] * ref_camera.K[5] / ref_camera.K[4]);

		tmp[6] = M[6] / ref_camera.K[0];
		tmp[7] = M[7] / ref_camera.K[4];
		tmp[8] = M[8] - (M[6] * ref_camera.K[2] / ref_camera.K[0] + M[7] * ref_camera.K[5] / ref_camera.K[4]);
	}
	// Finally multiply by K_j
	{
		// K_j =  [ fx'  0    cx']
		//        [ 0    fy'  cy']
		//        [ 0    0    1 ]
		H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
		H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
		H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];

		H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
		H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
		H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];

		H[6] = src_camera.K[8] * tmp[6];
		H[7] = src_camera.K[8] * tmp[7];
		H[8] = src_camera.K[8] * tmp[8];
	}
}

__device__ float2 ComputeCorrespondingPoint(const float *H, const int2 p)
{
    float3 pt;
    pt.x = H[0] * p.x + H[1] * p.y + H[2];
    pt.y = H[3] * p.x + H[4] * p.y + H[5];
    pt.z = H[6] * p.x + H[7] * p.y + H[8];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
    transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
    transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
    transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
    transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
	const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist) / (2.0f * sigma_spatial* sigma_spatial);
    const float color_dist = (fabs(pix - center_pix)) / (2.0f * sigma_color * sigma_color);
    return exp(-spatial_dist - color_dist);
}

__device__   float SpatialGauss(float x1, float y1, float x2, float y2, float sigma, float mu = 0.0)
{
	float dis = pow(x1 - x2, 2) + pow(y1 - y2, 2) - mu;
	return exp(-1.0 * dis / (2 * sigma * sigma));
}

__device__  float RangeGauss(float x, float sigma, float mu = 0.0)
{
	float x_p = x - mu;
	return exp(-1.0 * (x_p * x_p) / (2 * sigma * sigma));
}
__device__  float UniSpatialRangeGauss(float x1, float y1, float x2, float y2, float ref_diff, float sigmaD, float sigmaR, float mu = 0.0)
{
	float dx_ref = x1 - x2;
	float dy_ref = y1 - y2;
	float spatial_sq_ref = dx_ref * dx_ref + dy_ref * dy_ref;
	float spatial_term = (spatial_sq_ref) / (2.0f * sigmaD * sigmaD);
	float texture_sq_ref = ref_diff * ref_diff;
	float texture_term = (texture_sq_ref) / (2.0f * sigmaR * sigmaR);
	float total_term = spatial_term + texture_term;
	return expf(-total_term);
}
__device__ float ComputeDualBilateralWeight(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, const float refpix, const float refcenter_pix, const float srcpix, const float srccenter_pix, const float sigma_spatial, const float sigma_color)
{
	// Compute spatial differences for the reference and source image
	float dx_ref = x1 - x2;
	float dy_ref = y1 - y2;
	float dx_src = x3 - x4;
	float dy_src = y3 - y4;
	// Use squared Euclidean distances
	float spatial_sq_ref = (dx_ref * dx_ref + dy_ref * dy_ref);
	float spatial_sq_src = (dx_src * dx_src + dy_src * dy_src);
	// Combine the spatial terms
	float spatial_term = sqrtf(spatial_sq_ref + spatial_sq_src) / (2.0f * sigma_spatial * sigma_spatial);
	// --- Texture (Intensity) Terms ---
	// Squared differences for each intensity difference
	const float ref_diff = fabsf(refpix - refcenter_pix), src_diff = fabsf(srcpix - srccenter_pix);
	float texture_sq_ref = ref_diff * ref_diff;
	float texture_sq_src = src_diff * src_diff;
	// Combine the texture terms
	float texture_term = sqrtf(texture_sq_ref + texture_sq_src) / (2.0f * sigma_color * sigma_color);
	// Total weight is given by an exponential decay of the sum of these terms
	float total_term = spatial_term + texture_term;
	return expf(-total_term);
}
__device__ float ComputeBilateralNCC(const cudaTextureObject_t ref_image, const Camera ref_camera, const cudaTextureObject_t src_image, const Camera src_camera, const int2 p, const float4 plane_hypothesis, const PatchMatchParams params, const int iter)
{
	const float cost_max = 2.0f;
	const int scale = iter;
	int radius = params.patch_size / 2;
	int radius_increment = params.radius_increment;
	int rx_start = radius, rx_end = radius + 1, ry_start = radius, ry_end = radius + 1;
	float H[9];
	ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
	float2 pt = ComputeCorrespondingPoint(H, p);
	float2 srcpt = pt; int2 refpt = p;
	int2  pr = p;         
	const float refcenterpix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

	float cost = 0.0f;
	{
		float sum_ref = 0.0f, sum_ref_ref = 0.0f, sum_src = 0.0f, sum_src_src = 0.0f, sum_ref_src = 0.0f;
		float bilateral_rsweight_sum = 0.0f;
		const float ref_center_pix = tex2D<float>(ref_image, pr.x + 0.5f, pr.y + 0.5f);
		const float src_center_pix = tex2D<float>(src_image, pt.x + 0.5f, pt.y + 0.5f);
		for (int i = -rx_start; i < rx_end; i += radius_increment) {
			float sum_ref_row = 0.0f, sum_src_row = 0.0f, sum_ref_ref_row = 0.0f, sum_src_src_row = 0.0f, sum_ref_src_row = 0.0f;
			float bilateral_rsweight_sum_row = 0.0f, rsweight = 0.0f;
			for (int j = -ry_start; j < ry_end; j += radius_increment) {
				// Get the central point for the reference image
				int2 ref_pt = make_int2(pr.x + i, pr.y + j);
				const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
				float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
				const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
				
				rsweight = ComputeDualBilateralWeight(pr.x, pr.y, ref_pt.x, ref_pt.y, pt.x, pt.y, src_pt.x, src_pt.y, ref_pix, ref_center_pix, src_pix, src_center_pix, params.sigma_spatial, params.sigma_color);
				
				sum_ref_row += rsweight * ref_pix;
				sum_ref_ref_row += rsweight * ref_pix * ref_pix;
				sum_src_row += rsweight * src_pix;
				sum_src_src_row += rsweight * src_pix * src_pix;
				sum_ref_src_row += rsweight * ref_pix * src_pix;
				bilateral_rsweight_sum_row += rsweight;	
			}
			sum_ref += sum_ref_row;
			sum_ref_ref += sum_ref_ref_row;
			sum_src += sum_src_row;
			sum_src_src += sum_src_src_row;
			sum_ref_src += sum_ref_src_row;
			bilateral_rsweight_sum += bilateral_rsweight_sum_row;
		}
		const float inv_bilateral_rsweight_sum = 1.0f / bilateral_rsweight_sum;
		sum_ref *= inv_bilateral_rsweight_sum;
		sum_ref_ref *= inv_bilateral_rsweight_sum;
		sum_src *= inv_bilateral_rsweight_sum;
		sum_src_src *= inv_bilateral_rsweight_sum;
		sum_ref_src *= inv_bilateral_rsweight_sum;

		const float var_ref = sum_ref_ref - sum_ref * sum_ref;
		const float var_src = sum_src_src - sum_src * sum_src;

		const float kMinVar = 1e-7f;
		if (var_ref < kMinVar || var_src < kMinVar) {
			return cost = cost_max;
		}
		else {
			const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
			const float var_ref_src = sqrt(var_ref * var_src);
			return cost = fmaxf(0.0f, fminf(cost_max, 1.0f - covar_src_ref / var_ref_src));
		}
	}
}

__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, unsigned int *selected_views, const PatchMatchParams params, const int iter)
{
	float cost_max = 2.0f;
	float cost_vector[32] = { 2.0f };
	float cost_vector_copy[32] = { 2.0f };
	float cost_vector_temp[32] = { 2.0f };
	int cost_count = 0;
	int num_valid_views = 0;
	//-------------------stereo view--------------------------------------
	for (int i = 1; i < params.num_images; ++i) {
		float c = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params, iter);
		cost_vector[i - 1] = c;
		cost_vector_copy[i - 1] = c;
		cost_count++;
		if (c < cost_max) {
			num_valid_views++;
		}
	}
	sort_small(cost_vector, cost_count);
	*selected_views = 0;
	//-----------------top-k-views------------------------------
	int top_k = min(max(num_valid_views, params.top_ka), min(num_valid_views, params.top_kb));
	if (top_k > 0) {
		float cost = 0.0f;
		for (int i = 0; i < top_k; ++i) {
			cost += cost_vector[i];
		}
		float cost_threshold = cost_vector[top_k - 1];
		for (int i = 0; i < params.num_images - 1; ++i) {
			if (cost_vector_copy[i] <= cost_threshold) {
				setBit(*selected_views, i);
			}
		}
		return cost / top_k;
	}
	else {
		return cost_max;
	}
	//-------------------------------------------------------
}

__device__ void ComputeMultiViewCostVector(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, float *cost_vector, const PatchMatchParams params, const int iter)
{
	//Single-plane hypothesis estimation for the depth map
	for (int i = 1; i < params.num_images; ++i) {
		cost_vector[i - 1] = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params, iter);
	}
}

__device__ float3 Get3DPointonWorld_cu(const float x, const float y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

__device__ void ProjectonCamera_cu(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

__device__ float ComputeGeomConsistencyCost(const cudaTextureObject_t depth_image, const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, const int2 p)
{
	const float max_cost = 3.0f;
	float depth = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
	float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);
	float2 src_pt;
	float src_d;
	ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);
	const float src_depth = tex2D<float>(depth_image, (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);
	if (src_depth == 0.0f) {
		return max_cost;
	}
	float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);
	float2 backward_point;
	float ref_d;
	ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);
	const float diff_col = p.x - backward_point.x;
	const float diff_row = p.y - backward_point.y;
	return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

__global__ void RandomInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses, float4 *scaled_plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter)
{
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int width = cameras[0].width;
	int height = cameras[0].height;

	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;
	curand_init(clock64(), p.y, p.x, &rand_states[center]);
	curandState local = rand_states[center];

	if (!params.geom_consistency && !params.hierarchy) {
		plane_hypotheses[center] = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], params.depth_min, params.depth_max);
		costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params, iter);
	}
	else {
		if (params.upsample) {
			const float w = float(params.scaled_cols) / float(width);
			const float h = float(params.scaled_rows) / float(height);
			const float dx = float(width) / float(params.scaled_cols);
			const float dy = float(height) / float(params.scaled_rows);
			const float sigmad = 0.50;
			const float sigmar = 25.5;
			const float Imagescale = fmaxf(dx, dy);
			int base = __float2int_ru(Imagescale); 
			int winWidth = (base % 2 == 0) ? base + 1 : base;
			const int num_neighbors = winWidth >> 1;
			const float  scale = fminf(h, w);
			const int sW = (int)params.scaled_cols;
			const int sH = (int)params.scaled_rows;
			int o_x = __float2int_rn(p.x * (float)sW / (float)width);
			int o_y = __float2int_rn(p.y * (float)sH / (float)height);
			o_x = (o_x < 0 ? 0 : (o_x >= sW ? sW - 1 : o_x));
			o_y = (o_y < 0 ? 0 : (o_y >= sH ? sH - 1 : o_y));
			const int2 o = make_int2(o_x, o_y);
			const int2 pr = p;
			const int2 ps = o;
			const int center = pr.y * width + pr.x;
			const int pcenter = pr.y * width + pr.x;
			const int scenter = ps.y * params.scaled_cols + ps.x;
			const bool srcCenterValid = (0 <= ps.x && ps.x < params.scaled_cols && 0 <= ps.y  && ps.y < params.scaled_rows);
			const bool refCenterValid = (0 <= pr.x && pr.x < width && 0 <= pr.y && pr.y < height);

			const float refPix_tex = tex2D<float>(texture_objects[0].images[0], pr.x + 0.5f, pr.y + 0.5f);
			const float srcPix_tex = tex2D<float>(texture_objects[0].images[1], ps.x + 0.5f, ps.y + 0.5f);
			const float refPix_w = plane_hypotheses[pcenter].w;
			const float srcPix_w = scaled_plane_hypotheses[scenter].w;
			float4 refNorm = plane_hypotheses[pcenter];
			float4 srcNorm = scaled_plane_hypotheses[scenter];
			float4 NrefNorm = plane_hypotheses[pcenter];
			float4 NsrcNorm = scaled_plane_hypotheses[scenter];
			float sgauss = 0.0, rgauss = 0.0, sNgauss = 0.0, rNgauss = 0.0, totalgauss = 0.0;
			float c_total_val = 0.0, w_total_val = 0.0, normalizing_factor = 0.0;
			float  NsrcPix_tex = 0, NrefPix_tex = 0;
			float  NsrcPix_w = 0, NrefPix_w = 0;
			float4 n_total_val;
			n_total_val.x = 0; n_total_val.y = 0; n_total_val.z = 0; n_total_val.w = 0;
			for (int j = -num_neighbors; j <= num_neighbors; ++j) {
				int r_y = ps.y + j; 
				r_y = (r_y >= 0 ? (r_y < sH ? r_y : sH - 1) : 0);
				int r_ys = pr.y + j;
				r_ys = (r_ys >= 0 ? (r_ys < height ? r_ys : height - 1) : 0);

				for (int i = -num_neighbors; i <= num_neighbors; ++i) {
					int r_x = ps.x + i;
					r_x = (r_x >= 0 ? (r_x < sW ? r_x : sW - 1) : 0);
					const int s_center = r_y * sW + r_x;
					if (s_center >= params.scaled_rows * params.scaled_cols) {
						printf("Illegal: %d, %d, %f, %f (%d, %d)\n", r_x, r_y, ps.x, ps.y, params.scaled_cols, params.scaled_rows);
					}
					NsrcPix_tex = tex2D<float>(texture_objects[0].images[1], r_x + 0.5f, r_y + 0.5f);
					NsrcPix_w = scaled_plane_hypotheses[s_center].w;
					NsrcNorm = scaled_plane_hypotheses[s_center];

					int r_xs = pr.x + i; // ref neighbor
					r_xs = (r_xs >= 0 ? (r_xs < width ? r_xs : width - 1) : 0);
					const int r_center = r_ys * width + r_xs;
					NrefPix_tex = tex2D<float>(texture_objects[0].images[0], r_xs + 0.5f, r_ys + 0.5f);
					NrefPix_w = plane_hypotheses[r_center].w;
					NrefNorm = plane_hypotheses[r_center];

					sgauss = SpatialGauss(ps.x, ps.y, r_x, r_y, sigmad);
					rgauss = RangeGauss(fabs(refPix_tex - NrefPix_tex), sigmar);
					totalgauss = sgauss * rgauss;

					float4 wNsrcNorm = NsrcNorm;
					normalizing_factor += totalgauss;
					w_total_val += NsrcPix_w * totalgauss;
					c_total_val += NsrcPix_tex * totalgauss;
					mul4((&wNsrcNorm), totalgauss);
					n_total_val.x += wNsrcNorm.x;
					n_total_val.y += wNsrcNorm.y;
					n_total_val.z += wNsrcNorm.z;
				}
			}

			costs[center] = c_total_val / normalizing_factor;
			vecdiv4((&n_total_val), normalizing_factor);
			NormalizeVec3(&n_total_val);

			costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params, iter);
			pre_costs[center] = costs[center];

			float4 plane_hypothesis = n_total_val;
			plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
			float depth = plane_hypotheses[center].w;
			plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
			plane_hypotheses[center] = plane_hypothesis;
			costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params, iter);
		}
		else {
			float4 plane_hypothesis;
			if (params.hierarchy) {	plane_hypothesis = scaled_plane_hypotheses[center]; }
			else { plane_hypothesis = plane_hypotheses[center]; }
			plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
			float depth = plane_hypothesis.w;
			plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
			plane_hypotheses[center] = plane_hypothesis;
			costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params, iter);
			pre_costs[center] = costs[center];
		}
	}
}

__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images, const cudaTextureObject_t *depth_images, const Camera *cameras, float4 *plane_hypothesis, float *depth, float *cost, curandState *rand_state, const float *view_weights, const float weight_norm, const int2 p, const PatchMatchParams params, const int iter)
{
	float perturbation = 0.02f;

	float depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
	float4 plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, *depth);

	float depth_perturbed = *depth;
	const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
	const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
	do {
		depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
	} while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);
	float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, perturbation * M_PI);

	const int num_planes = 10;
	float4 plane_hypothesis_average = *plane_hypothesis, plane_average = *plane_hypothesis;
	float4 plane_hypothesis_min = *plane_hypothesis;
	float depth_average = (depth_rand + *depth + depth_perturbed) / 3.0f;
	//Average of all the plane hypothesis
	plane_average.x = (plane_hypothesis->x + plane_hypothesis_rand.x + plane_hypothesis_perturbed.x) / 3.0f;
	plane_average.y = (plane_hypothesis->y + plane_hypothesis_rand.y + plane_hypothesis_perturbed.y) / 3.0f;
	plane_average.z = (plane_hypothesis->z + plane_hypothesis_rand.z + plane_hypothesis_perturbed.z) / 3.0f;
	plane_average.w = (plane_hypothesis->w + plane_hypothesis_rand.w + plane_hypothesis_perturbed.w) / 3.0f;

	//10 planes
	float depths[num_planes] = { depth_rand, *depth, depth_perturbed, depth_rand, *depth, depth_perturbed, depth_rand, *depth, depth_perturbed, depth_average };
	float4 normals[num_planes] = { *plane_hypothesis, *plane_hypothesis,*plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, plane_hypothesis_perturbed, plane_hypothesis_perturbed, plane_hypothesis_average};

	for (int i = 0; i < num_planes; ++i) {
		float cost_vector[32] = { 2.0f };
		float4 temp_plane_hypothesis = normals[i];
		temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
		ComputeMultiViewCostVector(images, cameras, p, temp_plane_hypothesis, cost_vector, params, iter);
		
		float temp_cost = 0.0f, lambda = 0.2f;
		for (int j = 0; j < params.num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				float pcost = cost_vector[j];
				if (params.geom_consistency) {
					float gcost = ComputeGeomConsistencyCost(depth_images[j + 1], cameras[0], cameras[j + 1], temp_plane_hypothesis, p);
					float ucost = (pcost + lambda * gcost);
					temp_cost += view_weights[j] * ucost;
				}
				else {
					temp_cost += view_weights[j] * pcost;
				}
			}
		}
		temp_cost /= weight_norm;

		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
		if (depth_before >= params.depth_min && depth_before <= params.depth_max && temp_cost < *cost) {
			*depth = depth_before;
			*plane_hypothesis = temp_plane_hypothesis;
			*cost = temp_cost;
		}
	}
}

__device__ void OctagramCheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, const int2 p, const PatchMatchParams params, const int iter)
{
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;
	int left_near = center - 1;
	int left_near_c = center - 1;
	int left_near_d = center - 1;
	int left_near_t = center - 1;
	int left_far = center - 1;
	int right_near = center + 1;
	int right_near_c = center + 1;
	int right_near_d = center + 1;
	int right_near_t = center + 1;
	int right_far = center + 1;
	int up_near = center - width;
	int up_near_c = center - width;
	int up_near_d = center - width;
	int up_near_t = center - width;
	int up_far = center - 1 * width;
	int down_near = center + width;
	int down_near_c = center + width;
	int down_near_d = center + width;
	int down_near_t = center + width;
	int down_far = center + 1 * width; 

	float cost_array[12][32] = { 2.0f };

	bool Octaflag[12] = { false };
	int num_valid_pixels = 0;

	float costMin; 
	int costMinPoint;
	int numpts = 45;
	// up_far and down_far
	if ((p.y > 0 && p.y < height - 1) || (p.y > 0) || (p.y < height - 1)) {

		if (p.y > 0 && p.y < height - 1) {
			if (costs[down_far] < costs[up_far]) {
				costMin = costs[down_far];
				costMinPoint = down_far;
				num_valid_pixels++;
				Octaflag[8] = true;
			}
			else {
				costMin = costs[up_far];
				costMinPoint = up_far;
				num_valid_pixels++;
				Octaflag[9] = true;
			}
		}
		else {
			if (p.y > 0) {
				costMin = costs[up_far];
				costMinPoint = up_far;
				num_valid_pixels++;
				Octaflag[8] = true;
			}
			else if (p.y < height - 1) {
				costMin = costs[down_far];
				costMinPoint = down_far;
				num_valid_pixels++;
				Octaflag[9] = true;
			}
		}
		for (int i = 1; i < numpts; ++i){
			if (p.y > 0 + 2 * i) { 
				int pointTemp = up_far - 2 * i * width; 
					if (costs[pointTemp] < costMin) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
			if (p.y < height - 1 - 2 * i) {
				int pointTemp = down_far + 2 * i * width;
					if (costs[pointTemp] < costMin) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
		}
		//Pixel index with lowest matching cost is kept
		up_far = costMinPoint;
		down_far = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_far], cost_array[8], params, iter);
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_far], cost_array[9], params, iter);
	}
	// left_far and right far
	if ((p.x > 0 && p.x < width - 1) || (p.x > 0) || (p.x < width - 1)) {

		if (p.x > 0 && p.x < width - 1) {
			if (costs[left_far] < costs[right_far]) {
				costMin = costs[left_far];
				costMinPoint = left_far;
				num_valid_pixels++;
				Octaflag[10] = true;
			}
			else {
				costMin = costs[right_far];
				costMinPoint = right_far;
				num_valid_pixels++;
				Octaflag[11] = true;
			}
		}
		else {
			if (p.x > 0) {
				costMin = costs[left_far];
				costMinPoint = left_far;
				num_valid_pixels++;
				Octaflag[10] = true;
			}
			else if (p.x < width - 1) {
				costMin = costs[right_far];
				costMinPoint = right_far;
				num_valid_pixels++;
				Octaflag[11] = true;
			}
		}
		for (int i = 1; i < numpts; ++i) {
				if (p.x > 0 + 2 * i) {
					int pointTemp = left_far - 2 * i;
					if (costs[pointTemp] < costMin) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				if (p.x < width - 1 - 2 * i) {
					int pointTemp = right_far + 2 * i;
					if (costMin < costs[pointTemp]) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
		}
		left_far = costMinPoint;
		right_far = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_far], cost_array[10], params, iter);
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_far], cost_array[11], params, iter);
	}
	// vertical left
	if ((p.x > 0 && p.y > 2 && p.y < height - 3) || (p.y > 2 && p.y < height - 3) || (p.x > 0 && p.y > 2) || (p.x > 0 && p.y < height - 3) || (p.x > 0) || (p.y > 2) || (p.y < height - 3)) {
		int left_near_up = left_near - 2 * width;
		int left_near_down = left_near + 2 * width;
		if (p.x > 0 && p.y > 2 && p.y < height - 3) {
			if (costs[left_near] < costs[left_near_up] && costs[left_near] < costs[left_near_down]) {
				costMin = costs[left_near];
				costMinPoint = left_near;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
			else if (costs[left_near_up] < costs[left_near] && costs[left_near_up] < costs[left_near_down]) {
				costMin = costs[left_near_up];
				costMinPoint = left_near_up;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
			else if (costs[left_near_down] < costs[left_near] && costs[left_near_down] < costs[left_near_up]) {
				costMin = costs[left_near_down];
				costMinPoint = left_near_down;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
		}
		if (p.x > 0 && p.y > 2) {
			if (costs[left_near_up] < costs[left_near]) {
				costMin = costs[left_near_up];
				costMinPoint = left_near_up;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
		}
		if (p.x > 0 && p.y < height - 3) {
			if (costs[left_near_down] < costs[left_near]) {
				costMin = costs[left_near_down];
				costMinPoint = left_near_down;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
		}
		if (p.y > 2 && p.y < height - 3) {
			if (costs[left_near_down] < costs[left_near_up]) {
				costMin = costs[left_near_down];
				costMinPoint = left_near_down;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
			else {
				costMin = costs[left_near_up];
				costMinPoint = left_near_up;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
		}
		else {
			if (p.y > 2) {
				costMin = costs[left_near_up];
				costMinPoint = left_near_up;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
			else if (p.y < height - 3) {
				costMin = costs[left_near_down];
				costMinPoint = left_near_down;
				num_valid_pixels++;
				Octaflag[0] = true;
			}
		}

		for (int i = 2; i < numpts; ++i) {
			if (p.y > 2 + 2 * i) {
				int pointTemp = left_near_up - 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.y < height - 3 - 2 * i) {
				int pointTemp = left_near_down + 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		left_near = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_near], cost_array[0], params, iter);
	}
	// vertical right
	if ((p.x < width - 1 && p.y > 2 && p.y < height - 3) || (p.y > 2 && p.y < height - 3) || (p.x < width - 1 && p.y > 2) || (p.x < width - 1 && p.y < height - 3) || (p.x < width - 1) || (p.y > 2) || (p.y < height - 3)) {
		int right_near_up = right_near - 2 * width;
		int right_near_down = right_near + 2 * width;
		if (p.x < width - 1 && p.y > 2 && p.y < height - 3) {
			if (costs[right_near] < costs[right_near_up] && costs[right_near] < costs[right_near_down]) {
				costMin = costs[right_near];
				costMinPoint = right_near;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
			else if (costs[right_near_up] < costs[right_near] && costs[right_near_up] < costs[right_near_down]) {
				costMin = costs[right_near_up];
				costMinPoint = right_near_up;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
			else if (costs[right_near_down] < costs[right_near] && costs[right_near_down] < costs[right_near_up]) {
				costMin = costs[right_near_down];
				costMinPoint = right_near_down;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
		}
		if (p.x < width - 1 && p.y > 2) {
			if (costs[right_near_up] < costs[right_near]) {
				costMin = costs[right_near_up];
				costMinPoint = right_near_up;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
		}
		if (p.x < width - 1 && p.y < height - 3) {
			if (costs[right_near_down] < costs[right_near]) {
				costMin = costs[right_near_down];
				costMinPoint = right_near_down;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
		}
		if (p.y > 2 && p.y < height - 3) {
			if (costs[right_near_down] < costs[right_near_up]) {
				costMin = costs[right_near_down];
				costMinPoint = right_near_down;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
			else {
				costMin = costs[right_near_up];
				costMinPoint = right_near_up;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
		}
		else {
			if (p.y > 2) {
				costMin = costs[right_near_up];
				costMinPoint = right_near_up;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
			else if (p.y < height - 3) {
				costMin = costs[right_near_down];
				costMinPoint = right_near_down;
				num_valid_pixels++;
				Octaflag[1] = true;
			}
		}
		for (int i = 2; i < numpts; ++i) {
			if (p.y > 2 + 2 * i) {
				int pointTemp = right_near_up - 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.y < height - 3 - 2 * i) {
				int pointTemp = right_near_down + 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		right_near = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_near], cost_array[1], params, iter);
	}

	// horizontal top
	if ((p.y > 1 && p.x > 2 && p.x < width - 3) || (p.x > 2 && p.x < width - 3) || (p.y > 1 && p.x > 2) || (p.y > 1 && p.x < width - 3) || (p.y > 1) || (p.x > 2) || (p.x < width - 3)) {
		int up_near_left = up_near - 2;
		int up_near_right = up_near + 2;
		if (p.y > 1 && p.x > 2 && p.x < width - 3) {
			if (costs[up_near] < costs[up_near_left] && costs[up_near] < costs[up_near_right]) {
				costMin = costs[up_near];
				costMinPoint = up_near;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
			else if (costs[up_near_left] < costs[up_near] && costs[up_near_left] < costs[up_near_right]) {
				costMin = costs[up_near_left];
				costMinPoint = up_near_left;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
			else if (costs[up_near_right] < costs[up_near] && costs[up_near_right] < costs[up_near_left]) {
				costMin = costs[up_near_right];
				costMinPoint = up_near_right;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
		}
		if (p.y > 1 && p.x > 2) {
			if (costs[up_near_left] < costs[up_near]) {
				costMin = costs[up_near_left];
				costMinPoint = up_near_left;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
		}
		if (p.y > 1 && p.x < width - 3) {
			if (costs[up_near_right] < costs[up_near]) {
				costMin = costs[up_near_right];
				costMinPoint = up_near_right;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
		}
		if (p.x > 2 && p.x < width - 3) {
			if (costs[up_near_right] < costs[up_near_left]) {
				costMin = costs[up_near_right];
				costMinPoint = up_near_right;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
			else {
				costMin = costs[up_near_left];
				costMinPoint = up_near_left;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
		}
		else {
			if (p.x > 2) {
				costMin = costs[up_near_left];
				costMinPoint = up_near_left;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
			else if (p.x < width - 3) {
				costMin = costs[up_near_right];
				costMinPoint = up_near_right;
				num_valid_pixels++;
				Octaflag[2] = true;
			}
		}
		for (int i = 2; i < numpts; ++i) {//Check if we expand to the neighbouring points
			if (p.x > 2 + 2 * i) {
				int pointTemp = up_near_left - 2 * i;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x < width - 3 - 2 * i) {
				int pointTemp = up_near_right + 2 * i;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		up_near = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_near], cost_array[2], params, iter);
	}
	// horizontal bottom left
	if ((p.y < height - 1 && p.x > 2 && p.x < width - 3) || (p.x > 2 && p.x < width - 3) || (p.y < height - 1 && p.x > 2) || (p.y < height - 1 && p.x < width - 3) || (p.y < height - 1) || (p.x > 2) || (p.x < width - 3)) {
		int down_near_left = down_near - 2;
		int down_near_right = down_near + 2;
		if (p.y < height - 1 && p.x > 2 && p.x < width - 3) {
			if (costs[down_near] < costs[down_near_left] && costs[down_near] < costs[down_near_right]) {
				costMin = costs[down_near];
				costMinPoint = down_near;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
			else if (costs[down_near_left] < costs[down_near] && costs[down_near_left] < costs[down_near_right]) {
				costMin = costs[down_near_left];
				costMinPoint = down_near_left;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
			else if (costs[down_near_right] < costs[down_near] && costs[down_near_right] < costs[down_near_left]) {
				costMin = costs[down_near_right];
				costMinPoint = down_near_right;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
		}
		if (p.y < height - 1 && p.x > 2) {
			if (costs[down_near_left] < costs[down_near]) {
				costMin = costs[down_near_left];
				costMinPoint = down_near_left;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
		}
		if (p.y < height - 1 && p.x < width - 3) {
			if (costs[down_near_right] < costs[down_near]) {
				costMin = costs[down_near_right];
				costMinPoint = down_near_right;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
		}
		if (p.x > 2 && p.x < width - 3) {
			if (costs[down_near_right] < costs[down_near_left]) {
				costMin = costs[down_near_right];
				costMinPoint = down_near_right;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
			else {
				costMin = costs[down_near_left];
				costMinPoint = down_near_left;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
		}
		else {
			if (p.x > 2) {
				costMin = costs[down_near_left];
				costMinPoint = down_near_left;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
			else if (p.x < width - 3) {
				costMin = costs[down_near_right];
				costMinPoint = down_near_right;
				num_valid_pixels++;
				Octaflag[3] = true;
			}
		}
		for (int i = 2; i < numpts; ++i) {//Check if we expand to the neighbouring points
			if (p.x > 2 + 2 * i) {
				int pointTemp = down_near_left - 2 * i;
				if (costs[pointTemp] < costMin) {//Update the costMin and costMinPoint
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x < width - 3 - 2 * i) {
				int pointTemp = down_near_right + 2 * i;
				if (costs[pointTemp] < costMin) {//Update the costMin and costMinPoint
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		down_near = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_near], cost_array[3], params, iter);
	}

	//***************************************************************************************************************//
	// diagonal - left_near and up_near
	if ((p.x > 0 && p.y > 0 && p.x < width - 1 && p.y > 0) || (p.x > 0 && p.y > 0) || (p.x < width - 1 && p.y > 0)) {
		if ((p.x > 0 && p.y > 0) && (p.x < width - 1 && p.y > 0)) {
			if (costs[left_near_d] < costs[up_near_t]) {
				costMin = costs[left_near_d];
				costMinPoint = left_near_d;
				num_valid_pixels++;
				Octaflag[4] = true;
			}
			else if (costs[up_near_t] < costs[left_near_d]) {
				costMin = costs[up_near_t];
				costMinPoint = up_near_t;
				num_valid_pixels++;
				Octaflag[4] = true;
			}
		}
		else {
			if ((p.x > 0 && p.y > 0)) {
				costMin = costs[left_near_d];
				costMinPoint = left_near_d;
				num_valid_pixels++;
				Octaflag[4] = true;
			}
			else if ((p.x < width - 1 && p.y > 0)) {
				costMin = costs[up_near_t];
				costMinPoint = up_near_t;
				num_valid_pixels++;
				Octaflag[4] = true;
			}
		}
		int pointTemp_up_near = up_near_t, pointTemp_left_near = left_near_d;
		for (int i = 1; i < numpts; ++i) {
			if (p.x < width - 1 - 2 * i && p.y > 2 * i) {
				int pointTemp = pointTemp_up_near + 2 * i - 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x > 2 * i && p.y < height - 1 - 2 * i) {
				int pointTemp = pointTemp_left_near - 2 * i + 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		left_near_d = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_near_d], cost_array[4], params, iter);
	}
	// diagonal - right_near and up_near
	if ((p.x < width - 1 && p.y > 0 && p.x > 0 && p.y > 0) || (p.x < width - 1 && p.y > 0) || (p.x > 0 && p.y > 0)) {
		if ((p.x < width - 1 && p.y > 0 && p.x > 0 && p.y > 0)) {
			if (costs[right_near_t] < costs[up_near_d]) {
				costMin = costs[right_near_t];
				costMinPoint = right_near_t;
				num_valid_pixels++;
				Octaflag[6] = true;
			}
			else if (costs[up_near_d] < costs[right_near_t]) {
				costMin = costs[up_near_d];
				costMinPoint = up_near_d;
				num_valid_pixels++;
				Octaflag[6] = true;
			}
		}
		else {
			if ((p.x < width - 1 && p.y > 0)) {
				costMin = costs[right_near_t];
				costMinPoint = right_near_t;
				num_valid_pixels++;
				Octaflag[6] = true;
			}
			else if ((p.x > 0 && p.y > 0)) {
				costMin = costs[up_near_d];
				costMinPoint = up_near_d;
				num_valid_pixels++;
				Octaflag[6] = true;
			}
		}
		int pointTemp_up_near = up_near_d, pointTemp_right_near = right_near_t;
		for (int i = 1; i < numpts; ++i) {
			if (p.x > 2 * i && p.y > 2 * i) {
				int pointTemp = pointTemp_up_near - 2 * i - 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x < width - 1 - 2 * i && p.y < height - 1 - 2 * i) {
				int pointTemp = pointTemp_right_near + 2 * i + 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		up_near_d = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_near_d], cost_array[6], params, iter);
	}

	// diagonal - left_near and down_near
	if ((p.x > 0 && p.y < height - 1 && p.x < width - 1 && p.y < height - 1) || (p.x > 0 && p.y < height - 1) || (p.x < width - 1 && p.y < height - 1)) {
		if ((p.x > 0 && p.y < height - 1 && p.x < width - 1 && p.y < height - 1)) {
			if (costs[left_near_t] < costs[down_near_d]) {
				costMin = costs[left_near_t];
				costMinPoint = left_near_t;
				num_valid_pixels++;
				Octaflag[7] = true;
			}
			else if (costs[down_near_d] < costs[left_near_t]) {
				costMin = costs[down_near_d];
				costMinPoint = down_near_d;
				num_valid_pixels++;
				Octaflag[7] = true;
			}
		}
		else {
			if ((p.x > 0 && p.y < height - 1)) {
				costMin = costs[left_near_t];
				costMinPoint = left_near_t;
				num_valid_pixels++;
				Octaflag[7] = true;
			}
			else if ((p.x < width - 1 && p.y < height - 1)) {
				costMin = costs[down_near_d];
				costMinPoint = down_near_d;
				num_valid_pixels++;
				Octaflag[7] = true;
			}
		}
		int pointTemp_left_near = left_near_t, pointTemp_down_near = down_near_d;
		for (int i = 1; i < numpts; ++i) {//Check if we expand to the neighbouring points
			if (p.x > 2 * i && p.y > 2 * i) {
				int pointTemp = pointTemp_left_near - 2 * i - 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x < width - 1 - 2 * i && p.y < height - 1 - 2 * i) {
				int pointTemp = pointTemp_down_near + 2 * i + 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		down_near_d = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_near_d], cost_array[7], params, iter);
	}
	// diagonal - right_near and down_near
	if ((p.x < width - 1 && p.y < height - 1 && p.x > 0 && p.y < height - 1) || (p.x < width - 1 && p.y < height - 1) || (p.x > 0 && p.y < height - 1)) {
		if ((p.x < width - 1 && p.y < height - 1 && p.x > 0 && p.y < height - 1)) {
			if (costs[right_near_d] < costs[down_near_t]) {
				costMin = costs[right_near_d];
				costMinPoint = right_near_d;
				num_valid_pixels++;
				Octaflag[5] = true;
			}
			else if (costs[down_near_t] < costs[right_near_d]) {
				costMin = costs[down_near_t];
				costMinPoint = down_near_t;
				num_valid_pixels++;
				Octaflag[5] = true;
			}
		}
		else {
			if ((p.x < width - 1 && p.y < height - 1)) {
				costMin = costs[right_near_d];
				costMinPoint = right_near_d;
				num_valid_pixels++;
				Octaflag[5] = true;
			}
			else if ((p.x > 0 && p.y < height - 1)) {
				costMin = costs[down_near_t];
				costMinPoint = down_near_t;
				num_valid_pixels++;
				Octaflag[5] = true;
			}
		}
		int pointTemp_right_near = right_near_d, pointTemp_down_near = down_near_t;
		for (int i = 1; i < numpts; ++i) {//Check if we expand to the neighbouring points
			if (p.x < width - 1 - 2 * i && p.y > 2 * i) {
				int pointTemp = pointTemp_right_near + 2 * i - 2 * i * width;
				if (costs[pointTemp] < costMin) {//Update the costMin and costMinPoint
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x > 2 * i && p.y < height - 1 - 2 * i) {
				int pointTemp = pointTemp_down_near - 2 * i + 2 * i * width;
				if (costs[pointTemp] < costMin) {//Update the costMin and costMinPoint
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		//Pixel index with lowest matching cost is kept
		right_near_d = costMinPoint;
		ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_near_d], cost_array[5], params, iter);
	}
	const int positions[12] = { left_near,		//vertical offset left
								right_near,		//vertical offset right
								up_near,		//horizontal offset up
								down_near,		//horizontal offset down
								left_near_d,	//diagonal left_near and up_near (45)
								right_near_d,   //diagonal right_near and down_near (45)
								up_near_d,		//diagonal right_near and up_near (135)
								down_near_d,	//diagonal left_near and down_near (135)
								up_far,			//Vertical line
								down_far,		//Vertical line 
								left_far,		//Horizontal line
								right_far };	//Horizontal line
	//Neighbouring Multi-hypotheses Joint View Selection
	float view_weights[32] = { 0.0f };
	float view_selection_priors[32] = { 0.0f };
	//Ray Khuboni - neighbouring positions considered
	for (int i = 0; i < 12; ++i) {
		if (Octaflag[i]) {
			for (int j = 0; j < params.num_images - 1; ++j) {
				view_selection_priors[i] += (isSet(selected_views[positions[j]], i)) ? 0.9f : 0.1f;
			}
		}
	}
	float sampling_probs[32] = { 0.0f };
	float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));
	// Compute sampling probabilities
	for (int i = 0; i < params.num_images - 1; i++) {
		float count = 0, tmpw = 0;
		int count_false = 0;
		for (int j = 0; j < 12; j++) {
			if (cost_array[j][i] < cost_threshold) {
				tmpw += expf(-cost_array[j][i] * cost_array[j][i] / 0.18f);
				count++;
			}
			if (cost_array[j][i] > 1.2f) {
				count_false++;
			}
		}

		if (count >= 2) {
			sampling_probs[i] = tmpw / count;
		}
		else if (count_false < 3) {
			sampling_probs[i] = expf(-cost_threshold * cost_threshold / 0.32f);
		}
		sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
	}

	// Convert PDF to CDF
	TransformPDFToCDF(sampling_probs, params.num_images - 1);

	// Random sampling using binary search on CDF
	for (int sample = 0; sample < 15; ++sample) {
		float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;
		int selected_view = binarySearchCDF(sampling_probs, params.num_images - 1, rand_prob);
		view_weights[selected_view] += 1.0f;
	}

	unsigned int temp_selected_views = 0;
	int num_selected_view = 0;
	float weight_norm = 0;
	for (int i = 0; i < params.num_images - 1; ++i)
	{
		if (view_weights[i] > 0) {
			setBit(temp_selected_views, i);
			weight_norm += view_weights[i];
			num_selected_view++;
		}
	}
	float final_costs[12] = { 0.0f }, lambda = 0.2f, cgcosts[32] = { 0.0f };
	float geo_sigma = 2 * 5.0f * 5.0f;
	float cost_sigma = 2 * 0.5f * 0.5f;
	float depth_s = (params.depth_max - params.depth_min) / 64.0f;
	float depth_sigma = 2 * depth_s * depth_s;
	float angle_s = M_PI * (5.0f / 180.0f);
	float angle_sigma = 2 * angle_s * angle_s;
	int cgcount = 0;
	for (int i = 0; i < params.num_images - 1; ++i) {
		cgcount = 0;
		for (int j = 0; j < 12; ++j) {
			if (!Octaflag[j])
				continue;
			if (params.geom_consistency) {
				float depth_i0 = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
				float depth_i1 = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[j]], p);
				float relative_depth_diff = fabs(depth_i0 - depth_i1) / depth_i0;
				float angle_cos = Vec3DotVec3(plane_hypotheses[center], plane_hypotheses[positions[j]]);
				float angle_diff = acos(angle_cos);
				float geo_cost = ComputeGeomConsistencyCost(depths[i + 1], cameras[0], cameras[i + 1], plane_hypotheses[positions[j]], p);
				float geo_conf = exp(-geo_cost * geo_cost / geo_sigma);
				float depth_conf = exp(-relative_depth_diff * relative_depth_diff / depth_sigma);
				float angle_conf = exp(-angle_diff * angle_diff / angle_sigma);
				float cost_conf = exp(-cost_array[j][i] * cost_array[j][i] / cost_sigma);
				cgcosts[i] += geo_conf * depth_conf * angle_conf * cost_conf;
				cgcount++;
			}
		}
		cgcosts[i] /= cgcount;
	}

	for (int j = 0; j < 12; ++j) {
		for (int i = 0; i < params.num_images - 1; ++i) {
			if (view_weights[i] > 0) {
				if (!Octaflag[j])
					continue;
				float pcost = cost_array[j][i];
				if (params.geom_consistency) {
					float ucost = (pcost + lambda * fabs(1.0f - cgcosts[i]));
					final_costs[j] += view_weights[i] * ucost;
				}
				else {
					final_costs[j] += view_weights[i] * pcost;
				}
			}
		}
		final_costs[j] = final_costs[j] > 0.0f ? final_costs[j] / weight_norm : FLT_MAX;
	}
	const int min_cost_idx = FindMinCostIndex(final_costs, 12);


	float cost_vector_now[32] = { 2.0f };
	for (int i = 0; i < params.num_images - 1; ++i)
		cost_vector_now[i] = 2.0f;
	ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[center], cost_vector_now, params, iter);

	float cost_now = 0.0f;
	for (int i = 0; i < params.num_images - 1; ++i) {
		if (view_weights[i] > 0) {
			float pcost = cost_vector_now[i];
			if (params.geom_consistency) {
				float gcost = ComputeGeomConsistencyCost(depths[i + 1], cameras[0], cameras[i + 1], plane_hypotheses[center], p);
				float ucost = (pcost + lambda * gcost);
				cost_now += view_weights[i] * ucost;
			}
			else {
				cost_now += view_weights[i] * pcost;
			}
		}
	}
	cost_now /= weight_norm;
	costs[center] = cost_now;
	float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	bool depth_now_flag = depth_now >= params.depth_min && depth_now <= params.depth_max;

	float4 plane_hypotheses_now = plane_hypotheses[center];
	float depth_before;
	if (Octaflag[min_cost_idx]) {
		depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);
		bool depth_before_flag = depth_before >= params.depth_min && depth_before <= params.depth_max;

		if (depth_before_flag && final_costs[min_cost_idx] < cost_now) {
			depth_now = depth_before;
			plane_hypotheses_now = plane_hypotheses[positions[min_cost_idx]];
			cost_now = final_costs[min_cost_idx];
			selected_views[center] = temp_selected_views;
		}

	}

	PlaneHypothesisRefinement(images, depths, cameras, &plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], view_weights, weight_norm, p, params, iter);

	if (params.hierarchy) {
		if (cost_now < pre_costs[center] - 0.1f){
			costs[center] = cost_now;
			plane_hypotheses[center] = plane_hypotheses_now;
		}
		else {
			costs[center] = pre_costs[center];
			plane_hypotheses[center] = plane_hypotheses[center];
		}
	}
	else {
		costs[center] = cost_now;
		plane_hypotheses[center] = plane_hypotheses_now;
	}
}

__global__ void BlackPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}

	OctagramCheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs, rand_states, selected_views, p, params, iter);
}

__global__ void RedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}

	OctagramCheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs, rand_states, selected_views, p, params, iter);
}

__global__ void GetDepthandNormal(Camera *cameras, float4 *plane_hypotheses, const PatchMatchParams params)
{
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = cameras[0].width;
	const int height = cameras[0].height;

	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;
	plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__device__ void CheckerboardFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const int2 p)
{
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;

	float filter[21];
	int index = 0;

	filter[index++] = plane_hypotheses[center].w;

	// Left
	const int left = center - 1;
	const int leftleft = center - 3;

	// Up
	const int up = center - width;
	const int upup = center - 3 * width;

	// Down
	const int down = center + width;
	const int downdown = center + 3 * width;

	// Right
	const int right = center + 1;
	const int rightright = center + 3;

	if (costs[center] < 0.001f) {
		return;
	}

	if (p.y > 0) {
		filter[index++] = plane_hypotheses[up].w;
	}
	if (p.y > 2) {
		filter[index++] = plane_hypotheses[upup].w;
	}
	if (p.y > 4) {
		filter[index++] = plane_hypotheses[upup - width * 2].w;
	}
	if (p.y < height - 1) {
		filter[index++] = plane_hypotheses[down].w;
	}
	if (p.y < height - 3) {
		filter[index++] = plane_hypotheses[downdown].w;
	}
	if (p.y < height - 5) {
		filter[index++] = plane_hypotheses[downdown + width * 2].w;
	}
	if (p.x > 0) {
		filter[index++] = plane_hypotheses[left].w;
	}
	if (p.x > 2) {
		filter[index++] = plane_hypotheses[leftleft].w;
	}
	if (p.x > 4) {
		filter[index++] = plane_hypotheses[leftleft - 2].w;
	}
	if (p.x < width - 1) {
		filter[index++] = plane_hypotheses[right].w;
	}
	if (p.x < width - 3) {
		filter[index++] = plane_hypotheses[rightright].w;
	}
	if (p.x < width - 5) {
		filter[index++] = plane_hypotheses[rightright + 2].w;
	}
	if (p.y > 0 &&
		p.x < width - 2) {
		filter[index++] = plane_hypotheses[up + 2].w;
	}
	if (p.y < height - 1 &&
		p.x < width - 2) {
		filter[index++] = plane_hypotheses[down + 2].w;
	}
	if (p.y > 0 &&
		p.x > 1)
	{
		filter[index++] = plane_hypotheses[up - 2].w;
	}
	if (p.y < height - 1 &&
		p.x>1) {
		filter[index++] = plane_hypotheses[down - 2].w;
	}
	if (p.x > 0 &&
		p.y > 2)
	{
		filter[index++] = plane_hypotheses[left - width * 2].w;
	}
	if (p.x < width - 1 &&
		p.y>2)
	{
		filter[index++] = plane_hypotheses[right - width * 2].w;
	}
	if (p.x > 0 &&
		p.y < height - 2) {
		filter[index++] = plane_hypotheses[left + width * 2].w;
	}
	if (p.x < width - 1 &&
		p.y < height - 2) {
		filter[index++] = plane_hypotheses[right + width * 2].w;
	}

	sort_small(filter, index);
	int median_index = index / 2;
	if (index % 2 == 0) {
		plane_hypotheses[center].w = (filter[median_index - 1] + filter[median_index]) / 2;
	}
	else {
		plane_hypotheses[center].w = filter[median_index];
	}
}

__global__ void BlackPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}

	//CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

__global__ void RedPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}

	//CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

void OCMM::RunPatchMatch()
{
	const int width = cameras[0].width;
	const int height = cameras[0].height;

	int BLOCK_W = 32;
	int BLOCK_H = (BLOCK_W / 2);

	dim3 grid_size_randinit;
	grid_size_randinit.x = (width + 16 - 1) / 16;
	grid_size_randinit.y = (height + 16 - 1) / 16;
	grid_size_randinit.z = 1;
	dim3 block_size_randinit;
	block_size_randinit.x = 16;
	block_size_randinit.y = 16;
	block_size_randinit.z = 1;

	dim3 grid_size_checkerboard;
	grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
	grid_size_checkerboard.y = ((height / 2) + BLOCK_H - 1) / BLOCK_H;
	grid_size_checkerboard.z = 1;
	dim3 block_size_checkerboard;
	block_size_checkerboard.x = BLOCK_W;
	block_size_checkerboard.y = BLOCK_H;
	block_size_checkerboard.z = 1;

	int max_iterations = params.max_iterations, scale = 1;

	RandomInitialization << <grid_size_randinit, block_size_randinit >> > (texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, scaled_plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, params, scale);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	for (int i = 0; i < max_iterations; ++i) {
		BlackPixelUpdate << <grid_size_checkerboard, block_size_checkerboard >> > (texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, params, i);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		RedPixelUpdate << <grid_size_checkerboard, block_size_checkerboard >> > (texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, params, i);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}

	GetDepthandNormal << <grid_size_randinit, block_size_randinit >> > (cameras_cuda, plane_hypotheses_cuda, params);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	BlackPixelFilter << <grid_size_checkerboard, block_size_checkerboard >> > (cameras_cuda, plane_hypotheses_cuda, costs_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	RedPixelFilter << <grid_size_checkerboard, block_size_checkerboard >> > (cameras_cuda, plane_hypotheses_cuda, costs_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
// -----------------------------------------------------------------------------
// JBU kernel
// -----------------------------------------------------------------------------
__global__ void JBUc_cu(JBUParameters *jp, JBUTexObj *jt, float *depth, const int level = 0)
{
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int rows = jp[0].height;
	const int cols = jp[0].width;
	if (p.x >= cols || p.y >= rows) return;

	const int center = p.y * cols + p.x;
	const float scaled_h = float(jp[0].s_height) / float(jp[0].height);
	const float scaled_w = float(jp[0].s_width) / float(jp[0].width);
	const float iscaled_h = float(jp[0].height) / float(jp[0].s_height);
	const float iscaled_w = float(jp[0].width) / float(jp[0].s_width);
	const float scale = fminf(scaled_h, scaled_w);
	const float iscale = fmaxf(iscaled_h, iscaled_w);
	const float sigmad = 0.50f;
	const float sigmar = 25.5f;

	int base = __float2int_ru(iscale);
	int winWidth = (base % 2 == 0) ? base + 1 : base;
	const int num_neighbors = winWidth >> 1;
	int o_x = __float2int_rn(p.x * scale);
	int o_y = __float2int_rn(p.y * scale);
	const int2 o = make_int2(o_x, o_y);
	const float scale_x = float(jp[0].s_width) / float(jp[0].width);
	const float scale_y = float(jp[0].s_height) / float(jp[0].height);
	const float ref_center_pix = tex2D<float>(jt[0].imgs[0], p.x + 0.5f, p.y + 0.5f);
	const int radius = num_neighbors;
	const int radius_inc = 1;
	const int2 pr =  p;
	const int2 ps =  o;
	const float refCenter = tex2D<float>(jt[0].imgs[0], pr.x + 0.5f, pr.y + 0.5f);

	float total_val = 0.0f;
	float normalizing_factor = 0.0f;

	for (int j = -num_neighbors; j <= num_neighbors; ++j) {
		int ry = ps.y + j;
		ry = (ry >= 0 ? (ry < jp[0].s_height ? ry : jp[0].s_height - 1) : 0);

		int rys = pr.y + j;
		rys = (rys >= 0 ? (rys < jp[0].height ? rys : jp[0].height - 1) : 0);

		for (int i = -num_neighbors; i <= num_neighbors; ++i) {
			int rx = ps.x + i;
			rx = (rx >= 0 ? (rx < jp[0].s_width ? rx : jp[0].s_width - 1) : 0);

			int rxs = pr.x + i;
			rxs = (rxs >= 0 ? (rxs < jp[0].width ? rxs : jp[0].width - 1) : 0);

			const float Nref = tex2D<float>(jt[0].imgs[0], rxs + 0.5f, rys + 0.5f);
			const float Nsrc = tex2D<float>(jt[0].imgs[1], rx + 0.5f, ry + 0.5f);

			float w = 0.0f;
			const float dRef = fabsf(refCenter - Nref);

			w = UniSpatialRangeGauss(
				ps.x, ps.y, rx, ry,
				dRef, sigmad, sigmar);

			normalizing_factor += w;
			total_val += Nsrc * w;
		}
	}
	depth[center] = (total_val / normalizing_factor);
}

void JBU::CudaRun(const int scale)
{
    int rows = jp_h.height;
    int cols = jp_h.width;

    dim3 grid_size_initrand;
    grid_size_initrand.x= (cols + 16 - 1) / 16;
    grid_size_initrand.y= (rows + 16 - 1) / 16;
    grid_size_initrand.z= 1;
    dim3 block_size_initrand;
    block_size_initrand.x = 16;
    block_size_initrand.y = 16;
    block_size_initrand.z = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaDeviceSynchronize();
	JBUc_cu <<< grid_size_initrand, block_size_initrand >>> (jp_d, jt_d, depth_d, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(depth_h, depth_d, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total time needed for computation: %f seconds\n", milliseconds / 1000.f);
}
