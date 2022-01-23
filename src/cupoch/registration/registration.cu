/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/
#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/registration.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/device_vector.h"
#include "cupoch/utility/helper.h"

#include <Eigen/Geometry>
#include <chrono>

using namespace cupoch;
using namespace cupoch::registration;
using namespace std::chrono;

namespace {

	RegistrationResult GetRegistrationResultAndCorrespondencesModified(
			const geometry::PointCloud &source,
			const geometry::PointCloud &target,
			float max_correspondence_distance,
			const Eigen::Matrix4f &transformation,
			utility::device_vector<int> &indices,
			utility::device_vector<float> &dists) 
	{


		RegistrationResult result(transformation);
		if (max_correspondence_distance <= 0.0) {
			return result;
		}

		const int n_pt = source.points_.size();

		result.correspondence_set_.resize(n_pt);

		const float error2 = thrust::transform_reduce(
				utility::exec_policy(0)->on(0),
				dists.begin(), dists.end(),
				[] __device__(float d) { return (isinf(d)) ? 0.0 : d; }, 0.0f,
				thrust::plus<float>());

		thrust::transform(enumerate_begin(indices), enumerate_end(indices),
				result.correspondence_set_.begin(),
				[] __device__(const thrust::tuple<int, int> &idxs) {
				int j = thrust::get<1>(idxs);
				return (j < 0) ? Eigen::Vector2i(-1, -1)
				: Eigen::Vector2i(thrust::get<0>(idxs),
						j);
				});

		auto end = thrust::remove_if(result.correspondence_set_.begin(),
				result.correspondence_set_.end(),
				[] __device__(const Eigen::Vector2i &x) -> bool {
				return (x[0] < 0);
				});

		//cudaStreamSynchronize(stream);
		int n_out = thrust::distance(result.correspondence_set_.begin(), end);

		result.correspondence_set_.resize(n_out);

		if (result.correspondence_set_.empty()) {
			result.fitness_ = 0.0;
			result.inlier_rmse_ = 0.0;
		} else {
			size_t corres_number = result.correspondence_set_.size();
			result.fitness_ = (float)corres_number / (float)source.points_.size();
			result.inlier_rmse_ = std::sqrt(error2 / (float)corres_number);
		}
		return result;
	}

	RegistrationResult GetRegistrationResultAndCorrespondences(
			const geometry::PointCloud &source,
			const geometry::PointCloud &target,
			const geometry::KDTreeFlann &target_kdtree,
			float max_correspondence_distance,
			const Eigen::Matrix4f &transformation) {
		RegistrationResult result(transformation);
		if (max_correspondence_distance <= 0.0) {
			return result;
		}

		const int n_pt = source.points_.size();


		utility::device_vector<int> indices(n_pt);
		utility::device_vector<float> dists(n_pt);

		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		target_kdtree.SearchRadius(source.points_, max_correspondence_distance, 1,
				indices, dists);

		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

		/*
			 std::cout << "Time taken " << time_span.count() << " seconds.\n";
		 */

		result.correspondence_set_.resize(n_pt);

		const float error2 = thrust::transform_reduce(
				utility::exec_policy(0)->on(0),
				dists.begin(), dists.end(),
				[] __device__(float d) { return (isinf(d)) ? 0.0 : d; }, 0.0f,
				thrust::plus<float>());

		thrust::transform(enumerate_begin(indices), enumerate_end(indices),
				result.correspondence_set_.begin(),
				[] __device__(const thrust::tuple<int, int> &idxs) {
				int j = thrust::get<1>(idxs);
				return (j < 0) ? Eigen::Vector2i(-1, -1)
				: Eigen::Vector2i(thrust::get<0>(idxs),
						j);
				});

		auto end = thrust::remove_if(result.correspondence_set_.begin(),
				result.correspondence_set_.end(),
				[] __device__(const Eigen::Vector2i &x) -> bool {
				return (x[0] < 0);
				});

		//cudaStreamSynchronize(stream);
		int n_out = thrust::distance(result.correspondence_set_.begin(), end);

		result.correspondence_set_.resize(n_out);

		if (result.correspondence_set_.empty()) {
			result.fitness_ = 0.0;
			result.inlier_rmse_ = 0.0;
		} else {
			size_t corres_number = result.correspondence_set_.size();
			result.fitness_ = (float)corres_number / (float)source.points_.size();
			result.inlier_rmse_ = std::sqrt(error2 / (float)corres_number);
		}
		return result;
	}

}  // namespace

RegistrationResult::RegistrationResult(const Eigen::Matrix4f &transformation)
	: transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}

RegistrationResult::RegistrationResult(const RegistrationResult &other)
	: transformation_(other.transformation_),
	correspondence_set_(other.correspondence_set_),
	inlier_rmse_(other.inlier_rmse_),
	fitness_(other.fitness_) {}

	RegistrationResult::~RegistrationResult() {}

	void RegistrationResult::SetCorrespondenceSet(
			const thrust::host_vector<Eigen::Vector2i> &corres) {
		correspondence_set_ = corres;
	}

thrust::host_vector<Eigen::Vector2i> RegistrationResult::GetCorrespondenceSet()
	const {
		thrust::host_vector<Eigen::Vector2i> corres = correspondence_set_;
		return corres;
	}

RegistrationResult cupoch::registration::EvaluateRegistration(
		const geometry::PointCloud &source,
		const geometry::PointCloud &target,
		float max_correspondence_distance,
		const Eigen::Matrix4f
		&transformation /* = Eigen::Matrix4d::Identity()*/) {
	geometry::KDTreeFlann kdtree(target);
	geometry::PointCloud pcd = source;
	if (!transformation.isIdentity()) {
		pcd.Transform(transformation);
	}
	return GetRegistrationResultAndCorrespondences(
			pcd, target, kdtree, max_correspondence_distance, transformation);
}

RegistrationResult cupoch::registration::RegistrationICP(
		const geometry::PointCloud &source,
		const geometry::PointCloud &target,
		float max_correspondence_distance,
		const Eigen::Matrix4f &init /* = Eigen::Matrix4f::Identity()*/,
		const TransformationEstimation &estimation
		/* = TransformationEstimationPointToPoint(false)*/,
		const ICPConvergenceCriteria
		&criteria /* = ICPConvergenceCriteria()*/) {
	if (max_correspondence_distance <= 0.0) {
		utility::LogError("Invalid max_correspondence_distance.");
	}

	//std::cout << (int)source.points_.size() << "\n";

	if ((estimation.GetTransformationEstimationType() ==
				TransformationEstimationType::PointToPlane ||
				estimation.GetTransformationEstimationType() ==
				TransformationEstimationType::ColoredICP) &&
			!target.HasNormals()) {
		utility::LogError(
				"TransformationEstimationPointToPlane and "
				"TransformationEstimationColoredICP "
				"require pre-computed target normal vectors.");
	}

	Eigen::Matrix4f transformation = init;
	geometry::KDTreeFlann kdtree(target);
	geometry::PointCloud pcd = source;
	if (init.isIdentity() == false) {
		pcd.Transform(init);
	}
	RegistrationResult result;
	result = GetRegistrationResultAndCorrespondences(
			pcd, target, kdtree, max_correspondence_distance, transformation);
	for (int i = 0; i < criteria.max_iteration_; i++) {
		utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
				result.fitness_, result.inlier_rmse_);
		Eigen::Matrix4f update = estimation.ComputeTransformation(
				pcd, target, result.correspondence_set_);
		transformation = update * transformation;
		pcd.Transform(update);
		RegistrationResult backup = result;
		result = GetRegistrationResultAndCorrespondences(
				pcd, target, kdtree, max_correspondence_distance,
				transformation);
		/*
			 if (std::abs(backup.fitness_ - result.fitness_) <
			 criteria.relative_fitness_ &&
			 std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
			 criteria.relative_rmse_) {
			 break;
			 }
		 */
	}
	return result;
}

RegistrationResult GetRegistrationResultAndCorrespondenceso3d(
		const geometry::PointCloud &source,
		const geometry::PointCloud &target,
		double max_correspondence_distance,
		const Eigen::Matrix4f &transformation,
		utility::device_vector<int> &indices,
		utility::device_vector<float> &dists) {
	RegistrationResult result(transformation);

	if (max_correspondence_distance <= 0.0) {
		return result;
	}

	float error2 = 0.0;
	for (int i = 0; i < (int)source.points_.size(); i++) {
		if ( indices[i] > 0) {
			result.correspondence_set_.push_back(
					Eigen::Vector2i(i, indices[i]));
		}
		if ( isinf(dists[i]) == 0) {
			error2 += dists[i];
		} 
	}

	if (result.correspondence_set_.empty()) {
		result.fitness_ = 0.0;
		result.inlier_rmse_ = 0.0;
	} else {
		size_t corres_number = result.correspondence_set_.size();
		result.fitness_ = (float)corres_number / (float)source.points_.size();
		result.inlier_rmse_ = std::sqrt(error2 / (float)corres_number);
	}
	return result;
}

Eigen::Matrix4f ComputeTransformationmodified(
		const geometry::PointCloud &source,
		const geometry::PointCloud &target,
		utility::device_vector<int> &indices) {

	/*
		 if (corres.empty()) return Eigen::Matrix4f::Identity();
	 */

	int count = 0;
	for (int i = 0; i < (int)source.points_.size(); i++){
		if(indices[i] > 0){
			count += 1;
		}
	}

	std::cout << (int)source.points_.size() << ":" << count << "\n";

	Eigen::MatrixXf source_mat(3, count);
	Eigen::MatrixXf target_mat(3, count);

	Eigen::Vector3f model_center = Eigen::Vector3f::Zero();
	Eigen::Vector3f target_center = Eigen::Vector3f::Zero();

	int j = 0;
	for (int i = 0; i < (int)source.points_.size(); i++){
		if (indices[i] > 0){

			Eigen::Vector3f src = source.points_[i];
			Eigen::Vector3f tgt = target.points_[indices[i]];

			source_mat.block<3, 1>(0, j) = src;
			target_mat.block<3, 1>(0, j) = tgt;

			model_center += source_mat.block<3, 1>(0, j);
			target_center += target_mat.block<3, 1>(0, j);

			j += 1;
		}
	}


	float divided_by = 1.0f / source.points_.size();
	model_center *= divided_by;
	target_center *= divided_by;


	Eigen::Matrix3f hh = Eigen::Matrix3f::Zero();

	for (size_t i = 0; i < count; i++) {
		hh += (source_mat.block<3, 1>(0, i) - model_center) * (target_mat.block<3, 1>(0, i) - target_center).transpose();
	}

	hh /= source.points_.size();

	std::cout << "from c code :\n" << hh << "\n";

	Eigen::JacobiSVD<Eigen::Matrix3f> svd(
			hh, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f ss = Eigen::Matrix3f::Identity();
	ss(2, 2) = (svd.matrixU() * svd.matrixV()).determinant();
	Eigen::Matrix4f tr = Eigen::Matrix4f::Identity();
	tr.block<3, 3>(0, 0) = svd.matrixV() * ss * svd.matrixU().transpose();

	// The translation
	tr.block<3, 1>(0, 3) = target_center;
	tr.block<3, 1>(0, 3) -= tr.block<3, 3>(0, 0) * model_center;

	return tr;
}

Eigen::Matrix4f SVDfunction(Eigen::Matrix3f hh, Eigen::Vector3f model_center, Eigen::Vector3f target_center) {
	//if (corres.empty()) return Eigen::Matrix4f::Identity();
	//SVD by Eigen
	//high_resolution_clock::time_point start_time, end_time;
	//duration<double> time_span;
	//start_time = high_resolution_clock::now();
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(
			hh, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f ss = Eigen::Matrix3f::Identity();
	ss(2, 2) = (svd.matrixU() * svd.matrixV()).determinant();
	Eigen::Matrix4f tr = Eigen::Matrix4f::Identity();
	tr.block<3, 3>(0, 0) = svd.matrixV() * ss * svd.matrixU().transpose();

	// The translation
	tr.block<3, 1>(0, 3) = target_center;
	tr.block<3, 1>(0, 3) -= tr.block<3, 3>(0, 0) * model_center;
	//end_time = high_resolution_clock::now();
	//time_span = duration_cast<duration<double>>(end_time - start_time);
	//std::cout << "SVD time : " << time_span.count() << "\n";


	return tr;
}



static const int SUM_THREAD  = 512; 
__global__ void test_kernel(Eigen::Vector3f* p_source,
		const Eigen::Vector3f* p_target,
		int *indices,
		int *start_index,
		int *size,
		Eigen::Vector3f *model_cen,
		Eigen::Vector3f *target_cen,
		float *hh,
		Eigen::Matrix4f *update_ker){

	int pcd_index = blockIdx.x;
	int tid = threadIdx.x;

	int	index = start_index[pcd_index];
	int sz = size[pcd_index];
	float divided_by = 1.0f / sz;

	float3 local_sum_source, local_sum_target;
	float local_hh_sum[3][3];

	float3 source_n;
	float3 target_n;

	local_sum_source.x = local_sum_source.y = local_sum_source.z = 0.0;
	local_sum_target.x = local_sum_target.y = local_sum_target.z = 0.0;

	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			local_hh_sum[i][j] = 0.0;
		}
	}

	for(int i = index+tid; i < index + sz; i += SUM_THREAD){
		if(indices[i] > 0){
			local_sum_source.x += p_source[i][0];
			local_sum_source.y += p_source[i][1];
			local_sum_source.z += p_source[i][2];

			local_sum_target.x += p_target[indices[i]][0];
			local_sum_target.y += p_target[indices[i]][1];
			local_sum_target.z += p_target[indices[i]][2];
		}
	}

	__shared__ float3 sum[SUM_THREAD];

	sum[tid].x = local_sum_source.x;
	sum[tid].y = local_sum_source.y;
	sum[tid].z = local_sum_source.z;
	__syncthreads(); 

	for(int i = SUM_THREAD/2; i > 0; i/=2){
		if(tid < i){
			sum[tid].x += sum[tid + i].x; 
			sum[tid].y += sum[tid + i].y; 
			sum[tid].z += sum[tid + i].z;
		}
		__syncthreads();
	}

	if(tid == 0){
		model_cen[pcd_index][0] = sum[0].x * divided_by;
		model_cen[pcd_index][1] = sum[0].y * divided_by;
		model_cen[pcd_index][2] = sum[0].z * divided_by;
	}


	sum[tid].x = local_sum_target.x;
	sum[tid].y = local_sum_target.y;
	sum[tid].z = local_sum_target.z;
	__syncthreads(); 

	for(int i = SUM_THREAD/2; i > 0; i/=2){
		if(tid < i){
			sum[tid].x += sum[tid + i].x; 
			sum[tid].y += sum[tid + i].y; 
			sum[tid].z += sum[tid + i].z; 
		}
		__syncthreads();
	}

	if(tid == 0){
		target_cen[pcd_index][0] = sum[0].x * divided_by;
		target_cen[pcd_index][1] = sum[0].y * divided_by;
		target_cen[pcd_index][2] = sum[0].z * divided_by;
	}
	__syncthreads();

	for(int i = index+tid; i < index + sz; i += SUM_THREAD){
		if(indices[i] > 0){

			source_n.x = p_source[i][0] - model_cen[pcd_index][0];
			source_n.y = p_source[i][1] - model_cen[pcd_index][1];
			source_n.z = p_source[i][2] - model_cen[pcd_index][2];

			target_n.x = p_target[indices[i]][0] - target_cen[pcd_index][0];
			target_n.y = p_target[indices[i]][1] - target_cen[pcd_index][1];
			target_n.z = p_target[indices[i]][2] - target_cen[pcd_index][2];

			local_hh_sum[0][0] += source_n.x * target_n.x;
			local_hh_sum[0][1] += source_n.x * target_n.y;
			local_hh_sum[0][2] += source_n.x * target_n.z;

			local_hh_sum[1][0] += source_n.y * target_n.x;
			local_hh_sum[1][1] += source_n.y * target_n.y;
			local_hh_sum[1][2] += source_n.y * target_n.z;

			local_hh_sum[2][0] += source_n.z * target_n.x;
			local_hh_sum[2][1] += source_n.z * target_n.y;
			local_hh_sum[2][2] += source_n.z * target_n.z;
		}
	}

	__shared__ float h_sum[SUM_THREAD][3][3];

	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			h_sum[tid][i][j] = local_hh_sum[i][j];
		}
	}
	__syncthreads();

	for(int i = SUM_THREAD/2; i > 0; i/=2){
		if(tid < i){

			for(int j = 0; j < 3; j++){
				for(int k = 0; k < 3; k++){
					h_sum[tid][j][k] += h_sum[tid + i][j][k]; 
				}
			}
		}
		__syncthreads();
	}

	if(tid == 0){
		int start = pcd_index * 9;
		int q = 0;
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				hh[start + q] = h_sum[0][i][j] * divided_by;
				q+=1;
			}
		}
	

    __syncthreads();
    //SVD

    float al, b, c, l, t, cs, sn, tmp, sign;
    int i, j, p, k;
    Eigen::Matrix3f u = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f v = Eigen::Matrix3f::Identity();
    Eigen::Vector3f s = Eigen::Vector3f::Zero();
    int n = 3;
    for (p = 0; p < 10; p++){
	for (i=0; i<n; i++){
	    for (j=i+1; j<n; j++){	
		al = b = c = l = t = cs = sn = tmp = sign = 0.0;
		for (k=0; k<n; k++){
		    al += hh[9 * pcd_index + 3*k + i] * hh[9 * pcd_index + 3*k + i];
		    b += hh[9 * pcd_index + 3*k + j] * hh[9 * pcd_index + 3*k + j];
		    c += hh[9 * pcd_index + 3*k + i] * hh[9 * pcd_index + 3*k + j];
		}

		l = (b - al)/(2.0 * c);
		sign = 1.0;
		if (l < 0.0)
			sign = -1.0;
		t = sign / ((sign*l) + sqrt(1.0 + l*l));
		cs = 1.0/sqrt(1.0 + t*t);
		sn = cs *t;

		for (k=0; k<n; k++){
			tmp = hh[9 * pcd_index + 3*k + i];
			hh[9 * pcd_index + 3*k + i] = cs*tmp - sn*hh[9 * pcd_index + 3*k + j];
			hh[9 * pcd_index + 3*k + j] = sn*tmp + cs*hh[9 * pcd_index + 3*k + j];
		}

		for (k=0; k<n; k++){
			tmp = v(k,i);
			v(k,i) = cs*tmp - sn*v(k,j);
			v(k,j) = sn*tmp + cs*v(k,j);
		}
	    }
        }
    }

    for (j=0; j<n; j++){
	for (i=0; i<n; i++){
		s(j) += hh[9 * pcd_index + 3*i + j] * hh[9 * pcd_index + 3*i + j];
	}	
	tmp = s(j);
	s(j) = sqrt(tmp);
    } 

    for (p=0; p<(n-1); p++){
	for (j=0; j<n-p-1; j++){
	    if (s[j] < s[j+1]){
		tmp = s(j);
		s(j) = s(j+1);
		s(j+1) = tmp;

		for (i=0; i<n; i++){
    		    tmp = v(i,j);
		    v(i,j) = v(i,j+1);
		    v(i,j+1) = tmp;
		    tmp = hh[9 * pcd_index + 3*i + j];
		    hh[9 * pcd_index + 3*i + j] = hh[9 * pcd_index + 3*i + j + 1];
		    hh[9 * pcd_index + 3*i + j + 1] = tmp;
		}
	    }
	}
    }

    for (i=0; i<n; i++){
	for (j=0; j<n; j++){
		hh[9 * pcd_index + 3*i + j] = hh[9 * pcd_index + 3*i + j]/s(j);
	}
    }

    for(int q = 0; q < 9; q++){
        u(q/3, q%3) = hh[9 * pcd_index + q];

    }

    Eigen::Matrix3f ss = Eigen::Matrix3f::Identity();
    ss(2, 2) = (u * v).determinant();
    Eigen::Matrix4f tr = Eigen::Matrix4f::Identity();
    tr.block<3, 3>(0, 0) = v * ss * u.transpose();

    // The translation
    tr.block<3, 1>(0, 3) = target_cen[pcd_index];
    tr.block<3, 1>(0, 3) -= tr.block<3, 3>(0, 0) * model_cen[pcd_index];
    update_ker[pcd_index] = tr;

    }


	return;
}

std::vector<RegistrationResult> cupoch::registration::AkashaRegistrationICP(

		std::vector<geometry::PointCloud> &source,
		const geometry::PointCloud &target,
		float max_correspondence_distance,
		const geometry::KDTreeFlann &kdtree,
		const int point_counts,
		std::vector<Eigen::Matrix4f> &init,
		const TransformationEstimation &estimation,
		const ICPConvergenceCriteria
		&criteria /* = ICPConvergenceCriteria()*/) {

	high_resolution_clock::time_point start_time, end_time;
        duration<double> time_span_init(0.0);
	//start_time = high_resolution_clock::now();

	if (max_correspondence_distance <= 0.0) {
		utility::LogError("Invalid max_correspondence_distance.");
	}

	if ((estimation.GetTransformationEstimationType() ==
				TransformationEstimationType::PointToPlane ||
				estimation.GetTransformationEstimationType() ==
				TransformationEstimationType::ColoredICP) &&
			!target.HasNormals()) {
		utility::LogError(
				"TransformationEstimationPointToPlane and "
				"TransformationEstimationColoredICP "
				"require pre-computed target normal vectors.");
	}

	Eigen::Matrix4f update[point_counts];
	std::vector<RegistrationResult> result(point_counts);

	utility::device_vector<int> source_start_index(point_counts);
	utility::device_vector<int> source_size(point_counts);
	utility::device_vector<Eigen::Vector3f> model_cen(point_counts);
	utility::device_vector<Eigen::Vector3f> target_cen(point_counts);
	utility::device_vector<Eigen::Matrix4f> update_ker(point_counts);
	utility::device_vector<float> hh(point_counts * 9);


	int total_point = 0;
	int size;

	for(int iteration = 0; iteration < point_counts; iteration+=1){
		source_start_index[iteration] = total_point;
		source_size[iteration] = source[iteration].points_.size();
		total_point += source[iteration].points_.size();
	}

	utility::device_vector<int> m_indices(total_point);
	utility::device_vector<float> m_dists(total_point);

	utility::device_vector<Eigen::Vector3f> s_point(total_point);
	Eigen::Matrix3f hh_svd;

	utility::device_vector<int> indices_arr[point_counts];
	utility::device_vector<float> dists_arr[point_counts];

	for(int iteration = 0; iteration < point_counts; iteration+=1){

		size = source[iteration].points_.size();

		utility::device_vector<int> p(size);
		indices_arr[iteration] = p;

		utility::device_vector<float> q(size);
		dists_arr[iteration] = q;

	}

	for(int iteration = 0; iteration < point_counts; iteration+=1){
		if (init[iteration].isIdentity() == false) {
			source[iteration].Transform(init[iteration]);
		}
	}
	//end_time = high_resolution_clock::now();
        //time_span_init = duration_cast<duration<double>>(end_time - start_time);
	duration<double> time_span_kdtree(0.0);
	duration<double> time_span_kernel(0.0);
	duration<double> time_span_memcpy(0.0);
	duration<double> time_span_update_d2h(0.0);


	for (int i = 0; i < criteria.max_iteration_+1; i++) {

		//start_time = high_resolution_clock::now();
		
		total_point = 0;
		for(int iteration = 0; iteration < point_counts; iteration+=1){
			thrust::copy(source[iteration].points_.begin(), source[iteration].points_.end(), s_point.begin() + total_point);
			total_point += source[iteration].points_.size();
		}

		kdtree.SearchRadius(s_point, max_correspondence_distance, 1,                                                                                                                                                  
				m_indices, m_dists);

		total_point = 0;
		for(int iteration = 0; iteration < point_counts; iteration+=1){
			size = source[iteration].points_.size();
			thrust::copy(m_indices.begin() + total_point,  m_indices.begin() + total_point + size , indices_arr[iteration].begin());
			thrust::copy(m_dists.begin() + total_point,  m_dists.begin() + total_point + size , dists_arr[iteration].begin());
			total_point += size;
		}
		//end_time = high_resolution_clock::now();
		//time_span_kdtree += duration_cast<duration<double>>(end_time - start_time);

		//start_time = high_resolution_clock::now();

		test_kernel <<< point_counts, SUM_THREAD >>>(thrust::raw_pointer_cast(s_point.data()),
				thrust::raw_pointer_cast(target.points_.data()),
				thrust::raw_pointer_cast(m_indices.data()),
				thrust::raw_pointer_cast(source_start_index.data()),
				thrust::raw_pointer_cast(source_size.data()),
				thrust::raw_pointer_cast(model_cen.data()),
				thrust::raw_pointer_cast(target_cen.data()),
				thrust::raw_pointer_cast(hh.data()),
				thrust::raw_pointer_cast(update_ker.data()));
		cudaDeviceSynchronize();
		
		//end_time = high_resolution_clock::now();
		//time_span_kernel += duration_cast<duration<double>>(end_time - start_time);
	

		//start_time = high_resolution_clock::now();
		cudaMemcpy(update[0].data(), thrust::raw_pointer_cast(update_ker.data()), 
				sizeof(Eigen::Matrix4f) * point_counts, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		for(int iteration = 0; iteration < point_counts; iteration+=1){
			
			if(i == criteria.max_iteration_){
				result[iteration] = GetRegistrationResultAndCorrespondencesModified(
						source[iteration], target, max_correspondence_distance,
						init[iteration], indices_arr[iteration], dists_arr[iteration]);
			}
			/*
			start_time = high_resolution_clock::now();
			update[iteration] = update_ker[iteration];
			end_time = high_resolution_clock::now();
			time_span_update_d2h += duration_cast<duration<double>>(end_time - start_time);
			*/ 
			init[iteration] = update[iteration] * init[iteration];
			source[iteration].Transform(update[iteration]);
		}
		//end_time = high_resolution_clock::now();
                //time_span_memcpy += duration_cast<duration<double>>(end_time - start_time);

	}
	//std::cout << "Time taken for kdtree : " << time_span_kdtree.count() << "\n";
	//std::cout << "Time taken for memcpy + transform : " << time_span_memcpy.count() << "\n";
	//std::cout << "Time taken for kernel : " << time_span_kernel.count() << "\n";
	//std::cout << "Time taken for init : " << time_span_init.count() << "\n";
	/*
	std::cout << "Time taken for update d2h : " << time_span_update_d2h.count() << "\n";
	*/
	return result;
}

