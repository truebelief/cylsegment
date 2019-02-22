////cyl_segmenter.hpp
////Description: region-growing-based cylinder clustering under constraints
////Author: Zhouxin Xi (zhouxin.xi@uleth.ca)
////Created On: Feb 22, 2019
////Reference: https://www.mdpi.com/1999-4907/7/11/252/htm

#pragma once
#include "stdafx.h"
#include "cyl_util.hpp"

#include <algorithm>
#include <random>

#include <armadillo>
#include <nanoflann.hpp>
#include <Eigen/Dense>
#include <functional>

using namespace std;
using namespace arma;
using namespace Eigen;

class CylSegment
{
public:

	//brutal range search for small amount of points, using only xyz
	//input: arma::mat& pcd_mat, arma::mat& query_mat, double radius
	//output: arma::uvec& nn_idx
	//index is 0-based
	void BruteRadSearch(arma::mat& pcd_mat, arma::mat& query_mat, double radius, arma::uvec& nn_idx)
	{
		if (pcd_mat.n_rows < 1)
		{
			return;
		}

		arma::uvec logi= arma::zeros<arma::uvec>(pcd_mat.n_rows);
		for (int k = 0; k < query_mat.n_rows; ++k)
		{
			arma::vec d = arma::ones<arma::vec>(pcd_mat.n_rows);
			//for (int t = 0; t < query_mat.n_cols;++t)
			for (int t = 0; t < 3; ++t)
			{
				d = d % (arma::abs(pcd_mat.col(t)- query_mat(k,t))<radius);
			}
			logi(arma::find(d > 0)) += 1;
		}
		nn_idx = find(logi > 0);
	}
	
	//range search based on nanoflann kdtree methods
	//input: nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>& tindex, arma::uvec& pcd_indices, arma::mat& query_mat, double radius
	//output: arma::uvec& nn_idx
	//tindex is the prebuilt kd tree; pcd_indices indicates subset of source points; query_mat is the query points; radius is search radius; nn_idx is the index in source points
	//index is 0-based
	void RadSearch(nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>& tindex, arma::uvec& pcd_indices, arma::mat& query_mat, double radius, arma::uvec& nn_idx)
	{
		if (query_mat.n_cols!=3)
		{
			query_mat = query_mat.t();
		}

		std::vector<std::pair<size_t, double>> ret_matches;

		nanoflann::SearchParams params;
		params.sorted = false;

		size_t n_matches;
		vector<int> ret_indices;
		for (int i = 0; i < query_mat.n_rows; i++)
		{
			vector<double> each_row = arma::conv_to<vector<double>>::from(query_mat.row(i));
			n_matches = tindex.index->radiusSearch(&each_row[0], radius, ret_matches, params);
			vector<int> ret_idx;
			std::transform(ret_matches.begin(), ret_matches.end(), std::back_inserter(ret_idx), [](const std::pair<int, int>& p) { return p.first; });
			ret_indices.insert(std::end(ret_indices), std::begin(ret_idx), std::end(ret_idx));
		}
		
		nn_idx = arma::unique(arma::conv_to<arma::uvec>::from(ret_indices));
		if (pcd_indices.n_elem > 0)
		{
			nn_idx = nn_idx(arma::find(pcd_indices(nn_idx) > 0));
		}
	}

	//approximately partition points into unconnected regions, using region growing method
	//input: arma::mat& mats, double gap
	//output: arma::uvec& partitions_inds
	//gap is the minimum gaps between connected regions; partitions_inds stores ID of each connected region (from 0);input mats should be xyz
	//ID is 0-based
	void FindConnectedPartitions(arma::mat& mats, double gap, arma::uvec& partitions_inds)
	{
		//bin points into voxels at gap size
		arma::mat mats_xyz = mats.cols(span(0, 2));
		arma::mat mat_min = arma::conv_to<arma::mat>::from(arma::min(mats_xyz, 0));
		arma::mat mat_max = arma::conv_to<arma::mat>::from(arma::max(mats_xyz, 0));
		arma::uvec key_shape = arma::conv_to<arma::uvec>::from(arma::floor((mat_max - mat_min) / gap + 1));
		arma::umat mat_even = arma::conv_to<arma::umat>::from(arma::floor((mats_xyz - arma::repmat(mat_min, mats_xyz.n_rows, 1)) / gap));
		mat_even = mat_even.t();

		arma::uvec mat_key = arma::sub2ind(arma::size(key_shape[0], key_shape[1], key_shape[2]), mat_even);
		arma::uvec mat_key_ind_unq;
		arma::umat mat_key_unq;
		arma::uvec mat_key_unq_index3;

		util_.UniqueIndex(mat_key, mat_key_ind_unq, mat_key_unq_index3);
		mat_key_unq = arma::ind2sub(arma::size(key_shape[0], key_shape[1], key_shape[2]), mat_key_ind_unq).t();

		arma::uvec part_inds = arma::zeros<arma::uvec>(mat_key_unq.n_rows);
		int class_iter = 1;
		arma::uvec part_inds_rest_idx = arma::linspace<arma::uvec>(0, mat_key_unq.n_rows - 1, mat_key_unq.n_rows);
		int n_part_inds_left = mat_key_unq.n_rows;

		//connected regions mean connected voxels
		//grow voxels into regions unitl no neighbor was found
		while (n_part_inds_left > 0)
		{
			arma::uvec nn_logi(1); nn_logi(0) = 0;
			arma::uvec part_nn_idx(1);
			part_nn_idx(0) = part_inds_rest_idx(0); //start region growing from 1st remaining point
			while (part_nn_idx.n_elem)
			{
				part_inds(part_nn_idx) = class_iter * arma::ones<uvec>(part_nn_idx.n_elem);
				util_.RemoveRows(part_inds_rest_idx, nn_logi);
				if (part_inds_rest_idx.n_elem<1)
				{
					break;
				}
				arma::mat subrow = arma::conv_to<arma::mat>::from(mat_key_unq.rows(part_inds_rest_idx));
				arma::mat subquery = arma::conv_to<arma::mat>::from(mat_key_unq.rows(part_nn_idx));
				BruteRadSearch(subrow, subquery, 2.0, nn_logi); //region growing until there is no neighboring voxels
				part_nn_idx = part_inds_rest_idx(nn_logi);
			}
			class_iter++;;
			n_part_inds_left = part_inds_rest_idx.n_elem;
		}
		partitions_inds = part_inds(mat_key_unq_index3);
	}

	//Randomly shuffle a vector
	//input: arma::uvec arma_nums
	//output: arma::uvec
	//random seed is set to 1 here
	arma::uvec randomShuffle(arma::uvec arma_nums)
	{
		arma::uvec arma_nums_unq;
		arma::uvec arma_nums_index3;
		util_.UniqueIndex(arma_nums, arma_nums_unq, arma_nums_index3);

		arma_rng::set_seed(1);
		arma::uvec arma_nums_unq_shuffle = shuffle(arma_nums_unq);
		arma::uvec arma_nums_shuffle = arma_nums_unq_shuffle(arma_nums_index3);

		return arma_nums_shuffle;
	}

	//main function
	//segment point clouds into cylindrical regions
	//input: PointCloudPtr pcd, const double rg_step, const double rg_mingap, string foutpath
	//output: a text file (x,y,z,regionID)
	//rg_step is the search range at each step of region growing; rg_mingap is the minimum gap between connected regions;
	//rg_step controls growing speed, small value means fine region resolution but slower speed
	//rg_mingap controls partition level
	void SegmentCylinders(arma::mat& pcd_mat, const double rg_step, const double rg_mingap, string foutpath)
	{
		std::cout << "2. Segmenting cylinders" << std::endl;
		std::cout << "region growing step:" << rg_step << std::endl;
		std::cout << "region minimum gap:" << rg_mingap << std::endl;

		block_size_ = rg_step * 5;//limit search range within an approximate block size, used to speed up computation.

		//arma::mat tip_indexs;
		//tip_indexs.load("tip_index.txt", raw_ascii);
		//tip_indexs = tip_indexs - 1;
		//arma::uvec tip_index = arma::conv_to<arma::uvec>::from(tip_indexs.col(0));

		//std::vector<double> seg_widths;

		arma::mat pcd_xyz= pcd_mat.cols(span(0,2));
		Eigen::MatrixXd eig_xyz = Eigen::Map<Eigen::MatrixXd>(pcd_xyz.memptr(),pcd_xyz.n_rows,pcd_xyz.n_cols);
		//pcd_mat = pcd_mat.t();

		//build nanoflann kd tree
		//eig_xyz= eig_xyz.transpose();
		nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> tindex(3, eig_xyz, 10);
		tindex.index->buildIndex();

		int n_scene_pts = pcd_mat.n_rows;
		arma::mat scene_pts_pos = pcd_mat.cols(span(0,2));
		arma::uvec seg_IDs = arma::zeros<arma::uvec>(n_scene_pts);

		//flag vectors (true: 1; false: 0);
		arma::uvec seg_flags_rest = arma::ones<arma::uvec>(n_scene_pts); //indicate remaining ungrown points
		arma::uvec seg_flags_avail = arma::ones<arma::uvec>(n_scene_pts);//indicate available starting points
		arma::uvec seg_flags_tipset = arma::zeros<arma::uvec>(n_scene_pts);//indicate available tip points from FastFindTip function

		int start_ind=0;
		if (tip_index_.n_elem > 0)
		{
			seg_flags_tipset.rows(tip_index_) = arma::ones<arma::uvec>(tip_index_.size());
			arma::uvec start_inds = arma::find(seg_flags_tipset>0, 1, "first");
			start_ind = start_inds(0);
		}


		int seg_id = 1;
		arma::mat col_index = arma::conv_to<arma::mat>::from(arma::linspace<uvec>(0, pcd_mat.n_rows-1,pcd_mat.n_rows));
		pcd_mat.insert_cols(3,col_index);

		bool tip_all_found = false;
		int iter = 0;

		while (start_ind > -1)
		{
			iter = iter + 1;
			int iter_grow = 0;

			if (iter % 100 == 0)
			{
				arma::uvec seg_processed=arma::find(seg_flags_avail > 0);
				cout << "segmenting progress: " << (1.0 - seg_processed.n_elem *1.0 / n_scene_pts) << std::endl;
				//cout << "iteration: " << iter << std::endl;
			}

			bool bound_end1_stop_flag = false;
			bool bound_end2_stop_flag = true; //%initially, bound_end2 is not active; grow bound_end1 first

			double bound_end1_width;
			double bound_end2_width;
			double bound_end1_width_sum = 0;
			double bound_end2_width_sum = 0;
			double bound_end1_width_count = 0;
			double bound_end2_width_count = 0;

			arma::rowvec bound_end1_vec_start = arma::zeros<rowvec>(3);
			arma::rowvec bound_end2_vec_start = arma::zeros<rowvec>(3);
			arma::rowvec bound_end1_vec_prev = arma::zeros<rowvec>(3);
			arma::rowvec bound_end2_vec_prev = arma::zeros<rowvec>(3);
			arma::rowvec bound_end1_vec = arma::zeros<rowvec>(3);
			arma::rowvec bound_end2_vec = arma::zeros<rowvec>(3);

			arma::rowvec start_pt = arma::conv_to<arma::rowvec>::from(pcd_mat.row(start_ind));
			arma::rowvec start_pos = start_pt(span(0, 2));

			arma::mat bound_end1(start_pt); 
			arma::mat bound_end2(start_pt);
			arma::rowvec bound_end1_centroid = start_pos;
			arma::rowvec bound_end2_centroid = start_pos;

			arma::rowvec bound_end1_centroid_prev = start_pos;
			arma::rowvec bound_end2_centroid_prev = start_pos;

			bool linear_grow = false; //(linear_grow == true) means the region growing reaches boundary of a tube
			arma::rowvec bound_end1_vec0;

			arma::uvec logi_range;

			RadSearch(tindex, seg_flags_rest, start_pos, block_size_, logi_range);
			arma::mat rest_range_pts= pcd_mat.rows(logi_range);

			// growing points to maximally allowed constraint
			while (iter_grow < 30)//limit the growing iteration to 30, can be customized.
			{
				//search nearest points within a range
				iter_grow = iter_grow + 1;
				if (!bound_end1_stop_flag)
				{
					arma::uvec logi_bound1;
					if (bound_end1.n_rows > 5) //decimate search points to voxels for faster range search
					{
						arma::mat bound_end1_pos = arma::conv_to<arma::mat>::from(bound_end1.cols(span(0, 2)));
						arma::umat unq;
						util_.UniqueByRow(arma::conv_to<arma::umat>::from(arma::round((bound_end1_pos - arma::repmat(arma::min(bound_end1_pos, 0), bound_end1_pos.n_rows, 1)) / (rg_step_ / 4.0))),unq);
						arma::mat bound_end1_approx_bin = arma::conv_to<arma::mat>::from(unq);
												
						bound_end1_approx_bin = bound_end1_approx_bin.cols(span(0, 2));
						arma::mat bound_end1_approx_pos = bound_end1_approx_bin * rg_step_ / 4 + arma::repmat(arma::min(bound_end1_pos, 0), bound_end1_approx_bin.n_rows, 1);
						 BruteRadSearch(rest_range_pts, bound_end1_approx_pos, rg_step_, logi_bound1);
					}
					else
					{
						BruteRadSearch(rest_range_pts, bound_end1, rg_step_, logi_bound1);
					}
					if (logi_bound1.n_elem > 0)
					{
						bound_end1 = rest_range_pts.rows(logi_bound1);
						util_.RemoveRows(rest_range_pts, logi_bound1);
						if (bound_end1.n_rows < 2) //exclude itself
						{
							bound_end1_stop_flag = true;
						}
					}
					else
					{
						bound_end1_stop_flag = true;
						
					}
				}


				if (!bound_end2_stop_flag  && linear_grow)
				{
					arma::uvec logi_bound2;
					if (bound_end2.n_rows > 5) //decimate search points to voxels for faster range search
					{
						arma::mat bound_end2_pos = arma::conv_to<arma::mat>::from(bound_end2.cols(span(0, 2)));
						arma::umat unq;
						util_.UniqueByRow(arma::conv_to<arma::umat>::from(arma::round((bound_end2_pos - arma::repmat(arma::min(bound_end2_pos, 0), bound_end2_pos.n_rows, 1)) / (rg_step_ / 4.0))), unq);
						arma::mat bound_end2_approx_bin = arma::conv_to<arma::mat>::from(unq);


						arma::mat bound_end2_approx_pos = bound_end2_approx_bin * rg_step_ / 4 + arma::repmat(arma::min(bound_end2_pos, 0), bound_end2_approx_bin.n_rows, 1);
						BruteRadSearch(rest_range_pts, bound_end2_approx_pos, rg_step_, logi_bound2);
					}
					else
					{
						BruteRadSearch(rest_range_pts, bound_end2, rg_step_, logi_bound2);
					}
					if (logi_bound2.n_elem > 0)
					{
						bound_end2 = rest_range_pts.rows(logi_bound2);
						util_.RemoveRows(rest_range_pts, logi_bound2);
					}
					else
					{
						bound_end2_stop_flag = true;
					}
				}

				if (bound_end1_stop_flag && bound_end2_stop_flag)
				{
					break;
				}

				//partition nearest points within the range into two biggest ends;
				//the two ends will keep growing in opposite directions
				int centerednum1 = 1;
				int centerednum2 = 1;
				arma::uvec partitions_inds1;
				arma::uvec partitions_inds2;
				if (!bound_end1_stop_flag)
				{
					FindConnectedPartitions(bound_end1, rg_mingap_, partitions_inds1);
					centerednum1 = arma::max(partitions_inds1);
				}
				if (!bound_end2_stop_flag  && linear_grow)
				{
					FindConnectedPartitions(bound_end2, rg_mingap_, partitions_inds2);
					centerednum2 = arma::max(partitions_inds2);
				}
				arma::mat centered_end1_1 = bound_end1;
				arma::mat centered_end1_2;
				if (!bound_end1_stop_flag)
				{
					if (centerednum1 > 1)
					{

						arma::umat unqind1;
						util_.UniqueByRow(partitions_inds1, unqind1);
						uvec centered_size = unqind1.col(2);
						uvec sz_ind = sort_index(centered_size, "descend")+1;

						centered_end1_1 = bound_end1.rows(arma::find(partitions_inds1 == sz_ind(0)));
						arma::rowvec centered_end1_1_dir = arma::mean(centered_end1_1.cols(span(0, 2)), 0) - start_pos;
						centered_end1_1_dir = arma::normalise(centered_end1_1_dir, 2);

						centered_end1_2 = bound_end1.rows(arma::find(partitions_inds1 == sz_ind(1)));
						arma::rowvec centered_end1_2_dir = arma::mean(centered_end1_2.cols(span(0, 2)), 0) - start_pos;
						centered_end1_2_dir = arma::normalise(centered_end1_2_dir, 2);

						if (dot(centered_end1_1_dir, centered_end1_2_dir) > -0.5)//<120 degree, avoid two ends along same growing direction 
						{
							centered_end1_2.clear();
						}
					}
				}

				arma::mat centered_end2_1 = bound_end2;
				arma::mat centered_end2_2;
				if (!bound_end2_stop_flag  && linear_grow)
				{
					if (centerednum2 > 1)
					{
						arma::umat unqind2;
						util_.UniqueByRow(partitions_inds2, unqind2);
						uvec centered_size = unqind2.col(2);
						uvec sz_ind = sort_index(centered_size, "descend")+1;

						centered_end2_1 = bound_end2.rows(arma::find(partitions_inds2 == sz_ind(0)));
						arma::rowvec centered_end2_1_dir = arma::mean(centered_end2_1.cols(span(0, 2)), 1) - start_pos;
						centered_end2_1_dir = arma::normalise(centered_end2_1_dir, 2);

						centered_end2_2 = bound_end2.rows(arma::find(partitions_inds2 == sz_ind(1)));
						arma::rowvec centered_end2_2_dir = arma::mean(centered_end2_2.cols(span(0, 2)), 1) - start_pos;
						centered_end2_2_dir = arma::normalise(centered_end2_2_dir, 2);

						if (dot(centered_end2_1_dir, centered_end2_2_dir) > -0.5)//<120 degree, avoid two ends along same growing direction 
						{
							centered_end2_2.clear();
						}
					}
				}


				if (centered_end1_2.n_rows > centered_end2_1.n_rows)
				{
					bound_end1 = centered_end1_1;
					bound_end2 = centered_end1_2;
					bound_end2_stop_flag = false;
					bound_end2_width_count = 1;
					bound_end2_width_sum = bound_end1_width_sum / bound_end1_width_count;
					bound_end2_centroid_prev = arma::mean(bound_end2.cols(span(0, 2)), 0) + bound_end1_vec * rg_mingap_;
				}
				else
				{
					bound_end1 = centered_end1_1;
					bound_end2 = centered_end2_1;
				}


				if (bound_end1.n_rows > 0 && bound_end2.n_rows > 0)
				{
					if (!bound_end1_stop_flag)
					{
						bound_end1_centroid = mean(bound_end1.cols(span(0, 2)), 0);
						bound_end1_vec = bound_end1_centroid - bound_end1_centroid_prev;
						bound_end1_vec = arma::normalise(bound_end1_vec, 2);
						arma::mat bound_end1_diff = bound_end1.cols(span(0, 2)) - arma::repmat(bound_end1_centroid, bound_end1.n_rows, 1);
						//% is element-wise multiple operator in armadillo
						arma::mat bound_end1_widths = bound_end1_diff - arma::repmat(arma::sum(bound_end1_diff % arma::repmat(bound_end1_vec, bound_end1.n_rows, 1), 1), 1, 3) % arma::repmat(bound_end1_vec, bound_end1.n_rows, 1);
						arma::vec bound_end1_width_v=arma::sqrt(arma::sum(bound_end1_widths % bound_end1_widths, 1));
						bound_end1_width = util_.Quantile(bound_end1_width_v, 0.9);//count the width of end

						bound_end1_vec_start = bound_end1_centroid - start_pos;
						bound_end1_vec_start = arma::normalise(bound_end1_vec_start, 2);

						double bound_end1_avg_width = bound_end1_width_sum / bound_end1_width_count;

						double width_incre = (bound_end1_width - bound_end1_avg_width) / bound_end1_avg_width;
						if (width_incre < 0.2 && (!linear_grow)) //an end has little width increase, meaning the region growing has hit the tube boundary
						{
							linear_grow = true;
							if (iter_grow < 3) //ignoring initial end widths, since initial points are sparse and width is unreliable;
							{
								bound_end1_width_sum = 0;
								bound_end1_width_count = 1;
							}
							bound_end1_vec0 = bound_end1_vec;
						}
						if ((!linear_grow) && (iter_grow > 2) && (bound_end1_avg_width < rg_mingap_))//force region growing if end width is small
						{
							linear_grow = true;
							bound_end1_vec0 = bound_end1_vec;
						}


						if (linear_grow && width_incre > width_ratio_limit_)//pauses growing, if width increases drastically 
						{
							bound_end1_stop_flag = true;
							arma::uvec subcol1 = arma::conv_to<arma::uvec>::from(bound_end1.col(3));
							seg_flags_avail(subcol1) = arma::zeros<uvec>(bound_end1.n_rows);//inactivate flags of available starting points
						}
						else
						{
							bound_end1_width_sum = bound_end1_width_sum + bound_end1_width;
							bound_end1_width_count = bound_end1_width_count + 1;
						}
					}
				
					if (!bound_end2_stop_flag && linear_grow)
					{
						bound_end2_centroid = arma::mean(bound_end2.cols(span(0,2)), 0);
						bound_end2_vec = bound_end2_centroid - bound_end2_centroid_prev;
						bound_end2_vec = arma::normalise(bound_end2_vec, 2);
						arma::mat bound_end2_diff = bound_end2.cols(span(0, 2)) - arma::repmat(bound_end2_centroid, bound_end2.n_rows, 1);


						arma::mat bound_end2_widths = bound_end2_diff - arma::repmat(arma::sum(bound_end2_diff % arma::repmat(bound_end2_vec, bound_end2.n_rows, 1), 1), 1, 3) % arma::repmat(bound_end2_vec, bound_end2.n_rows, 1);
						arma::vec bound_end2_width_v = arma::sqrt(arma::sum(bound_end2_widths % bound_end2_widths, 1));
						bound_end2_width = util_.Quantile(bound_end2_width_v, 0.9);

						bound_end2_vec_start = bound_end2_centroid - start_pos;
						bound_end2_vec_start = arma::normalise(bound_end2_vec_start, 2); // may be nan value, but not a problem

						double bound_end2_avg_width = bound_end2_width_sum / bound_end2_width_count;

						if ((iter_grow > 1) && ((bound_end2_width - bound_end2_avg_width) / bound_end2_avg_width > width_ratio_limit_))//pauses growing, if width increases drastically
						{
							bound_end2_stop_flag = true;arma::conv_to<arma::uvec>::from(bound_end1.col(3));
							arma::uvec subcol1 = arma::conv_to<arma::uvec>::from(bound_end1.col(3));
							seg_flags_avail(subcol1) = arma::zeros<uvec>(bound_end1.n_rows);//inactivate flags of available starting points
						}
						else
						{
							bound_end2_width_sum = bound_end2_width_sum + bound_end2_width;
							bound_end2_width_count = bound_end2_width_count + 1;
						}
					}

					if (linear_grow)
					{
						if (arma::dot(bound_end1_vec_start, bound_end2_vec_start) > 0)//terminate growing if two ends no longer grow in opposite directions
						{
							arma::uvec subcol1 = arma::conv_to<arma::uvec>::from(bound_end1.col(3));
							arma::uvec subcol2 = arma::conv_to<arma::uvec>::from(bound_end2.col(3));
							seg_flags_avail(subcol1) = arma::zeros<uvec>(bound_end1.n_rows);//inactivate flags of available starting points
							seg_flags_avail(subcol2) = arma::zeros<uvec>(bound_end2.n_rows);//inactivate flags of available starting points
							break;
						}
						if (arma::dot(bound_end1_vec, bound_end1_vec_prev) < direction_change_limit_) //60 degree, terminate growing if growing direction of an end changes drastically
						{
							arma::uvec subcol1 = arma::conv_to<arma::uvec>::from(bound_end1.col(3));
							seg_flags_avail(subcol1) = arma::zeros<uvec>(bound_end1.n_rows);//turn off flags of available
							break;
						}

						if (arma::dot(bound_end2_vec, bound_end2_vec_prev) < direction_change_limit_) //60 degree, terminate growing if growing direction of an end changes drastically
						{
							arma::uvec subcol2 = arma::conv_to<arma::uvec>::from(bound_end2.col(3));
							seg_flags_avail(subcol2) = arma::zeros<uvec>(bound_end2.n_rows);//inactivate flags of available starting points
							break;
						}
						if (bound_end1_vec0.n_elem > 0)
						{
							double bound_end1_accum_angle = acos(arma::dot(bound_end1_vec, bound_end1_vec0))*3.1415926 / 180.0;
							if (bound_end1_accum_angle > 90)//90 degree, terminate growing if accumulative growing direction changes beyond 90 degree
							{
								arma::uvec subcol2 = arma::conv_to<arma::uvec>::from(bound_end2.col(3));
								seg_flags_avail(subcol2) = arma::zeros<uvec>(bound_end2.n_rows);//inactivate flags of available starting points
								break;
							}
						}

						bound_end1_vec_prev = bound_end1_vec;
						bound_end2_vec_prev = bound_end2_vec;

					}
					bound_end1_centroid_prev = bound_end1_centroid;
					bound_end2_centroid_prev = bound_end2_centroid;
				}
				if (!bound_end1_stop_flag)
				{
					arma::uvec subcol1 = arma::conv_to<arma::uvec>::from(bound_end1.col(3));
					seg_IDs(subcol1) = seg_id * arma::ones<uvec>(bound_end1.n_rows);
					seg_flags_rest(subcol1) = arma::zeros<uvec>(bound_end1.n_rows);//inactivate flags of grown points
					seg_flags_avail(subcol1) = arma::zeros<uvec>(bound_end1.n_rows);//inactivate flags of available starting points
				}
				if (!bound_end2_stop_flag)
				{
					arma::uvec subcol2 = arma::conv_to<arma::uvec>::from(bound_end2.col(3));
					seg_IDs(subcol2) = seg_id * arma::ones<uvec>(bound_end2.n_rows);
					seg_flags_rest(subcol2) = arma::zeros<uvec>(bound_end2.n_rows);//inactivate flags of grown points
					seg_flags_avail(subcol2) = arma::zeros<uvec>(bound_end2.n_rows);//inactivate flags of available starting points
				}
			}

			if (iter_grow > 1)
			{
				//seg_widths.push_back(bound_end1_width);//record width of each segment, can be useful
				seg_id++;
			}

			//choose next starting seed of region growing;
			//priority is given to tip points 
			if (!tip_all_found)
			{
				arma:: uvec start_inds = arma::find((seg_flags_avail % seg_flags_tipset)>0, 1, "first");
				
				if (start_inds.n_elem == 0)
				{
					start_inds = arma::find(seg_flags_avail>0, 1, "first");
					start_ind = start_inds(0);
					tip_all_found = true;
				}
				else
				{
					start_ind = start_inds(0);
				}
			}
			else
			{
				arma::uvec start_inds = arma::find(seg_flags_avail>0, 1, "first");
				if (start_inds.n_elem == 0)
				{
					start_ind = -1;
					break;
				}
				start_ind = start_inds(0);
			}
			seg_flags_avail(start_ind) = 0;
		}

		//Randomly shuffle segment ID for better visualization in 3D software such as cloudcompare
		arma::uvec seg_ID_shuffle = randomShuffle(seg_IDs);

		//Attach segment ID to final column
		pcd_mat = pcd_mat.cols(span(0, 2));
		pcd_mat = arma::join_rows(pcd_mat, arma::conv_to<arma::vec>::from(seg_ID_shuffle));

		util_.WriteMatToAscii(foutpath, pcd_mat);

	}

	//approximately find tip points from point clouds; tip points will serve as the starting seeds of region growing
	//input: arma::mat& pcd_mat, const double tip_resolution
	//output: arma::uvec
	//tip_resolution is the size of voxelization
	arma::uvec FastFindTip(arma::mat& pcd_mat, const double tip_resolution)
	{
		std::cout << "1. Finding tips" << std::endl;
		arma::mat pcd_xyz = pcd_mat.cols(span(0, 2));

		rowvec pcd_min=arma::min(pcd_xyz, 0);
		rowvec pcd_max = arma::max(pcd_xyz, 0);
		uvec pcd_n_even= arma::conv_to<arma::uvec>::from(arma::floor((pcd_max - pcd_min) / tip_resolution)+1);
		arma::umat pcd_even = arma::conv_to<arma::umat>::from(arma::floor((pcd_xyz - arma::repmat(pcd_min, pcd_xyz.n_rows, 1)) / tip_resolution));

		arma::umat pcd_unq;
		util_.UniqueByRow(pcd_even, pcd_n_even, pcd_unq);



		arma::umat pcd_unq_shrink = pcd_unq.rows(arma::find(pcd_unq.col(4) > 1));
		arma::mat pcd_unq_sub = conv_to<arma::mat>::from(pcd_unq_shrink.cols(span(0, 2)));


		Eigen::MatrixXd eig_xyz = Eigen::Map<Eigen::MatrixXd>(pcd_unq_sub.memptr(), pcd_unq_sub.n_rows, pcd_unq_sub.n_cols);

		std::vector<int> tip_inds;
		nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> tindex(3, eig_xyz, 10);
		tindex.index->buildIndex();


		std::vector<std::pair<size_t, double>> ret_matches;

		nanoflann::SearchParams params;
		params.sorted = false;

		//note that nanoflann kd tree uses l2 distance as radius
		double radius = 2.0*2.0;
		arma::uvec dummy;
		int iter = 1;
		for (int i = 0; i < pcd_unq_sub.n_rows; i++)
		{
			if (((i+1)*100 / pcd_unq_sub.n_rows) >= 25*iter)
			{
				cout << "searching progress: " << (i*1.0 / pcd_unq_sub.n_rows) << std::endl;
				iter++;
			}
			arma::uvec nn_idx;
			arma::mat query_point = pcd_unq_sub.row(i);
			RadSearch(tindex, dummy, query_point, radius, nn_idx);
			arma::mat nn_vecs = (pcd_unq_sub.rows(nn_idx) - arma::repmat(pcd_unq_sub.row(i), nn_idx.n_elem, 1));
			//double nn_curvature_degree = arma::norm(arma::mean(arma::normalise(nn_vecs, 2, 1)), 1);
			double nn_curvature_degree = abs(arma::accu(arma::mean(arma::normalise(nn_vecs, 2, 1), 0)));
			if (nn_curvature_degree > curv_degree_)//tip points are considered with large curvature
			{
				tip_inds.push_back(pcd_unq_shrink.at(i, 3));
			}
		}
		std::vector<uword> vec_convert(tip_inds.begin(), tip_inds.end());
		arma::uvec tip_index(vec_convert.data(), vec_convert.size());//cast arma mat from eigen mat
		arma::mat tip_mat = pcd_xyz.rows(tip_index);
		util_.WriteMatToAscii("tips.txt", tip_mat);

		return tip_index;
	}
	
	//wrapper for the FastFindTip and SegmentCylinders functions
	//input: arma::mat& pcd_mat, const double tip_resolution, const double rg_step, const double rg_mingap,string fout
	//output: a text file (x,y,z,regionID)
	int RegionGrowingSegment(arma::mat& pcd_mat, const double tip_resolution, const double rg_step, const double rg_mingap,string fout)
	{
		tip_resolution_ = tip_resolution;
		rg_step_ = rg_step;
		rg_mingap_ = rg_mingap;


		tip_index_ = FastFindTip(pcd_mat, tip_resolution);
		if (tip_index_.n_elem < 1)
		{
			std::cout << "No tips was found, return..." << std::endl;
			//return -1;
		}
		SegmentCylinders(pcd_mat, rg_step, rg_mingap, fout);
		return 0;
	}

	//wrapper for the FastFindTip and SegmentCylinders functions
	//input: string fpath, const double approx_resolution
	//output: a text file (x,y,z,regionID)
	int RegionGrowingSegment(string fpath, const double approx_resolution, string fout)
	{
		arma::mat pcd_mat;
		if (util_.ReadAsciiToMat(fpath, pcd_mat))
		{
			return -1;
		}
		//util_.WriteMat("test1.txt", pcd_mat);
		return RegionGrowingSegment(pcd_mat, max(0.1, approx_resolution * 5.0), max(0.05, approx_resolution * 5.0), max(0.02, approx_resolution * 3.0), fout);
	}



private:
	double tip_resolution_=0.05;
	double rg_step_= 0.05;
	double rg_mingap_ = 0.02;

	double curv_degree_ = 0.8;
	double width_ratio_limit_ = 0.8;
	double direction_change_limit_ = 0.5; //60 degree
	double block_size_ = rg_step_*5; //for speed up, limit the search range

	arma::uvec tip_index_;
	
	CylUtil util_;
};