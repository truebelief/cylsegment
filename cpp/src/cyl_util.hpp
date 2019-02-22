#pragma once
//#include <boost/algorithm/string.hpp>
#include <armadillo>


using namespace std;
using namespace arma;

class CylUtil
{
public:

	//write arma::mat to file
	template <typename K>
	void WriteMat(string foutpath, K& mats)
	{
		std::cout << "Writing point clouds to an ascii file..." << std::endl;
		std::ofstream file(foutpath, ios::out | ios::trunc);
		if (file.is_open())
		{
			file << mats << '\n';
		}
		file.close();
	}

	//write arma::mat to file
	void WriteMatToAscii(string save_path, arma::mat& mats, string delimiter = " ", string header = "")
	{
		std::cout << "Writing point clouds to an ascii file..." << std::endl;
		ofstream myfile;
		myfile.open(save_path);
		if (header.length()) {
			myfile << header << endl;
		}
		for (size_t i = 0; i < mats.n_rows; ++i)
		{
			arma::rowvec each_row = mats.row(i);
			for (size_t j = 0; j < mats.n_cols - 1; ++j)
			{
				myfile << each_row(j) << delimiter;
			}
			myfile << each_row(mats.n_cols - 1) << std::endl;
		}
		myfile.close();
	}

	char DetermineDelimiter(string file_path)
	{
		int header_num=DetectHeaders(file_path);

		ifstream myfile;
		myfile.open(file_path);
		string testline;
		for (; header_num; --header_num)
		{
			getline(myfile, testline);
		}
		getline(myfile, testline);
		double x;
		char delimiter;
		istringstream ist = istringstream(testline);
		ist >> std::noskipws >> x >> delimiter;
		myfile.close();
		return delimiter;
	}

	int DetectHeaders(string file_path)
	{
		ifstream myfile;
		myfile.open(file_path);
		string testline;
		getline(myfile, testline);
		int headers = 0;
		istringstream ist = istringstream(testline);

		double peek_value;
		while (!(ist>> peek_value))
		{
			getline(myfile, testline);
			ist = istringstream(testline);
			++headers;
		}


		return headers;
	}

	//read file and load arma::mat 
	int ReadAsciiToMat(string file_path,arma::mat& mats, bool auto_detect=true, char delimiter=' ', int headers = 0)
	{
		std::cout << "Reading ascii point clouds..." << std::endl;
		ifstream f;
		f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		try {
			f.open(file_path);
		}
		catch (std::system_error& e) {
			std::cout << "Cannot open point cloud file." << std::endl;
			std::cout << e.code().message() << std::endl;
			return -1;
		}

		if (auto_detect)
		{
			delimiter = DetermineDelimiter(file_path);
			headers = DetectHeaders(file_path);
		}

		std::string header_str;
		ifstream myfile;
		myfile.open(file_path);
		for (;headers; --headers)
		{
			std::getline(myfile, header_str);
		};

		std::vector<double> values;
		std::string line;
		int rows = 0;
		while (std::getline(myfile, line)) {
			std::stringstream lineStream(line);
			std::string cell;
			while (std::getline(lineStream, cell, delimiter)) {
				values.push_back(std::stod(cell));
			}
			++rows;
		}
		
		mats = arma::mat(values.data(), values.size() / rows, rows).t();
		myfile.close();
		return 0;
	}

	//convert std::map to arma::umat
	template <typename K, typename V>
	void MapToMat(std::map<K, std::pair<V, V>> const& maps, arma::umat& mats)
	{
		for (auto const &ent1 : maps)
		{
			urowvec each_row(3);
			each_row(0) = ent1.first;
			each_row(1) = ent1.second.first;
			each_row(2) = ent1.second.second;
			mats.insert_rows(mats.n_rows, each_row);
		}
	}

	//convert arma:;mat to std::vector 2D, row-wisely
	vector<vector<double>> MatToVector2D(arma::mat& mats)
	{
		vector<vector<double>> vec2d(mats.n_rows);
		for (size_t i = 0; i < mats.n_rows; ++i)
		{
			vec2d[i] = arma::conv_to<vector<double>>::from(mats.row(i));
		};
		return vec2d;
	}

	//convert vector 2D to arma:;mat, row-wisely
	arma::mat Vector2DToMat(vector<vector<double>>& vec2d)
	{
		arma::mat mats(vec2d.size(), vec2d[0].size());
		for (size_t i = 0; i < mats.n_rows; ++i)
		{
			mats.row(i) = arma::mat(vec2d[i]);
		};
		return mats;
	}

	//similar to the Matlab function [U,~,id2]=unique(A);
	//find unique value (U) of A, and also the index id2 so that U(id2)==A
	//performance limited by Armadillo's sort function (have to sort twice to get both sorted values and indices)
	//input: arma::uvec& keys
	//output: arma::uvec& keys_unq, arma::uvec& keys_unq_index2
	//index is 0-based
	void UniqueIndex(arma::uvec& keys, arma::uvec& keys_unq, arma::uvec& keys_unq_index2)
	{
		arma::uvec key_sorted = arma::sort(keys, "ascend");
		arma::uvec key_sorted_indices = arma::sort_index(keys, "ascend");
		key_sorted.insert_rows(0, -1 * arma::ones<arma::uvec>(1));
		arma::uvec locs = arma::find(arma::diff(key_sorted)>0);

		keys_unq_index2 = arma::zeros<uvec>(keys.n_elem);
		arma::uvec keys_unq_ind2 = arma::zeros<uvec>(keys.n_elem);

		keys_unq = key_sorted(locs(span(0, locs.n_elem - 1)) + 1);
		for (int i = 0; i<locs.n_elem; i++)
		{
			keys_unq_ind2(span(locs[i], keys.n_elem - 1)) += 1;
		}
		keys_unq_index2(key_sorted_indices) = keys_unq_ind2 - 1;
	}

	//similar to the Matlab function [U,id1]=unique(A), and also count redundant numbers for each unique value
	//find unique value (U) of A, and also the index id1 so that A(id1)==U
	//input: std::vector<size_t>& keys
	//output: arma::umat& mat_unq
	//index is 0-based
	void UniqueByRow(std::vector<size_t>& keys, arma::umat& mat_unq)
	{
		////use map to get sorted unique value + unique index + unique count
		map<size_t, std::pair<size_t, size_t>> m;

		for (auto it = keys.begin(); it != keys.end(); ++it)
		{
			m[*it].first = it - keys.begin();
			m[*it].second = m[*it].second + 1;
		}
		MapToMat(m, mat_unq);
	}

	//similar to the Matlab function [U,id1]=unique(A,'rows'), and also count redundant numbers for each unique value
	//input arma::mat is limited to non-negative integer, there will be a problem to have float/double precision
	//see https://stackoverflow.com/questions/6684573/floating-point-keys-in-stdmap/6684830#6684830
	//input: arma::umat& mats, arma::uvec& shape
	//output: arma::umat& mat_unq
	//index is 0-based
	void UniqueByRow(arma::umat& mats, arma::uvec& shape, arma::umat& mat_unq)
	{
		if (mats.n_rows > 3)
		{
			mats = mats.t();
		}
		arma::uvec inds = arma::sub2ind(arma::size(shape[0], shape[1], shape[2]), mats);//must be 3xC elements, i.e. 3 rows
		std::vector<size_t> keys(inds.memptr(), inds.memptr() + inds.n_elem);

		arma::umat unq;
		UniqueByRow(keys, unq);
		arma::umat unq_keys = unq.col(0);

		arma::umat unq_ijk = arma::ind2sub(arma::size(shape[0], shape[1], shape[2]), unq_keys).t();
		mat_unq = arma::join_rows(unq_ijk, unq.cols(span(1, unq.n_cols - 1)));
	}

	//similar to the Matlab function [U,id1]=unique(A,'rows'), and also count redundant numbers for each unique value
	//input: arma::umat& mats
	//output: arma::umat& mat_unq
	//index is 0-based
	void UniqueByRow(arma::umat& mats, arma::umat& mat_unq)
	{
		arma::uvec shape = arma::conv_to<arma::uvec>::from(arma::max(mats, 0));
		shape += 1;
		if (mats.n_rows > 3)
		{
			mats = mats.t();
		}
		UniqueByRow(mats, shape, mat_unq);
	}


	//similar to the Matlab function [U,id1]=unique(A), and also count redundant numbers for each unique value
	//input: arma::uvec& vecs
	//output: arma::umat& mat_unq
	//index is 0-based
	void UniqueByRow(arma::uvec& vecs, arma::umat& mat_unq)
	{
		map<size_t, std::pair<size_t, size_t>> m;
		for (auto it = vecs.begin(); it != vecs.end(); ++it)
		{
			m[*it].first = it - vecs.begin();
			m[*it].second = m[*it].second + 1;
		}
		MapToMat(m, mat_unq);
	}

	//remove rows based on indices from arma::mat or arma::vec
	//input: arma::mat mats, arma::uvec& inds
	//output: none
	template <typename K>
	void RemoveRows(K& mats, arma::uvec& inds)
	{
		arma::uvec inds_sorted = arma::unique(inds);
		for (int i = inds_sorted.n_elem - 1; i >= 0; --i)
		{
			mats.shed_row(inds_sorted(i));
		}
	}

	//flatten std::vector from 2d to 1d
	template <typename K>
	void Flatten2DVector(std::vector<std::vector<K>>& vec2d, std::vector<K>& vec1d)
	{
		vec1d.clear();
		for (auto it = vec2d.begin(); it != vec2d.end(); ++it)
		{
			vec1d.insert(vec1d.end(), (*it).begin(), (*it).end());
		}
	}

	//calculate quantile interpolation, similar to Matlab
	//https://www.mathworks.com/help/stats/quantile.html
	double Quantile(arma::vec& vecs, double quant)
	{
		arma::vec vec_sorted = arma::sort(vecs, "ascend");

		int n = vec_sorted.n_elem;
		arma::vec quants = arma::linspace<arma::vec>(1, n, n);
		quants = (quants - 0.5) / n;

		if (quant < quants(0) + (1e-8))
		{
			return vec_sorted(0);
		}
		if (quant > quants(n - 1) - (1e-8))
		{
			return vec_sorted(n - 1);
		}
		arma::uvec quant_is = arma::find((quants - quant) > 0, 1, "first");
		int quant_i = quant_is(0);
		double v1 = as_scalar(vec_sorted[quant_i - 1]);
		double v2 = as_scalar(vec_sorted[quant_i]);
		return v1 + (quant - quants(quant_i - 1)) / (quants(quant_i) - quants(quant_i - 1))*(v2 - v1);

	}
};