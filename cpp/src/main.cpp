#include "stdafx.h"
#include "cyl_segmenter.hpp"

using namespace std;


int main(int argc, char *argv[])
{

	
	string filename = argv[1];
	string fout = argv[2];
	double approx_res = std::stod(argv[3]);

	CylSegment cylSeg;
	cylSeg.RegionGrowingSegment(filename,approx_res, fout);
	std::cout << "Finished." << std::endl;

	return 0;
}
