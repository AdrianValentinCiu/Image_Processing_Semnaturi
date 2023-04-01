// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include "OpenCVApplication.h"

wchar_t* projectPath;

typedef struct {
	double x;
	double y;
	long long  timestamp;
	double pressure;
	double fingerarea;
	double velocityx;
	double velocityy;
	double accelx;
	double accely;
	double accelz;
}RowCSV;

typedef struct {
	std::vector<RowCSV> rows;
	long number_of_rows;
}DataCSV;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		std::cout << fname << std::endl;
		Mat src;
		src = imread(fname);
		imshow("opened image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	std::cout << folderName << std::endl;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		std::cout << fname << std::endl;
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("original image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

DataCSV readCSV(char* file_name)
{
	std::ifstream file(file_name);
	DataCSV data;
	data.number_of_rows = 0;
	// Check if file is opened successfully
	if (!file.is_open()) {
		std::cerr << "Error opening file" << std::endl;
		return data;
	}

	std::string line, field;

	// Read each line of the file
	std::getline(file, line);
	while (std::getline(file, line)) {
		std::stringstream ss(line);

		// Parse each line
		RowCSV row;
		std::getline(ss, field, ',');
		row.x = std::stod(field);
		std::getline(ss, field, ',');
		row.y = std::stod(field);
		std::getline(ss, field, ',');
		row.timestamp = (long long)std::stol(field);
		std::getline(ss, field, ',');
		row.pressure = std::stod(field);
		std::getline(ss, field, ',');
		row.fingerarea = std::stod(field);
		std::getline(ss, field, ',');
		row.velocityx = std::stod(field);
		std::getline(ss, field, ',');
		row.velocityy = std::stod(field);
		std::getline(ss, field, ',');
		row.accelx = std::stod(field);
		std::getline(ss, field, ',');
		row.accely = std::stod(field);
		std::getline(ss, field, ',');
		row.accelz = std::stod(field);


		data.rows.push_back(row);
		data.number_of_rows++;
	}

	// Print the data to the console
	/*for (int i = 0; i < data.number_of_rows; i++) {
		std::cout << "x: " << data.rows.at(i).x << ", y: " << data.rows.at(i).y;
		std::cout << ", timestamp: " << data.rows.at(i).timestamp << ", pressure: " << data.rows.at(i).pressure;
		std::cout << ", fingerarea: " << data.rows.at(i).fingerarea << ", velocityx: " << data.rows.at(i).velocityx;
		std::cout << ", velocityy: " << data.rows.at(i).velocityy << ", accelx: " << data.rows.at(i).accelx;
		std::cout << ", accley: " << data.rows.at(i).accely << ", accelz: " << data.rows.at(i).accelz << std::endl;
	}*/

	file.close(); // Close the file
	return data;
}

int calculateEuclidianDistance(cv::Point pointStart, cv::Point pointEnd)
{
	int distance = sqrt((pointEnd.x - pointStart.x) * (pointEnd.x - pointStart.x) + (pointEnd.y - pointStart.y) * (pointEnd.y - pointStart.y));
	return distance;
}

std::tuple<Mat, Mat> split_image(Mat & image, cv::Point split_coord, bool isHorizontal)
{
	if (isHorizontal) 
	{
		int split_point = split_coord.y;
		Rect roi1(0, 0, image.cols, split_point);
		Rect roi2(0, split_point, image.cols, image.rows - split_point);
		Mat top_half = image(roi1);
		Mat bottom_half = image(roi2);
		return std::make_tuple(top_half, bottom_half);
	}
	else
	{
		int split_point = split_coord.x;
		Rect roi1(0, 0, split_point, image.rows);
		Rect roi2(split_point, 0, image.cols - split_point, image.rows);
		Mat left_half = image(roi1);
		Mat right_half = image(roi2);
		return std::make_tuple(left_half, right_half);
	}
}

Point2f mass_center(Mat& image)
{
	Moments m = moments(image, true);
	// Calculate mass center
	Point2f center(m.m10 / m.m00, m.m01 / m.m00);
	std::cout << "Mass center coordinates: " << center.x << ", " << center.y << std::endl;
	//circle(image, center, 5, Scalar(255, 255, 255), FILLED);
	return center;
}

std::vector<Point2f> splitting_coord(Mat& image, Point2f init_mass_center, bool isHorizonatal)
{
	std::vector<Point2f> feature_points;
	std::tuple<Mat, Mat> splitted_img = split_image(image, init_mass_center, isHorizonatal);
	Point2f mass_center_top = mass_center(std::get<0>(splitted_img));
	Point2f mass_center_bottom = mass_center(std::get<1>(splitted_img));
	feature_points.push_back(mass_center_top);
	feature_points.push_back(mass_center_bottom);
	imshow("top", std::get<0>(splitted_img));
	imshow("bottom", std::get<1>(splitted_img));


	std::tuple<Mat, Mat> splitted_img_top = split_image(std::get<0>(splitted_img), mass_center_top, isHorizonatal);
	Point2f mass_center_top_top = mass_center(std::get<0>(splitted_img_top));
	Point2f mass_center_top_bottom = mass_center(std::get<1>(splitted_img_top));
	feature_points.push_back(mass_center_top_top);
	feature_points.push_back(mass_center_top_bottom);
	imshow("top_top", std::get<0>(splitted_img_top));
	imshow("top_bottom", std::get<1>(splitted_img_top));

	std::tuple<Mat, Mat> splitted_img_bottom = split_image(std::get<1>(splitted_img), mass_center_bottom, isHorizonatal);
	Point2f mass_center_bottom_top = mass_center(std::get<0>(splitted_img_bottom));
	Point2f mass_center_bottom_bottom = mass_center(std::get<1>(splitted_img_bottom));
	feature_points.push_back(mass_center_bottom_top);
	feature_points.push_back(mass_center_bottom_bottom);
	imshow("bottom_top", std::get<0>(splitted_img_bottom));
	imshow("bottom_bottom", std::get<1>(splitted_img_bottom));

	return feature_points;
}

std::vector<Point2f> feature_extraction(Mat& image)
{
	std::vector<Point2f> feature_points;
	std::vector<Point2f> left_feature_points = splitting_coord(image, mass_center(image), true);
	std::vector<Point2f> right_feature_points = splitting_coord(image, mass_center(image), false);
	feature_points.insert(feature_points.begin(), left_feature_points.begin(), left_feature_points.end());
	feature_points.insert(feature_points.end(), right_feature_points.begin(), right_feature_points.end());
	return feature_points;
}

void drawPoints(DataCSV& data, Mat& image)
{
	int treshhold = 150;
	for (int i = 0;i < data.number_of_rows-1; i++)
	{
		cv::Point pointStart(data.rows.at(i).x, data.rows.at(i).y);
		cv::Point pointEnd(data.rows.at(i+1).x, data.rows.at(i+1).y);
		//std::cout << pointStart.x << " " << pointStart.y << " " << pointEnd.x << " " << pointEnd.y << std::endl;
		//circle(image, pointStart, 5, Scalar(255, 255, 255), FILLED);
		if(calculateEuclidianDistance(pointStart, pointEnd) < treshhold)
			line(image, pointStart, pointEnd, Scalar(255, 255, 255), 3);
	}
	
}

void draw_feature_points(Mat img, std::vector<Point2f> feature_extraction_points)
{
	for (Point2f feature_point : feature_extraction_points)
	{
		circle(img, feature_point, 5, Scalar(125, 125, 125), FILLED);
	}
}

void showDataFromCSV(char* fname)
{
	
	DataCSV points = readCSV(fname);

	// Create an image to draw the points on
	int minX = MAXINT, minY = MAXINT, maxX = 0, maxY = 0;
	for (int i = 0; i < points.number_of_rows - 1; i++)
	{
		if (minX > points.rows.at(i).x)
			minX = points.rows.at(i).x;
		if (minY > points.rows.at(i).y)
			minY = points.rows.at(i).y;
		if (maxX < points.rows.at(i).x)
			maxX = points.rows.at(i).x;
		if (maxY < points.rows.at(i).y)
			maxY = points.rows.at(i).y;
	}
	maxX -= minX;
	maxY -= minY;
	//std::cout << maxX << " " << minX << " " << maxY << "  " << minY << std::endl;
	for (int i = 0; i < points.number_of_rows - 1; i++)
	{
		points.rows.at(i).x -= minX;
		points.rows.at(i).y -= minY;
	}
	Mat_<uchar> img = Mat::zeros(Size(maxX, maxY), CV_8UC1);

	// Draw the points on the image
	drawPoints(points, img);
	std::vector<Point2f> feature_extraction_points = feature_extraction(img);
	for (Point2f feature_point : feature_extraction_points)
	{
		std::cout << feature_point.x << " " << feature_point.y << std::endl;
	}
	draw_feature_points(img, feature_extraction_points);
	// Show the image
	imshow("Points_from_Signature", img);
	waitKey(0);
	
}

void testShowDataFromCSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		showDataFromCSV(fname);
	}
}


int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Basic image opening...\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Color to Gray\n");
		printf(" 4 - Show data from csv file\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testColor2Gray();
				break;
			case 4:
				testShowDataFromCSV();
				break;
		}
	}
	while (op!=0);

	return 0;
}
