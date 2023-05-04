// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include "OpenCVApplication.h"

wchar_t* projectPath;
void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		std::cout << fname << std::endl;
		Mat src;
		src = imread(fname);
		imshow("opened image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	std::cout << folderName << std::endl;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		std::cout << fname << std::endl;
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i, j) = (r + g + b) / 3;
			}
		}

		imshow("original image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

//------------------------------------------------------------------P R O J E C T-------------------------------------------------------------------------

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
	std::vector<double> coord;
	int label;
} DataPoint;

typedef struct {
	std::vector<RowCSV> rows;
	long number_of_rows;
}DataCSV;

double euclideanDistanceDrawSignature(const Point2f& a, const Point2f& b) {
	double distance = 0.0;
	distance += pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
	return sqrt(distance);
}

double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	double distance = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		distance += pow(a[i] - b[i], 2);
	}
	return sqrt(distance);
}

// k = nr of neighbours
int knn_classify(const std::vector<DataPoint> dataset, const std::vector<double> input, int k) {
	std::vector<std::pair<double, int>> distances;
	for (size_t i = 0; i < dataset.size(); ++i) {
		double distance = euclideanDistance(dataset[i].coord, input);
		distances.emplace_back(distance, dataset[i].label);
	}
	sort(distances.begin(), distances.end());
	for (std::pair<double, int> distance : distances) {
		std::cout << "("<< distance.first << ", " << distance.second << ")  ";
	}
	std::cout << std::endl;
	std::vector<int> counts(dataset.size());
	for (int i = 0; i < k; ++i) {
		int label = distances[i].second;
		counts[label]++;
	}
	int maxCount = 0;
	int maxLabel = -1;
	for (int i = 0; i < dataset.size(); ++i) {
		if (counts[i] > maxCount) {
			maxCount = counts[i];
			maxLabel = i;
		}
	}
	return maxLabel;
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

std::vector<DataPoint> readDataSetPoint(char* file_name)
{
	std::ifstream file(file_name);
	std::vector<DataPoint> dataPoints;
	// Check if file is opened successfully
	if (!file.is_open()) {
		std::cerr << "Error opening dataset file" << std::endl;
		return dataPoints;
	}

	std::string line, field;

	// Read each line of the file
	std::getline(file, line);//reading the first line with headers and ignoring it
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		DataPoint data;
		std::getline(ss, field, ',');//get label
		data.label = std::stoi(field);
		for (int i = 0; i < 24; i++) {
			std::getline(ss, field, ',');//get coord
			data.coord.push_back(std::stod(field));
		}
		dataPoints.push_back(data);
	}

	// Print the data to the console
	/*for (int i = 0; i < dataPoints.size(); i++) {
		std::cout << "x: " << dataPoints.at(i).coord.x << ", y: " << dataPoints.at(i).coord.y;
		std::cout << ", label: " << dataPoints.at(i).label << std::endl;
	}*/

	file.close(); // Close the file
	return dataPoints;
}

std::tuple<Mat, Mat> splitImage(Mat& image, cv::Point split_coord, bool isHorizontal)
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

//the next two functions does the same thing
//Point2f massCenter(Mat& image)
//{
//	Moments m = moments(image, true);
//	// Compute center of mass
//	Point2f center(m.m10 / m.m00, m.m01 / m.m00);
//	//std::cout << "Mass center coordinates: " << center.x << ", " << center.y << std::endl;
//	//circle(image, center, 5, Scalar(255, 255, 255), FILLED);
//	return center;
//}

Point2f massCenterSimple(Mat_<uchar> image)
{
	double m00 = 0, m10 = 0, m01 = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image(i,j) > 0) {
				m00 += 1;
				m10 += j;
				m01 += i;
			}
		}
	}
	// Compute center of mass
	Point2f center(m10 / m00, m01 / m00);
	//std::cout << "Mass center coordinates manual: " << center.x << ", " << center.y << std::endl;
	//circle(image, center, 5, Scalar(255, 255, 255), FILLED);
	return center;
}

//NOT WORKING
Point2f massCenterPressure(Mat_<uchar> image, int add_rows, int add_cols, DataCSV data)
{
	double m00 = 0, m10 = 0, m01 = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image(i, j) > 0) {
				for (int k = 0; k < data.number_of_rows; k++) {
					if ((int)data.rows.at(k).x == (i + add_rows) && (int)data.rows.at(k).y == (j + add_cols)){
						m00 += data.rows.at(k).pressure;
						m10 += j * data.rows.at(k).pressure;
						m01 += i * data.rows.at(k).pressure;
						break;
					}
				}
			}
		}
	}
	// Compute center of mass
	Point2f center(m10 / m00, m01 / m00);
	std::cout << "Mass center coordinates manual: " << center.x << ", " << center.y << std::endl;
	//circle(image, center, 5, Scalar(255, 255, 255), FILLED);
	return center;
}

//this function cuts the rows and columns (from the beginning and the end of the image) that doesn't have any colored pixel. It fits the signature in a window
void cropImage(Mat& image)
{
	int rows_to_delete_from_start = 0;
	int rows_to_delete_from_end = 0;
	int cols_to_delete_from_start = 0;
	int cols_to_delete_from_end = 0;

	boolean ok = false;
	for (int i = 0; i < image.rows && ok == 0; i++) {
		for (int j = 0; j < image.cols && ok == 0; j++) {
			if (image.at<uchar>(i, j) > 0) {
				rows_to_delete_from_start = i;
				ok = 1;
			}
		}
	}
	if (ok == 0) {
		image = NULL;
	}
	ok = false;
	for (int i = image.rows - 1; i >= 0 && ok == 0; i--) {
		for (int j = 0; j < image.cols && ok == 0; j++) {
			if (image.at<uchar>(i, j) > 0) {
				rows_to_delete_from_end = image.rows - i - 1;
				ok = 1;
			}
		}
	}
	ok = false;
	for (int j = 0; j < image.cols && ok == 0; j++) {
		for (int i = 0; i < image.rows && ok == 0; i++) {
			if (image.at<uchar>(i, j) > 0) {
				cols_to_delete_from_start = j;
				ok = 1;
			}
		}
	}
	ok = false;
	for (int j = image.cols - 1; j >= 0 && ok == 0; j--) {
		for (int i = 0; i < image.rows && ok == 0; i++) {
			if (image.at<uchar>(i, j) > 0) {
				cols_to_delete_from_end = image.cols - j - 1;
				ok = 1;
				break;
			}
		}
	}

	// Compute the new size of the image
	int new_height = image.rows - rows_to_delete_from_start - rows_to_delete_from_end;
	int new_width = image.cols - cols_to_delete_from_start - cols_to_delete_from_end;

	// Crop the image
	Mat cropped_image = image.rowRange(rows_to_delete_from_start, image.rows - rows_to_delete_from_end).colRange(cols_to_delete_from_start, image.cols - cols_to_delete_from_end);

	// Resize the cropped image to the desired size
	resize(cropped_image, cropped_image, Size(new_width, new_height));
	image = cropped_image;
}

std::vector<Point2f> splittingCoordHorizontal(Mat& image, Point2f init_mass_center)
{
	std::vector<Point2f> feature_points;
	std::tuple<Mat, Mat> splitted_img = splitImage(image, init_mass_center, true);
	Point2f mass_center_top = massCenterSimple(std::get<0>(splitted_img));
	Point2f mass_center_bottom = massCenterSimple(std::get<1>(splitted_img));
	int difference = std::get<0>(splitted_img).rows;
	feature_points.push_back(mass_center_top);
	feature_points.push_back(Point2f(mass_center_bottom.x, mass_center_bottom.y + difference));
	//imshow("top", std::get<0>(splitted_img));
	//imshow("bottom", std::get<1>(splitted_img));


	std::tuple<Mat, Mat> splitted_img_top = splitImage(std::get<0>(splitted_img), mass_center_top, true);
	Point2f mass_center_top_top = massCenterSimple(std::get<0>(splitted_img_top));
	Point2f mass_center_top_bottom = massCenterSimple(std::get<1>(splitted_img_top));
	int difference2 = std::get<0>(splitted_img_top).rows;
	mass_center_top_bottom.y += difference2;
	feature_points.push_back(mass_center_top_top);
	feature_points.push_back(mass_center_top_bottom);
	//imshow("top_top", std::get<0>(splitted_img_top));
	//imshow("top_bottom", std::get<1>(splitted_img_top));

	std::tuple<Mat, Mat> splitted_img_bottom = splitImage(std::get<1>(splitted_img), mass_center_bottom, true);
	Point2f mass_center_bottom_top = massCenterSimple(std::get<0>(splitted_img_bottom));
	Point2f mass_center_bottom_bottom = massCenterSimple(std::get<1>(splitted_img_bottom));
	int difference3 = difference + std::get<0>(splitted_img_bottom).rows;
	mass_center_bottom_bottom.y += difference3;
	feature_points.push_back(mass_center_bottom_top);
	feature_points.push_back(mass_center_bottom_bottom);
	//imshow("bottom_top", std::get<0>(splitted_img_bottom));
	//imshow("bottom_bottom", std::get<1>(splitted_img_bottom));

	return feature_points;
}

std::vector<Point2f> splittingCoordVertical(Mat& image, Point2f init_mass_center, DataCSV data)
{
	std::vector<Point2f> feature_points;
	std::tuple<Mat, Mat> splitted_img = splitImage(image, init_mass_center, false);
	Mat matLeft= std::get<0>(splitted_img);
	Mat matRight = std::get<1>(splitted_img);
	Point2f mass_center_left = massCenterSimple(matLeft);
	Point2f mass_center_right = massCenterSimple(matRight);
	feature_points.push_back(mass_center_left);
	feature_points.push_back(Point2f(mass_center_right.x + matLeft.cols, mass_center_right.y));
	//imshow("left", matLeft);
	//imshow("right", matRight);


	std::tuple<Mat, Mat> splitted_img_left = splitImage(matLeft, mass_center_left, false);
	Mat matLeftLeft = std::get<0>(splitted_img_left);
	Mat matLeftRight = std::get<1>(splitted_img_left);
	Point2f mass_center_left_left = massCenterSimple(matLeftLeft);
	Point2f mass_center_left_right = massCenterSimple(matLeftRight);
	mass_center_left_right.x += matLeftLeft.cols;
	feature_points.push_back(mass_center_left_left);
	feature_points.push_back(mass_center_left_right);
	//imshow("1", matLeftLeft);
	//imshow("2", matLeftRight);

	std::tuple<Mat, Mat> splitted_img_right = splitImage(matRight, mass_center_right, false);
	Mat matRightLeft = std::get<0>(splitted_img_right);
	Mat matRightRight = std::get<1>(splitted_img_right);
	Point2f mass_center_right_left = massCenterSimple(matRightLeft);
	Point2f mass_center_right_right = massCenterSimple(matRightRight);
	mass_center_right_left.x += matLeft.cols;
	mass_center_right_right.x += matLeft.cols + matRightLeft.cols;
	feature_points.push_back(mass_center_right_left);
	feature_points.push_back(mass_center_right_right);
	//imshow("3", matRightLeft);
	//imshow("4", matRightRight);

	return feature_points;
}

std::vector<Point2f> featureExtraction(Mat& image, DataCSV data)
{
	std::vector<Point2f> feature_points;
	std::vector<Point2f> left_feature_points = splittingCoordHorizontal(image, massCenterSimple(image));
	std::vector<Point2f> right_feature_points = splittingCoordVertical(image, massCenterSimple(image), data);
	feature_points.insert(feature_points.begin(), left_feature_points.begin(), left_feature_points.end());
	feature_points.insert(feature_points.end(), right_feature_points.begin(), right_feature_points.end());
	return feature_points;
}

void drawSignature(Mat& image, DataCSV& data)
{
	int treshhold = 150.0;
	for (int i = 0;i < data.number_of_rows-1; i++)
	{
		cv::Point2f pointStart(data.rows.at(i).x, data.rows.at(i).y);
		cv::Point2f pointEnd(data.rows.at(i+1).x, data.rows.at(i+1).y);
		//std::cout << pointStart.x << " " << pointStart.y << " " << pointEnd.x << " " << pointEnd.y << std::endl;
		//circle(image, pointStart, 5, Scalar(255, 255, 255), FILLED);
		if(euclideanDistanceDrawSignature(pointStart, pointEnd) < treshhold)
			line(image, pointStart, pointEnd, Scalar(255, 255, 255), 3);
	}
	
}

void drawFeaturePoints(Mat& image, std::vector<Point2f> feature_extraction_points)
{
	for (Point2f feature_point : feature_extraction_points)
	{
		circle(image, feature_point, 5, Scalar(125, 125, 125), FILLED);
	}
}

Mat_<uchar> getCenteredWindow(DataCSV& points)
{
	// Create an image to draw the points on
	double minX = FLT_MAX, minY = FLT_MAX, maxX = 0.0, maxY = 0.0;
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
	return img;
}

void showSignature(char* fname)
{
	
	DataCSV points = readCSV(fname);
	Mat_<uchar> img = getCenteredWindow(points);

	// Draw the signature centered
	drawSignature(img, points);

	// Show the image
	imshow("Signature", img);
	waitKey(0);
	
}

void testShowSignature()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		showSignature(fname);
	}
}

void signatureFeatureExtraction(char* fname) {
	DataCSV points = readCSV(fname);
	Mat_<uchar> img = getCenteredWindow(points);

	// Draw the signature centered
	drawSignature(img, points);
	std::vector<Point2f> feature_extraction_points = featureExtraction(img, points);
	std::cout << feature_extraction_points.size() << std::endl;
	for (Point2f feature_point : feature_extraction_points)
	{
		std::cout << feature_point.x << " " << feature_point.y << std::endl;
	}
	drawFeaturePoints(img, feature_extraction_points);

	// Show the image with feature extraction points
	imshow("Signature with feature extraction points", img);
	waitKey(0);
}

void testSignatureFeatureExtraction() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		signatureFeatureExtraction(fname);
	}
}

std::vector<std::string> matching_table = { "Naggy",  "Kovues", "Toth", "Szabo", "Honot", "Varga", "Kiv", "Molnar", "Wemeth", "Fwba", "Balogh", "Pepp", "Taracs", "Fahasz", "Lakatos", "Meszavos", "Olah", "Simon", "Hm", "Fehete"};

void normalizeCoordinates(std::vector<Point2f>& points)
{
	float maxX = 0.0f;
	float maxY = 0.0f;
	for (Point2f actualPoint : points) {
		if (actualPoint.x > maxX) {
			maxX = actualPoint.x;
		}
		if (actualPoint.y > maxY) {
			maxY = actualPoint.y;
		}
	}
	std::cout << maxX << ", " << maxY << std::endl;
	for (Point2f& actualPoint : points) {
		actualPoint.x /= maxX;
		actualPoint.y /= maxY;
	}
	for (Point2f actualPoint : points) {
		std::cout << actualPoint.x << ", " << actualPoint.y << std::endl;
	}
}


void classifySignature(char* fname)
{
	DataCSV points = readCSV(fname);
	//fname
	Mat_<uchar> img = getCenteredWindow(points);

	// Draw the signature centered
	drawSignature(img, points);
	std::vector<Point2f> feature_extraction_points = featureExtraction(img, points);
	normalizeCoordinates(feature_extraction_points);
	std::vector<double> feature_extraction_points_double;
	for (Point2f feature_extraction_point : feature_extraction_points)
	{
		//std::cout << feature_extraction_point.x << " " << feature_extraction_point.y << std::endl;
		feature_extraction_points_double.push_back(feature_extraction_point.x);
		feature_extraction_points_double.push_back(feature_extraction_point.y);
	}
	drawFeaturePoints(img, feature_extraction_points);
	std::vector<DataPoint> dataset = readDataSetPoint("D:\\ANUL3\\PI\\1.1.1.1.1.1.Proiect\\OpenCVApplication-VS2022_OCV460_basic\\DataSet.csv");
	for (double point : feature_extraction_points_double) {
		std::cout << point << "  ";
	}
	std::cout << std::endl;
	int label = knn_classify(dataset, feature_extraction_points_double, 23);
	std::cout << fname << std::endl;
	std::cout << "USER" << label << std::endl;
	if(label>=0 && label<20)
		std::cout << "Signature belongs to: "<< matching_table[label - 1] << std::endl;
	imshow("Signature with feature extraction points", img);
	waitKey(0);
}

void testClassifySignature()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		classifySignature(fname);
	}
}

//a nu se deschide fisierul .csv pana nu se termina de scris toate datele
void writeDataSet(char* fname, int label) {
	DataCSV points = readCSV(fname);
	Mat_<uchar> img = getCenteredWindow(points);
	// Draw the signature centered
	drawSignature(img, points);
	std::vector<Point2f> feature_extraction_points = featureExtraction(img, points);
	normalizeCoordinates(feature_extraction_points);

	//WRITE IN CSV FILE
	std::fstream fout;
	fout.open("C:/Users/stef_/Desktop/Cursuri/PI/Project/OpenCVApplication/Output.csv", std::ios::out | std::ios::app);
	// Insert the data to file 
	fout << label << ", ";
	for (Point2f feature_point : feature_extraction_points)
	{
		fout << feature_point.x << ", " << feature_point.y << ", ";
	}
	fout << "\n";
	// close the file
	fout.close();
}

void buildDataSet() {
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "csv");
	int label = 0;
	while (fg.getNextAbsFile(fname))
	{
		//std::cout << fname << std::endl;
		if (fname[88] == '_') {
			label = fname[87] - '0';
		}
		else {
			label = (fname[87] - '0') * 10 + (fname[88] - '0');
		}
		writeDataSet(fname, label);
		//Mat src;
		//src = imread(fname);
		//imshow(fg.getFoundFileName(), src);
		//if (waitKey() == 27) //ESC pressed
		//	break;
	}
}

//------------------------------------------------------------------M A I N-------------------------------------------------------------------------

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
		printf(" 4 - Show signature\n");
		printf(" 5 - Show signature + feature extraction\n");
		printf(" 6 - Classify signature\n");
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
			testShowSignature();
			break;
		case 5:
			testSignatureFeatureExtraction();
			break;
		case 6:
			testClassifySignature();
			break;
		}
	} while (op != 0);
	//buildDataSet();
	return 0;
}
