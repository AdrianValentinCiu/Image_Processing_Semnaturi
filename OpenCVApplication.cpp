// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>

struct DataCSV {
	double x;
	double y;
};

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
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
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
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

std::vector<DataCSV> readCSV()
{
	std::ifstream file("test_points.csv");
	std::vector<DataCSV> data;
	// Check if file is opened successfully
	if (!file.is_open()) {
		std::cerr << "Error opening file" << std::endl;
		return data;
	}

	std::string line, field;

	// Read each line of the file
	while (std::getline(file, line)) {
		std::stringstream ss(line);

		// Parse the x and y values from each line
		DataCSV row;
		std::getline(ss, field, ',');
		row.x = std::stod(field);
		std::getline(ss, field, ',');
		row.y = std::stod(field);

		data.push_back(row);
	}

	// Print the data to the console
	for (const auto& row : data) {
		std::cout << "x: " << row.x << ", y: " << row.y << std::endl;
	}

	file.close(); // Close the file
	return data;
}

void drawPoints(const std::vector<DataCSV>& data, Mat& image)
{
	std::vector<cv::Point2f> points;
	for (const auto& row : data) {
		points.emplace_back(row.x, row.y);
	}
	for (const auto& point : points)
	{
		circle(image, point, 5, Scalar(255, 255, 255), FILLED);
	}
	std::cout << "done draw data" << "\n";
}

void showDataFromCSV()
{
	
	std::vector<DataCSV> points = readCSV();

	// Create an image to draw the points on
	Mat_<uchar> img = Mat::zeros(Size(1500, 720), CV_8UC1);

	// Draw the points on the image
	drawPoints(points, img);

	// Show the image
	imshow("Points_from_Signature", img);
	waitKey(0);
	
}

void testShowDataFromCSV()
{
	showDataFromCSV();
}


int main()
{
	//readCSV();
	showDataFromCSV();
	/*
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
	*/
	return 0;
}
